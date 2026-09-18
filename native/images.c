#define _POSIX_C_SOURCE 200809L
/* Native PNG + RGB8 Lanczos kernel; no Python runtime.
 * Lanczos coefficients and fixed-point rounding adapted from Pillow 12.3.0
 * src/libImaging/Resample.c. See licenses/Pillow.txt for MIT-CMU attribution.
 * The PNG wrapper and buffer API are part of cl-docling (MIT).
 */
#include <png.h>
#include <math.h>
#include <stdint.h>
#include <stdatomic.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define MAX_EDGE 16384
#define MAX_PIXELS (16u * 1024u * 1024u)
#define PRECISION 22

typedef struct { unsigned char *pixels; int w, h; } dd_image;
static _Atomic size_t live_images = 0;
size_t dd_image_live_count(void) { return atomic_load(&live_images); }
static int valid_size(int w, int h) {
    return w > 0 && h > 0 && w <= MAX_EDGE && h <= MAX_EDGE &&
           (size_t)w * h <= MAX_PIXELS;
}
void dd_image_free(dd_image *im) {
    if (im) { free(im->pixels); free(im); atomic_fetch_sub(&live_images,1); }
}
int dd_image_width(const dd_image *im) { return im->w; }
int dd_image_height(const dd_image *im) { return im->h; }
static dd_image *image_new(int w, int h) {
    if (!valid_size(w,h)) return NULL;
    dd_image *im = calloc(1,sizeof(*im));
    if (!im) return NULL;
    im->w = w; im->h = h; im->pixels = malloc((size_t)w*h*3);
    if (!im->pixels) { free(im); return NULL; }
    atomic_fetch_add(&live_images,1);
    return im;
}

/* All longjmp state lives on the heap; no Lisp callback or foreign unwind. */
typedef struct { dd_image *im; png_bytep *rows; char *error; size_t capacity; } png_state;
static void png_failure(png_structp png, png_const_charp message) {
    png_state *s = png_get_error_ptr(png);
    snprintf(s->error,s->capacity,"PNG: %s",message);
    png_longjmp(png,1);
}
static void png_warning_ignore(png_structp png, png_const_charp message) {
    (void)png; (void)message;
}
static int reject_animation(png_structp png, png_unknown_chunkp chunk) {
    if (memcmp(chunk->name,"acTL",4)==0) png_error(png,"Animated PNG is unsupported");
    return PNG_HANDLE_CHUNK_AS_DEFAULT;
}
dd_image *dd_read_png(const char *filename, char *error, size_t capacity) {
    struct stat st;
    if (stat(filename,&st) || !S_ISREG(st.st_mode) || st.st_size > 64*1024*1024) {
        snprintf(error,capacity,"Expected a regular PNG file of at most 64 MiB"); return NULL;
    }
    FILE *f = fopen(filename,"rb");
    if (!f) { snprintf(error,capacity,"Cannot open PNG"); return NULL; }
    png_state *s = calloc(1,sizeof(*s));
    if (!s) { fclose(f); snprintf(error,capacity,"Out of memory"); return NULL; }
    s->error=error; s->capacity=capacity;
    png_structp png=png_create_read_struct(PNG_LIBPNG_VER_STRING,s,png_failure,png_warning_ignore);
    png_infop info=png ? png_create_info_struct(png) : NULL;
    if (!png || !info) {
        if (png) png_destroy_read_struct(&png,NULL,NULL);
        free(s); fclose(f); snprintf(error,capacity,"Cannot allocate PNG reader"); return NULL;
    }
    if (setjmp(png_jmpbuf(png))) {
        dd_image_free(s->im); free(s->rows); free(s);
        png_destroy_read_struct(&png,&info,NULL); fclose(f); return NULL;
    }
    png_set_user_limits(png,MAX_EDGE,MAX_EDGE);
    png_set_chunk_malloc_max(png,1024*1024);
    png_set_chunk_cache_max(png,128);
    png_set_keep_unknown_chunks(png,PNG_HANDLE_CHUNK_ALWAYS,(png_const_bytep)"acTL",1);
    png_set_read_user_chunk_fn(png,NULL,reject_animation);
    png_set_crc_action(png,PNG_CRC_ERROR_QUIT,PNG_CRC_ERROR_QUIT);
    png_init_io(png,f); png_read_info(png,info);
    int w=(int)png_get_image_width(png,info), h=(int)png_get_image_height(png,info);
    int color=png_get_color_type(png,info), depth=png_get_bit_depth(png,info);
    if (depth != 8 || (color != PNG_COLOR_TYPE_RGB && color != PNG_COLOR_TYPE_RGBA &&
                       color != PNG_COLOR_TYPE_GRAY && color != PNG_COLOR_TYPE_GRAY_ALPHA))
        png_error(png,"Only 8-bit RGB, RGBA, grayscale and gray-alpha are supported");
    if (!valid_size(w,h)) png_error(png,"Image exceeds 16 megapixels / 16384 edge limit");
    /* Match PIL.convert(RGB): no gamma/ICC conversion, alpha blending or EXIF rotation. */
    if (color & PNG_COLOR_MASK_ALPHA) png_set_strip_alpha(png);
    if (color == PNG_COLOR_TYPE_GRAY || color == PNG_COLOR_TYPE_GRAY_ALPHA) png_set_gray_to_rgb(png);
    png_set_interlace_handling(png); png_read_update_info(png,info);
    if (png_get_rowbytes(png,info) != (size_t)w*3) png_error(png,"Unexpected RGB row size");
    s->im=image_new(w,h); s->rows=malloc((size_t)h*sizeof(*s->rows));
    if (!s->im || !s->rows) png_error(png,"Out of memory");
    for (int y=0; y<h; y++) s->rows[y]=s->im->pixels+(size_t)y*w*3;
    png_read_image(png,s->rows); png_read_end(png,info);
    dd_image *result=s->im; free(s->rows); free(s);
    png_destroy_read_struct(&png,&info,NULL); fclose(f); return result;
}

int dd_write_png_crop(const char *source, const char *destination,
                      int left, int top, int right, int bottom,
                      int *output_width, int *output_height,
                      char *error, size_t capacity) {
    if (!source || !destination || !output_width || !output_height || !error || capacity == 0 ||
        left < 0 || top < 0 || right > 499 || bottom > 499 || left >= right || top >= bottom) {
        if (error && capacity) snprintf(error,capacity,"Expected nonempty quantized picture bounds in [0,499]");
        return 0;
    }
    dd_image *image=dd_read_png(source,error,capacity);
    if (!image) return 0;
    int x0=(int)((int64_t)left*image->w/500), y0=(int)((int64_t)top*image->h/500);
    int x1=(int)((int64_t)right*image->w/500), y1=(int)((int64_t)bottom*image->h/500);
    if (x0>=x1 || y0>=y1) {
        snprintf(error,capacity,"Quantized picture is empty at source resolution");
        dd_image_free(image); return 0;
    }
    int fd=open(destination,O_WRONLY|O_CREAT|O_EXCL,0600);
    if (fd<0) { snprintf(error,capacity,"Cannot exclusively create picture asset"); dd_image_free(image); return 0; }
    FILE *file=fdopen(fd,"wb");
    if (!file) { close(fd); unlink(destination); snprintf(error,capacity,"Cannot open picture asset stream"); dd_image_free(image); return 0; }
    png_state state={0}; state.error=error; state.capacity=capacity;
    png_structp png=png_create_write_struct(PNG_LIBPNG_VER_STRING,&state,png_failure,png_warning_ignore);
    png_infop info=png ? png_create_info_struct(png) : NULL;
    if (!png || !info) {
        if (png) png_destroy_write_struct(&png,NULL);
        fclose(file); unlink(destination); dd_image_free(image);
        snprintf(error,capacity,"Cannot allocate PNG writer"); return 0;
    }
    if (setjmp(png_jmpbuf(png))) {
        png_destroy_write_struct(&png,&info); fclose(file); unlink(destination); dd_image_free(image); return 0;
    }
    int width=x1-x0, height=y1-y0;
    png_init_io(png,file);
    png_set_IHDR(png,info,(png_uint_32)width,(png_uint_32)height,8,PNG_COLOR_TYPE_RGB,
                 PNG_INTERLACE_NONE,PNG_COMPRESSION_TYPE_DEFAULT,PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png,info);
    for (int y=y0;y<y1;y++) png_write_row(png,image->pixels+((size_t)y*image->w+x0)*3);
    png_write_end(png,info); png_destroy_write_struct(&png,&info);
    if (fclose(file)!=0) { unlink(destination); dd_image_free(image); snprintf(error,capacity,"Cannot close picture asset"); return 0; }
    dd_image_free(image); *output_width=width; *output_height=height; return 1;
}

static double sinc(double x) {
    if (x == 0.0) return 1.0;
    x *= 3.14159265358979323846; return sin(x)/x;
}
typedef struct { int size; int *bounds; int32_t *weights; } coefficients;
static void coefficients_free(coefficients *k) { free(k->bounds); free(k->weights); }
static int coefficients_make(int input, int output, coefficients *k) {
    double scale=(double)input/output, fs=scale < 1.0 ? 1.0 : scale, support=3.0*fs;
    k->size=(int)ceil(support)*2+1;
    k->bounds=malloc((size_t)output*2*sizeof(int));
    k->weights=calloc((size_t)output*k->size,sizeof(int32_t));
    double *work=malloc((size_t)k->size*sizeof(double));
    if (!k->bounds || !k->weights || !work) { free(work); coefficients_free(k); return 0; }
    double inv=1.0/fs;
    for (int i=0;i<output;i++) {
        double center=(i+0.5)*scale, total=0;
        int start=(int)(center-support+0.5), end=(int)(center+support+0.5);
        if (start<0) start=0;
        if (end>input) end=input;
        int count=end-start;
        for (int j=0;j<count;j++) {
            double x=(j+start-center+0.5)*inv;
            work[j]=(-3.0<=x && x<3.0) ? sinc(x)*sinc(x/3) : 0.0;
            total+=work[j];
        }
        for (int j=0;j<count;j++) {
            double v=total != 0.0 ? work[j]/total : work[j];
            k->weights[(size_t)i*k->size+j]=(int32_t)((v<0 ? -0.5 : 0.5)+v*(1<<PRECISION));
        }
        k->bounds[2*i]=start; k->bounds[2*i+1]=count;
    }
    free(work); return 1;
}
static unsigned char clip(int64_t sum) {
    /* Avoid implementation-defined right shift for negative signed integers. */
    if (sum<=0) return 0;
    sum >>= PRECISION; return sum>255 ? 255 : (unsigned char)sum;
}
static dd_image *resize_axis(const dd_image *im, int output, int horizontal) {
    int input=horizontal ? im->w : im->h;
    dd_image *out=image_new(horizontal ? output : im->w,horizontal ? im->h : output);
    if (!out) return NULL;
    if (input==output) { memcpy(out->pixels,im->pixels,(size_t)im->w*im->h*3); return out; }
    coefficients k={0};
    if (!coefficients_make(input,output,&k)) { dd_image_free(out); return NULL; }
    for (int y=0;y<out->h;y++) for (int x=0;x<out->w;x++) {
        int i=horizontal ? x : y, start=k.bounds[2*i], count=k.bounds[2*i+1];
        for (int c=0;c<3;c++) {
            int64_t sum=1<<(PRECISION-1);
            for (int j=0;j<count;j++) {
                size_t p=horizontal ? (size_t)y*im->w+start+j : (size_t)(start+j)*im->w+x;
                sum+=(int64_t)im->pixels[p*3+c]*k.weights[(size_t)i*k.size+j];
            }
            out->pixels[((size_t)y*out->w+x)*3+c]=clip(sum);
        }
    }
    coefficients_free(&k); return out;
}
dd_image *dd_resize(const dd_image *im, int width, int height) {
    if (!valid_size(width,height) || !valid_size(width,im->h)) return NULL;
    dd_image *tmp=resize_axis(im,width,1);
    if (!tmp) return NULL;
    dd_image *out=resize_axis(tmp,height,0); dd_image_free(tmp); return out;
}
/* One FFI call per tile, not per pixel. Output is normalized contiguous CHW FP32. */
int dd_normalize_tile(const dd_image *im, int x0, int y0, int side, float *out) {
    if (side<=0 || x0<0 || y0<0 || side>im->w-x0 || side>im->h-y0) return 0;
    float table[256];
    for (int i=0;i<256;i++) {
        float scaled=(float)((double)i*(1.0/255.0));
        table[i]=(scaled-0.5f)/0.5f;
    }
    for (int c=0;c<3;c++) for (int y=0;y<side;y++) for (int x=0;x<side;x++)
        out[((size_t)c*side+y)*side+x]=table[im->pixels[((size_t)(y+y0)*im->w+x+x0)*3+c]];
    return 1;
}
const char *dd_png_version(void) { return png_get_libpng_ver(NULL); }
