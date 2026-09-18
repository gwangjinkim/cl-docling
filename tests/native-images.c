/* Standalone sanitizer gate. Reuses the real kernel, never Python. */
#include "../native/images.c"
#include <assert.h>
#include <unistd.h>
int main(int argc, char **argv) {
    assert(argc==2);
    char filename[4096], crop_path[4096], error[512];
    float output[3*16*16];
    for (int repeat=0;repeat<40;repeat++) {
        const char *good[]={"portrait","landscape","gray","alpha","gray-alpha","thin","downsample"};
        for (size_t i=0;i<sizeof(good)/sizeof(*good);i++) {
            snprintf(filename,sizeof(filename),"%s/%s.png",argv[1],good[i]);
            dd_image *im=dd_read_png(filename,error,sizeof(error)); assert(im);
            dd_image *resized=dd_resize(im,16,16); assert(resized);
            assert(dd_normalize_tile(resized,0,0,16,output));
            assert(!dd_normalize_tile(resized,1,0,16,output));
            assert(!dd_resize(im,0,16)); assert(!dd_resize(im,16385,1));
            dd_image_free(resized); dd_image_free(im);
        }
        snprintf(filename,sizeof(filename),"%s/portrait.png",argv[1]);
        snprintf(crop_path,sizeof(crop_path),"/tmp/cl-docling-native-crop-%ld.png",(long)getpid());
        int width=0,height=0;
        assert(dd_write_png_crop(filename,crop_path,0,0,250,250,&width,&height,error,sizeof(error)));
        assert(width==9 && height==18);
        assert(!dd_write_png_crop(filename,crop_path,0,0,250,250,&width,&height,error,sizeof(error)));
        dd_image *crop=dd_read_png(crop_path,error,sizeof(error)); assert(crop && crop->w==9 && crop->h==18);
        dd_image_free(crop); assert(unlink(crop_path)==0);
        assert(!dd_write_png_crop(filename,crop_path,0,0,0,250,&width,&height,error,sizeof(error)));
        const char *bad[]={"missing","unsupported-palette","unsupported-16bit","truncated","bad-crc","animated"};
        for (size_t i=0;i<sizeof(bad)/sizeof(*bad);i++) {
            snprintf(filename,sizeof(filename),"%s/%s.png",argv[1],bad[i]);
            assert(!dd_read_png(filename,error,sizeof(error)));
        }
        assert(dd_image_live_count()==0);
    }
    puts("PASS: 40 native sanitizer/crop cycles, successful/error paths; zero live images.");
    return 0;
}
