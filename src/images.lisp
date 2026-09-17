;;; Resize geometry and image markers adapted from Transformers 5.16.1 Idefics3
;;; (Hugging Face contributors, Apache-2.0); see THIRD-PARTY.md. Native orchestration
;;; and public API are cl-docling. This is the explicit PIL backend contract.
(in-package #:cl-docling)
(export '(image-processor make-image-processor preprocess-images image-prompt
          image-processor-tile-size image-processor-longest-edge image-processor-image-seq-len))

(defstruct (image-processor (:constructor %make-image-processor))
  (tile-size 512 :read-only t) (longest-edge 2048 :read-only t)
  (image-seq-len 64 :read-only t) (split-images t :read-only t))

(defun make-image-processor (&key (tile-size 512) (longest-edge 2048)
                                 (image-seq-len 64) (split-images t))
  "Explicit Idefics3 PIL policy: RGB8 Lanczos, rescale 1/255, mean/std 0.5.
No automatic processor/config guessing. Defaults match the pinned SmolDocling assets."
  (dolist (pair (list (list tile-size "tile-size") (list longest-edge "longest-edge")
                     (list image-seq-len "image-seq-len")))
    (apply #'require-positive-integer pair))
  (unless (and (<= tile-size 512) (<= longest-edge 2048)
               (zerop (mod longest-edge tile-size)) (<= 1 (/ longest-edge tile-size) 4)
               (<= image-seq-len 1024) (member split-images '(nil t)))
    (invalid-layout "Unsupported processor policy: tile ~S, longest ~S, tokens ~S, split ~S"
                    tile-size longest-edge image-seq-len split-images))
  (%make-image-processor :tile-size tile-size :longest-edge longest-edge
                        :image-seq-len image-seq-len :split-images split-images))

(defvar *image-library* nil)
(defun ensure-image-library ()
  (or *image-library*
      (setf *image-library*
            (cffi:load-foreign-library
             (or (uiop:getenv "DOCLING_IMAGE_LIBRARY")
                 (asdf:system-relative-pathname "cl-docling"
                  #+darwin ".build/native/libdocling_images.dylib"
                  #-darwin ".build/native/libdocling_images.so"))))))
(cffi:defcfun ("dd_read_png" %read-png) :pointer (filename :string) (error :pointer) (capacity :size))
(cffi:defcfun ("dd_image_free" %image-free) :void (image :pointer))
(cffi:defcfun ("dd_image_width" %image-width) :int (image :pointer))
(cffi:defcfun ("dd_image_height" %image-height) :int (image :pointer))
(cffi:defcfun ("dd_resize" %resize) :pointer (image :pointer) (width :int) (height :int))
(cffi:defcfun ("dd_normalize_tile" %normalize-tile) :int
  (image :pointer) (x :int) (y :int) (side :int) (output :pointer))
(cffi:defcfun ("dd_png_version" %png-version) :string)
(cffi:defcfun ("dd_image_live_count" %image-live-count) :size)

(defun read-native-png (file)
  (unless (or (pathnamep file) (stringp file)) (invalid-layout "Expected a PNG pathname, got ~S" file))
  (let ((name (namestring file)))
    (when (find #\Null name) (invalid-layout "PNG pathname contains NUL"))
    (cffi:with-foreign-object (message :char 512)
      (let ((image (%read-png name message 512)))
        (when (cffi:null-pointer-p image)
          (invalid-layout "~A: ~A" name (cffi:foreign-string-to-lisp message)))
        image))))

(defun resize-native-image (image width height)
  (let ((result (%resize image width height)))
    (when (cffi:null-pointer-p result) (invalid-layout "Native resize allocation/size failure (~Dx~D)" width height))
    result))

(defun image-resize-plan (processor height width)
  "Return first H/W, second H/W and tile rows/cols. Match Python float/truncation."
  (let* ((edge (image-processor-longest-edge processor))
         (side (image-processor-tile-size processor))
         (ratio (/ (float width 1d0) height)))
    (if (>= width height)
        (setf width edge height (truncate (/ edge ratio)))
        (setf height edge width (truncate (* edge ratio))))
    (if (= width edge) (when (oddp height) (incf height)) (when (oddp width) (incf width)))
    (setf height (max height 1) width (max width 1))
    (let ((h height) (w width))
      (if (image-processor-split-images processor)
          (let ((ratio (/ (float width 1d0) height)))
            (if (>= width height)
                (setf w (* side (ceiling width side)) h (* side (ceiling (truncate (/ w ratio)) side)))
                (setf h (* side (ceiling height side)) w (* side (ceiling (truncate (* h ratio)) side)))))
          (setf h side w side))
      (values height width h w
              (if (or (> h side) (> w side)) (/ h side) 0)
              (if (or (> h side) (> w side)) (/ w side) 0)))))

(defun image-prompt (processor rows cols)
  "Expanded image marker string; not tokenization or a chat-template engine."
  (unless (and (typep processor 'image-processor)
               (integerp rows) (integerp cols)
               (or (= rows cols 0) (and (<= 1 rows 4) (<= 1 cols 4))))
    (invalid-layout "Expected a processor and a 0/0 or 1..4 image grid"))
  (with-output-to-string (out)
    (flet ((tokens () (dotimes (i (image-processor-image-seq-len processor))
                       (write-string "<image>" out))))
      (dotimes (row rows)
        (dotimes (col cols)
          (format out "<fake_token_around_image><row_~D_col_~D>" (1+ row) (1+ col))
          (tokens))
        (terpri out))
      (unless (zerop rows) (terpri out))
      (write-string "<fake_token_around_image><global-img>" out)
      (tokens)
      (write-string "<fake_token_around_image>" out))))

(defun fill-image-row (image plan side output output-mask offset)
  (destructuring-bind (h1 w1 h2 w2 rows cols) plan
    (let ((first (resize-native-image image w1 h1)))
      (unwind-protect
           (let ((second (resize-native-image first w2 h2)))
             (unwind-protect
                  (let ((index offset) (tile-values (* 3 side side)))
                    (flet ((emit (im x y)
                             (unless (= 1 (%normalize-tile im x y side
                                           (cffi:inc-pointer output (* index tile-values 4))))
                               (invalid-layout "Invalid native tile region"))
                             (fill output-mask 1 :start (* index side side) :end (* (1+ index) side side))
                             (incf index)))
                      (if (zerop rows)
                          (emit second 0 0)
                          (progn
                            (dotimes (row rows) (dotimes (col cols) (emit second (* col side) (* row side))))
                            (let ((global (resize-native-image second side side)))
                              (unwind-protect (emit global 0 0) (%image-free global)))))))
               (%image-free second)))
        (%image-free first)))))

(defun preprocess-images (processor files &key (max-output-bytes (* 512 1024 1024)))
  "One PNG per batch row; returns FP32 (B T 3 H W), U8 (B T H W), (rows cols) per row.
T is the largest tile count. Missing tiles contain normalized-space zeros and mask 0.
Outputs own Lisp storage; all native image buffers are freed before return/error.
This does not feed variable tiles to the M2 model, tokenize text, or decode JPEG/PDF."
  (unless (and (typep processor 'image-processor) (listp files) (<= 1 (length files) 8))
    (invalid-layout "Expected an image processor and 1..8 PNG paths (one image per row)"))
  (require-positive-integer max-output-bytes "max-output-bytes")
  (ensure-image-library)
  (let ((images nil) (side (image-processor-tile-size processor)))
    (unwind-protect
         (progn
           (dolist (file files) (push (read-native-png file) images))
           (setf images (nreverse images))
           (let* ((plans (mapcar (lambda (im) (multiple-value-list
                                              (image-resize-plan processor (%image-height im) (%image-width im)))) images))
                  (grids (mapcar (lambda (plan) (nthcdr 4 plan)) plans))
                  (count (loop for (rows cols) in grids maximize (1+ (* rows cols))))
                  (batch (length images)) (n (* batch count side side)))
             (when (> (* n 13) max-output-bytes)
               (invalid-layout "Processor output needs ~D bytes; limit is ~D" (* n 13) max-output-bytes))
             (let ((pixels (make-array (* 3 n) :element-type 'single-float :initial-element 0f0))
                   (mask (make-array n :element-type '(unsigned-byte 8) :initial-element 0)))
               (cffi:with-pointer-to-vector-data (output pixels)
                 (loop for im in images for plan in plans for row from 0
                       do (fill-image-row im plan side output mask (* row count))))
               (values (make-array (list batch count 3 side side) :element-type 'single-float :displaced-to pixels)
                       (make-array (list batch count side side) :element-type '(unsigned-byte 8) :displaced-to mask)
                       grids))))
      (mapc #'%image-free images))))
