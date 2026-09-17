;;; Native PNG -> normalized tiles, masks, image markers. No model inference yet.
(require :asdf)
(let ((root (uiop:pathname-parent-directory-pathname
             (uiop:pathname-directory-pathname *load-truename*))))
  (load (merge-pathnames "scripts/load-images.lisp" root)))
(let* ((file (or (first (uiop:command-line-arguments))
                 (asdf:system-relative-pathname "cl-docling" "tests/fixtures/processor/portrait.png")))
       (processor (docling:make-image-processor)))
  (multiple-value-bind (pixels mask grids) (docling:preprocess-images processor (list file))
    (format t "~&Pixels: ~S; masks: ~S.~%" (array-dimensions pixels) (array-dimensions mask))
    (format t "Grid: ~S; image placeholders: ~D; expanded image marker characters: ~D.~%"
            grids (* (docling:image-processor-image-seq-len processor) (array-dimension pixels 1))
            (length (docling:image-prompt processor (caar grids) (cadar grids))))
    (format t "Preprocessing-only example. Use examples/generate-image.lisp for model output.~%")))
