;; Run from repository root after following docs/native-vision.md setup.
(load (merge-pathnames "../scripts/load-mlx.lisp" *load-truename*))
(let ((directory (or (uiop:getenv "DOCLING_MODEL") ".build/smoldocling/"))
      (device (if (equal (uiop:getenv "TB_DEVICE") "gpu") :gpu :cpu)))
  (tb:with-resource (backend (tb:make-backend :device device))
    (tb:with-resource (model (docling:load-vision-model directory :backend backend))
      (tb:with-resource (pixels (tb:tensor-from-array
                                 backend (make-array '(1 3 512 512) :element-type 'single-float
                                                                          :initial-element 0.0)))
        (tb:with-resource (features (docling:encode-vision model pixels))
          (format t "~D vision/connector tensors; blank normalized tile -> ~S on ~A.~%"
                  (docling:vision-weight-count model) (tb:tensor-shape features) device))))))
