(load (merge-pathnames "load-mlx.lisp" *load-truename*))
(handler-case
    (progn (asdf:load-system "cl-docling/pipeline")
           (load (merge-pathnames "../tests/training-inputs-integration.lisp" *load-truename*)))
  (error (e) (format *error-output* "~&FAIL: ~A~%" e) (uiop:quit 1)))
