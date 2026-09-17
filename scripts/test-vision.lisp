(load (merge-pathnames "load-mlx.lisp" *load-truename*))
(load (asdf:system-relative-pathname "cl-docling" "tests/vision.lisp"))
(uiop:symbol-call :docling-vision-tests :run-tests)
