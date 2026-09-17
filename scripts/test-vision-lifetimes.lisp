(load (merge-pathnames "load-mlx.lisp" *load-truename*))
(load (asdf:system-relative-pathname "cl-docling" "tests/vision-lifetimes.lisp"))
(uiop:symbol-call :docling-lifetime-tests :run-tests)
