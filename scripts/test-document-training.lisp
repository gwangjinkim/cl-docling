(load (merge-pathnames "load-mlx.lisp" *load-truename*))
(load (merge-pathnames "../tests/document-training-suite.lisp" *load-truename*))
(uiop:symbol-call :docling-training-tests :run-tests)
