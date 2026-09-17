(load (merge-pathnames "load-mlx.lisp" *load-truename*))
(load (merge-pathnames "../tests/model.lisp" *load-truename*))
(uiop:symbol-call :docling-model-tests :run-tests)
