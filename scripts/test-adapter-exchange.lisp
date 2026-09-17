(load (merge-pathnames "load-mlx.lisp" *load-truename*))
(load (merge-pathnames "../tests/adapter-exchange-suite.lisp" *load-truename*))
(uiop:symbol-call :docling-exchange-tests :run-tests)
