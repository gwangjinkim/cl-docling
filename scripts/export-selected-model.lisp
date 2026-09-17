;;; Export only the supplied, previously selected native adapter; never train.
(load (merge-pathnames "experiment-common.lisp" *load-truename*))
(let ((destination (first (uiop:command-line-arguments))))
  (when (probe-file destination) (error "Export destination must be new."))
  (tb:with-resource (model (docling:load-document-model (uiop:getenv "DOCLING_MODEL") :device :gpu))
    (docling:load-document-adapter model (uiop:getenv "DOCLING_BENCHMARK_ADAPTER"))
    (docling:merge-document-lora model)
    (tb:save-pretrained model destination)))
(format t "~&PASS: native selected adapter merged and exported as an ordinary model.~%")
