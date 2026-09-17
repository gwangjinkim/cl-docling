;;; Frozen two-page public-PDF case study. No training or output repair.
(load (merge-pathnames "experiment-common.lisp" *load-truename*))
(asdf:load-system "cl-docling/pdf")
(destructuring-bind (pdf destination) (uiop:command-line-arguments)
  (let* ((output (uiop:ensure-directory-pathname destination))
         (root (asdf:system-source-directory "cl-docling"))
         (manifest (with-open-file (s (merge-pathnames "tests/fixtures/public-pdf/protocol.json" root))
                     (yason:parse s :json-arrays-as-vectors t)))
         (raster (merge-pathnames "raster/" output))
         (device (ecase (intern (string-upcase (or (uiop:getenv "TB_DEVICE") "gpu")) :keyword)
                   (:cpu :cpu) (:gpu :gpu))))
    (when (probe-file output) (error "Output must be new."))
    (ensure-directories-exist output)
    (docling:rasterize-pdf pdf raster :pages '(27 28) :dpi 144 :timeout-seconds 60)
    (tb:with-resource (backend (tb:make-backend :device device))
      (tb:with-resource (model (docling:load-document-model
                                (or (uiop:getenv "DOCLING_MODEL") (error "Set DOCLING_MODEL.")) :backend backend))
        (experiment-generate model (coerce (gethash "cases" manifest) 'list) raster output
                             "base" device (gethash "generation" manifest)))
      (unless (zerop (getf (tb:backend-memory backend) :handles)) (error "Model handle leak.")))
    (experiment-write (merge-pathnames "completion.json" output)
      (experiment-object "device" (string-downcase device) "remaining_handles" 0
                         "lisp_version" (lisp-implementation-version)) t)))
