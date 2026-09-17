(load (merge-pathnames "../scripts/load-mlx.lisp" *load-truename*))
(let* ((root (asdf:system-source-directory "cl-docling"))
       (checkpoint (or (uiop:getenv "DOCLING_MODEL") (merge-pathnames ".build/smoldocling/" root)))
       (fixture (uiop:ensure-directory-pathname
                  (or (uiop:getenv "DOCLING_FIXTURE") (merge-pathnames ".build/model-real/" root))))
       (device (let ((name (or (uiop:getenv "TB_DEVICE") "cpu")))
                 (cond ((equal name "cpu") :cpu) ((equal name "gpu") :gpu)
                       (t (error "TB_DEVICE must be cpu or gpu"))))))
  (tb:with-resource (backend (tb:make-backend :device device))
    (tb:with-resource (model (docling:load-document-model checkpoint :backend backend))
      (let ((inputs (tbk:load-weight-file (merge-pathnames "reference.safetensors" fixture) backend)))
        (unwind-protect
             (let* ((host (tb:tensor-array (gethash "input_ids" inputs)))
                    (ids (make-array (array-dimensions host))))
               (dotimes (i (array-total-size host))
                 (setf (row-major-aref ids i) (round (row-major-aref host i))))
               (multiple-value-bind (tokens reason)
                   (docling:generate-document model ids (gethash "pixels" inputs) :max-new-tokens 8)
                 (format t "~&~D parameters on ~A; generated IDs: ~S~%Raw output: ~S~%Stop: ~A~%"
                         (length (tb:named-parameters model)) device tokens
                         (tb:decode-tokens model tokens) reason))
               (when (uiop:getenv "DOCLING_EXPORT")
                 (tb:save-pretrained model (uiop:getenv "DOCLING_EXPORT"))))
          (tbk:dispose-weights inputs))))))
