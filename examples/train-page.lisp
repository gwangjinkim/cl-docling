;;; Two native SGD steps on a public synthetic page; no quality claim or upload.
(load (merge-pathnames "../scripts/load-mlx.lisp" *load-truename*))
(asdf:load-system "cl-docling/pipeline")
(let* ((destination (first (uiop:command-line-arguments)))
       (checkpoint (or (uiop:getenv "DOCLING_MODEL")
                       (asdf:system-relative-pathname "cl-docling" ".build/smoldocling/")))
       (device (ecase (intern (string-upcase (or (uiop:getenv "TB_DEVICE") "cpu")) :keyword)
                 (:cpu :cpu) (:gpu :gpu)))
       (case (with-open-file (s (asdf:system-relative-pathname "cl-docling" "tests/fixtures/supervision/cases.sexp"))
               (let ((*read-eval* nil)) (first (read s))))))
  (unless (and destination (plusp (length destination)) (not (probe-file destination)))
    (error "Supply a new adapter output directory as the only argument."))
  (tb:with-resource (model (docling:load-document-model checkpoint :device device))
    (multiple-value-bind (ids labels attention features tiles answer-start)
        (docling:prepare-image-training-example model
          (asdf:system-relative-pathname "cl-docling" (getf case :image)) (getf case :answer)
          :task (getf case :task) :pad-to (getf case :pad-to))
      (unwind-protect
           (progn
             (format t "~&~D tiles; ~D prompt tokens; ~D positions including padding.~%"
                     tiles answer-start (array-dimension ids 1))
             (docling:make-document-lora model :rank 2 :alpha 4 :seed 17)
             (tb:with-resource (optimizer (tb:make-sgd :learning-rate 0.001))
               (dotimes (step 2)
                 (format t "~&Step ~D answer loss: ~,8F~%" step
                   (docling:document-train-step model optimizer ids features
                     :labels labels :attention-mask attention :tile-count tiles))))
             (docling:save-document-adapter model destination)
             (format t "~&Saved local PEFT adapter: ~A~%No held-out quality improvement is implied.~%" destination))
        (tb:dispose features)))))
