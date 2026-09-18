;;; Native all-tile input preparation only; no Python, forward, gradients or update.
(load (merge-pathnames "experiment-common.lisp" *load-truename*))

(defun context-flat (array)
  (map-into (make-array (array-total-size array)) #'identity
            (make-array (array-total-size array) :element-type (array-element-type array) :displaced-to array)))

(destructuring-bind (input-path output) (uiop:command-line-arguments)
  (let ((inputs (uiop:ensure-directory-pathname input-path))
        (checkpoint (uiop:ensure-directory-pathname (or (uiop:getenv "DOCLING_MODEL") (error "Set DOCLING_MODEL."))))
        (records nil) (remaining nil))
    (when (probe-file output) (error "Output must be new."))
    (tb:with-resource (backend (tb:make-backend :device :cpu))
      (tb:with-resource (model (docling:load-document-model checkpoint :backend backend))
        (with-open-file (config-stream (merge-pathnames "config.json" checkpoint))
         (let ((config (yason:parse config-stream)))
          (with-open-file (in (merge-pathnames "report.json" inputs))
          (dolist (case (gethash "inventory" (yason:parse in)))
            (when (equal "approved" (gethash "decision" case))
              (multiple-value-bind (ids labels mask pixels tiles start)
                  (docling:prepare-image-training-input
                   model (merge-pathnames (gethash "image" case) inputs)
                   (uiop:read-file-string (merge-pathnames (gethash "target" case) inputs) :external-format :utf-8))
                (push (experiment-object
                       "name" (pathname-name (gethash "image" case))
                       "ids" (context-flat ids) "labels" (context-flat labels) "attention" (context-flat mask)
                       "answer_start" start "tiles" tiles "pixel_shape" (coerce (array-dimensions pixels) 'vector)
                       "image_token_id" (gethash "image_token_id" config)
                       "eos_token_id" (aref (tb:encode-text model "<end_of_utterance>" :add-special-tokens nil) 0)
                       "image_seq_len" 64
                       "context_limit" (min 8192 (gethash "max_position_embeddings"
                                                       (gethash "text_config" config)))) records)
                (format t "~&~A: ~D prompt + ~D answer/EOS = ~D; ~D tiles.~%"
                        (gethash "image" case) start (- (array-dimension ids 1) start) (array-dimension ids 1) tiles))))))))
      (setf remaining (getf (tb:backend-memory backend) :handles))
      (unless (zerop remaining) (error "Leaked native handles.")))
    (experiment-write output (experiment-object "cases" (coerce (nreverse records) 'vector)
                                                "remaining_handles" remaining "sbcl" (lisp-implementation-version)) t)))
