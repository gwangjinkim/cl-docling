;;; Fresh native process: old identity rejection, staged adapter, selected outputs.
(load (merge-pathnames "experiment-common.lisp" *load-truename*))

(destructuring-bind (base old-adapter staged-adapter output) (uiop:command-line-arguments)
  (when (probe-file output) (error "Output must be new."))
  (let ((device (ecase (intern (string-upcase (or (uiop:getenv "TB_DEVICE") "cpu")) :keyword)
                  (:cpu :cpu) (:gpu :gpu)))
        (rejection nil) (cases (make-hash-table :test 'equal)) (remaining nil))
    (tb:with-resource (backend (tb:make-backend :device device))
      (tb:with-resource (model (docling:load-document-model base :backend backend))
        (let ((before (getf (tb:backend-memory backend) :handles)))
          (handler-case (docling:load-document-adapter model old-adapter)
            (tb:compatibility-error (condition) (setf rejection (princ-to-string condition))))
          (unless (and rejection (search "base" (string-downcase rejection)))
            (error "Expected original base-identity rejection, got ~S" rejection))
          (unless (= before (getf (tb:backend-memory backend) :handles))
            (error "Rejected adapter leaked handles.")))
        (docling:load-document-adapter model staged-adapter)
        (unless (= 120 (length (docling:document-lora-parameters model)))
          (error "Expected all 120 selected adapter factors."))
        (dolist (name '("library" "pets" "fruit" "tea"))
          (let* ((expected-file (asdf:system-relative-pathname "cl-docling"
                                  (format nil "tests/fixtures/selection-results/final/adapted/~A.json" name)))
                 (expected (with-open-file (s expected-file)
                             (yason:parse s :json-arrays-as-vectors t)))
                 (file (asdf:system-relative-pathname "cl-docling"
                         (format nil "tests/fixtures/selection-pages/~A.png" name))))
            (multiple-value-bind (raw tokens reason) (docling:generate-image model file :max-new-tokens 512)
              (unless (and (equal raw (gethash "raw" expected))
                           (equalp tokens (gethash "tokens" expected))
                           (equal (string-downcase reason) (gethash "stop_reason" expected)))
                (error "Relocated selected output mismatch: ~A" name))
              (setf (gethash name cases) (experiment-object "tokens" tokens "raw" raw
                                        "stop_reason" (string-downcase reason) "matches_selected" yason:true))
              (format t "~&PASS ~A ~A: ~D exact tokens.~%" device name (length tokens))
              (finish-output)))))
      (setf remaining (getf (tb:backend-memory backend) :handles))
      (unless (zerop remaining) (error "Model disposal leaked handles.")))
    (experiment-write output (experiment-object
      "device" (string-downcase device) "original_rejection" rejection
      "adapter_factors" 120 "remaining_handles" remaining "cases" cases
      "lisp_version" (lisp-implementation-version)) t)))
