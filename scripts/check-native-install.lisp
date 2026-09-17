;;; Bounded real-model installation check; no downloads, builds or Python runtime.
(load (merge-pathnames "experiment-common.lisp" *load-truename*))

(let* ((output (or (first (uiop:command-line-arguments)) (error "Supply a NEW JSON report path.")))
       (root (asdf:system-source-directory "cl-docling"))
       (checkpoint (uiop:ensure-directory-pathname
                     (or (uiop:getenv "DOCLING_MODEL") (merge-pathnames ".build/smoldocling/" root))))
       (device (ecase (intern (string-upcase (or (uiop:getenv "TB_DEVICE") "cpu")) :keyword)
                 (:cpu :cpu) (:gpu :gpu)))
       (expected (with-open-file (s (merge-pathnames "tests/fixtures/benchmark/python-cpu.json" root))
                   (gethash "native-page" (gethash "cases" (yason:parse s :json-arrays-as-vectors t)))))
       (page (merge-pathnames "examples/data/native-page.png" root))
       (systems (make-hash-table :test 'equal)) (libraries nil) (result nil) (remaining nil))
  (when (probe-file output) (error "Report must not already exist."))
  (dolist (name '("cl-docling" "cl-transformer-blocks" "cl-transformer-blocks/kernels"
                  "cffi" "yason" "babel" "alexandria" "trivial-features"
                  "trivial-garbage" "trivial-gray-streams"))
    (setf (gethash name systems)
          (experiment-object "version" (asdf:component-version (asdf:find-system name))
                             "source" (namestring (truename (asdf:system-source-directory name))))))
  (tb:with-resource (backend (tb:make-backend :device device))
    (unless (eq device (tb:backend-device backend)) (error "Requested device was not selected."))
    (tb:with-resource (model (docling:load-document-model checkpoint :backend backend))
      (multiple-value-bind (ids pixels tiles) (docling:prepare-image-input model page)
        (declare (ignore pixels))
        (unless (and (= tiles (gethash "tiles" expected))
                     (= (array-dimension ids 1) (length (gethash "prompt_ids" expected)))
                     (loop for i below (array-dimension ids 1)
                           always (= (aref ids 0 i) (aref (gethash "prompt_ids" expected) i))))
          (error "Installed processor/tokenizer differs from the reference.")))
      (multiple-value-bind (raw tokens reason) (docling:generate-image model page :max-new-tokens 256)
        (unless (and (equal raw (gethash "raw" expected)) (equalp tokens (gethash "tokens" expected))
                     (equal (string-downcase reason) (gethash "stop_reason" expected)))
          (error "Installed native generation differs from the reference."))
        (let ((markdown (docling:document-to-markdown
                          (docling:parse-doctags raw :token-ids tokens :stop-reason reason))))
          (unless (equal markdown (uiop:read-file-string
                                    (merge-pathnames "tests/fixtures/doctags/native-page.md" root)))
            (error "Installed Markdown output differs from the reference."))
          (setf result (experiment-object "tokens" tokens "raw" raw "stop_reason" (string-downcase reason)
                         "markdown" markdown "tiles" (gethash "tiles" expected)
                         "prompt_tokens" (length (gethash "prompt_ids" expected)))))))
    (setf remaining (getf (tb:backend-memory backend) :handles))
    (unless (zerop remaining) (error "Installation smoke check leaked native tensor handles.")))
  (setf libraries (sort (loop for library in (cffi:list-foreign-libraries)
                              for path = (cffi:foreign-library-pathname library)
                              when path collect (namestring path)) #'string<))
  (experiment-write output (experiment-object "device" (string-downcase device) "case" result
    "source_systems" systems "direct_foreign_libraries" (coerce libraries 'vector)
    "checkpoint" (namestring (truename checkpoint)) "remaining_handles" remaining
    "lisp_version" (lisp-implementation-version) "matches_reference" yason:true) t)
  (format t "~&PASS: installed ~A pipeline matches all ~D reference tokens and Markdown; zero tensor handles.~%"
          device (length (gethash "tokens" result))))
