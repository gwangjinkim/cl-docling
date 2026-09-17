;; Offline replay with all original token IDs; no Python or model runtime.
(require :asdf)
(let* ((root (uiop:pathname-parent-directory-pathname
              (uiop:pathname-directory-pathname *load-truename*)))
       (output (or (first (uiop:command-line-arguments)) (merge-pathnames ".build/public-document-bundle/" root))))
  (asdf:initialize-source-registry `(:source-registry (:directory ,root) :ignore-inherited-configuration))
  (asdf:initialize-output-translations `(:output-translations (t ,(merge-pathnames ".build/" root)) :ignore-inherited-configuration))
  (asdf:load-system "cl-docling/bundles")
  (let* ((records (with-open-file (in (merge-pathnames "tests/fixtures/bundles/generation.sexp" root))
                    (let ((*read-eval* nil)) (read in))))
         (pages (loop for record in records for page = (getf record :page-number) collect
                  (uiop:symbol-call :docling :parse-doctags
                    (uiop:read-file-string
                     (merge-pathnames (format nil "tests/fixtures/public-pdf/results/native/page-~D.doctags" page) root)
                     :external-format :utf-8)
                    :page-number page :stop-reason (getf record :stop-reason)
                    :token-ids (getf record :token-ids)))))
    (multiple-value-bind (directory status) (uiop:symbol-call :docling :write-document-bundle pages output)
      (format t "~&~A: ~A (recorded-output replay; known recognition errors remain).~%" status directory))))
