;; Offline parser replay, never a new inference run. No Python or native libraries.
(require :asdf)
(let* ((root (uiop:pathname-parent-directory-pathname
              (uiop:pathname-directory-pathname *load-truename*)))
       (output (uiop:ensure-directory-pathname
                (or (first (uiop:command-line-arguments))
                    (merge-pathnames ".build/public-page-markdown/" root)))))
  (asdf:initialize-source-registry
   `(:source-registry (:directory ,root) :ignore-inherited-configuration))
  (asdf:initialize-output-translations
   `(:output-translations (t ,(merge-pathnames ".build/" root)) :ignore-inherited-configuration))
  (asdf:load-system "cl-docling")
  (when (probe-file output) (error "Output must be a new directory: ~A" output))
  (let* ((documents
           (loop for page in '(27 28) collect
             (uiop:symbol-call :docling :parse-doctags
               (uiop:read-file-string
                (merge-pathnames (format nil "tests/fixtures/public-pdf/results/native/page-~D.doctags" page) root)
                :external-format :utf-8)
               :page-number page :stop-reason :eos)))
         ;; Validate/render the entire batch before creating output files.
         (combined (uiop:symbol-call :docling :documents-to-markdown documents))
         (outputs (cons (cons "document.md" combined)
                        (loop for document in documents for page in '(27 28) collect
                          (cons (format nil "page-~D.md" page)
                                (uiop:symbol-call :docling :document-to-markdown document))))))
    (dolist (entry outputs)
      (let ((file (merge-pathnames (car entry) output)))
        (ensure-directories-exist file)
        (with-open-file (out file :direction :output :if-exists :error :external-format :utf-8)
          (write-string (cdr entry) out))
        (format t "~&Saved ~A (parser replay only; recorded table errors remain).~%" file)))))
