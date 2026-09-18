;;; Model-free parser replay; the Python caller verifies frozen EOS and IDs.
(require :asdf)
(let ((root (uiop:pathname-parent-directory-pathname
             (uiop:pathname-directory-pathname *load-truename*))))
  (asdf:initialize-source-registry
   `(:source-registry (:directory ,root) :ignore-inherited-configuration))
  (asdf:initialize-output-translations
   `(:output-translations (t ,(merge-pathnames ".build/" root)) :ignore-inherited-configuration))
  (asdf:load-system "cl-docling"))

(destructuring-bind (destination &rest files) (uiop:command-line-arguments)
  (loop for file in files for index from 0 do
    (let* ((raw (uiop:read-file-string file :external-format :utf-8))
           (document (docling:parse-doctags raw :stop-reason :eos))
           (elements (docling:parsed-document-elements document))
           (diagnostics (docling:parsed-document-diagnostics document))
           (markdown (handler-case (docling:document-to-markdown document)
                       (docling:document-error () nil))))
      ;; This replay is deliberately limited to the four table-free baseline pages.
      (assert (notany (lambda (e) (eq :table (docling:document-element-kind e))) elements))
      (assert (string= raw (docling:parsed-document-raw document)))
      (when markdown
        (with-open-file (out (merge-pathnames (format nil "page-~3,'0D.md" index)
                                             (uiop:ensure-directory-pathname destination))
                             :direction :output :if-exists :error :external-format :utf-8)
          (write-string markdown out)))
      (format t "~&REPLAY~C~D~C~D~C~D~C~{~(~A~)~^,~}~%"
              #\Tab index #\Tab (if markdown 1 0) #\Tab (length elements) #\Tab
              (mapcar #'docling:document-diagnostic-code diagnostics)))))
