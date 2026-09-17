;;; Offline replay of the public M3 native output; no inference/download is performed.
(require :asdf)
(let* ((root (uiop:pathname-parent-directory-pathname
              (uiop:pathname-directory-pathname *load-truename*))))
  (asdf:initialize-source-registry
   `(:source-registry (:directory ,root) :ignore-inherited-configuration))
  (asdf:initialize-output-translations
   `(:output-translations (t ,(merge-pathnames ".build/" root)) :ignore-inherited-configuration))
  (asdf:load-system "cl-docling"))
(let* ((source (asdf:system-relative-pathname "cl-docling" "tests/fixtures/doctags/native-page.doctags"))
       (document (docling:parse-doctags (uiop:read-file-string source) :stop-reason :eos)))
  (write-string (docling:document-to-markdown document)))
