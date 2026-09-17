(require :asdf)
(let* ((root (uiop:pathname-parent-directory-pathname
              (uiop:pathname-directory-pathname *load-truename*)))
       (build (merge-pathnames ".build/" root)))
  (asdf:initialize-source-registry
   `(:source-registry (:directory ,root) :ignore-inherited-configuration))
  (asdf:initialize-output-translations
   `(:output-translations (t ,build) :ignore-inherited-configuration))
  (handler-case
      (progn (asdf:test-system "cl-docling") (uiop:quit 0))
    (error (condition)
      (format *error-output* "~&FAIL: ~A~%" condition)
      (uiop:quit 1))))
