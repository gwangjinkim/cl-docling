(require :asdf)
(let* ((root (uiop:pathname-parent-directory-pathname
              (uiop:pathname-directory-pathname *load-truename*))))
  (asdf:initialize-source-registry `(:source-registry (:directory ,root) :ignore-inherited-configuration))
  (asdf:initialize-output-translations `(:output-translations (t ,(merge-pathnames ".build/" root)) :ignore-inherited-configuration))
  (handler-case
      (progn (asdf:test-system "cl-docling/bundles") (uiop:quit 0))
    (error (condition) (format *error-output* "~&FAIL: ~A~%" condition) (uiop:quit 1))))
