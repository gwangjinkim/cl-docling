(require :asdf)
(let ((root (uiop:pathname-parent-directory-pathname
             (uiop:pathname-directory-pathname *load-truename*))))
  (asdf:initialize-source-registry
   `(:source-registry (:directory ,root) :ignore-inherited-configuration))
  (asdf:initialize-output-translations
   `(:output-translations (t ,(merge-pathnames ".build/" root)) :ignore-inherited-configuration))
  (asdf:load-system "cl-docling"))

(let ((layout (docling:make-vision-layout)))
  (format t "512px tile: ~D patches -> ~D visual tokens.~%"
          (docling:patch-count layout) (docling:image-token-count layout))
  (format t "Connector width: ~D -> ~D (geometry only).~%"
          (docling:connector-input-width layout) (docling:vision-layout-text-hidden-size layout)))
