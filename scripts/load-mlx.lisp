(require :asdf)
(let* ((root (uiop:pathname-parent-directory-pathname
              (uiop:pathname-directory-pathname *load-truename*)))
       (engine (uiop:ensure-directory-pathname
                (or (uiop:getenv "DOCLING_ENGINE")
                    (merge-pathnames "../cl-transformer-blocks/" root))))
       (deps (or (uiop:getenv "TB_DEPENDENCY_ROOT") (merge-pathnames ".build/deps/" engine))))
  (asdf:initialize-source-registry
   `(:source-registry (:directory ,root) (:directory ,engine) (:tree ,deps)
                     :ignore-inherited-configuration))
  (asdf:initialize-output-translations
   `(:output-translations (t (,(merge-pathnames ".build/fasl/" root) :implementation))
                         :ignore-inherited-configuration))
  (asdf:load-system "cl-docling/mlx"))
