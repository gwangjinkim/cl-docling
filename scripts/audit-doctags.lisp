;;; Plain target files only: never read dataset strings as Lisp forms.
(require :asdf)
(let ((root (uiop:pathname-parent-directory-pathname
             (uiop:pathname-directory-pathname *load-truename*))))
  (asdf:initialize-source-registry
   `(:source-registry (:directory ,root) :ignore-inherited-configuration))
  (asdf:initialize-output-translations
   `(:output-translations (t ,(merge-pathnames ".build/" root))
                         :ignore-inherited-configuration))
  (asdf:load-system "cl-docling"))

(loop for file in (uiop:command-line-arguments) for index from 0 do
  (handler-case
      (let* ((raw (uiop:read-file-string file :external-format :utf-8))
             ;; EOS here describes a complete supplied annotation, not model output.
             (document (docling:parse-doctags raw :stop-reason :eos))
             (codes (remove-duplicates
                     (mapcar #'docling:document-diagnostic-code
                             (docling:parsed-document-diagnostics document))))
             (renderable (handler-case
                             (progn (docling:document-to-markdown document) t)
                           (docling:document-error () nil))))
        (format t "~&AUDIT~C~D~C~D~C~{~(~A~)~^,~}~%"
                #\Tab index #\Tab (if renderable 1 0) #\Tab codes))
    (docling:document-error ()
      (format t "~&AUDIT~C~D~C0~Cdocument-error~%" #\Tab index #\Tab #\Tab))))
