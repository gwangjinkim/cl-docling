;;; Model-free reference/output projection; supplied text is never Lisp code.
(require :asdf)
(let ((root (uiop:pathname-parent-directory-pathname (uiop:pathname-directory-pathname *load-truename*))))
  (asdf:initialize-source-registry `(:source-registry (:directory ,root) :ignore-inherited-configuration))
  (asdf:initialize-output-translations `(:output-translations (t ,(merge-pathnames ".build/" root)) :ignore-inherited-configuration))
  (asdf:load-system "cl-docling"))
(destructuring-bind (destination &rest arguments) (uiop:command-line-arguments)
  (unless (evenp (length arguments)) (error "Expected stop/file pairs."))
  (loop for (stop file) on arguments by #'cddr for index from 0 do
    (let* ((reason (cond ((equal stop "eos") :eos) ((equal stop "length") :length)
                         ((equal stop "unknown") :unknown) (t (error "Invalid stop reason."))))
           (document (docling:parse-doctags (uiop:read-file-string file :external-format :utf-8) :stop-reason reason))
           (elements (docling:parsed-document-elements document))
           (codes (mapcar #'docling:document-diagnostic-code (docling:parsed-document-diagnostics document)))
           (markdown (handler-case (docling:document-to-markdown document) (docling:document-error () nil))))
      ;; References use EOS for syntax checking; outputs retain their actual stop reason.
      (when markdown
        (with-open-file (out (merge-pathnames (format nil "~D.md" index) (uiop:ensure-directory-pathname destination))
                             :direction :output :if-exists :error :external-format :utf-8)
          (write-string markdown out)))
      (format t "~&BOOK~C~D~C~D~C~{~A~^,~}~C~{~(~A~)~^,~}~%" #\Tab index #\Tab (if markdown 1 0) #\Tab
              (mapcar (lambda (e) (format nil "~(~A~):~D" (docling:document-element-kind e)
                                         (or (docling:document-element-level e) 0))) elements) #\Tab codes))))
