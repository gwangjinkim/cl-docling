;;; One explicit page image + DocTags to source-backed picture Markdown; no model.
(load (merge-pathnames "load-images.lisp" *load-truename*))
(require :sb-posix)

(destructuring-bind (raw-path image-path output-path) (uiop:command-line-arguments)
  (let* ((output (uiop:ensure-directory-pathname output-path))
         (assets (merge-pathnames "assets/" output)))
    (when (probe-file output) (error "Output must be new."))
    (sb-posix:mkdir (uiop:native-namestring output) #o700)
    (sb-posix:mkdir (uiop:native-namestring assets) #o700)
    (let* ((raw (uiop:read-file-string raw-path :external-format :utf-8))
           (document (docling:parse-doctags raw :stop-reason :eos))
           (elements (docling:parsed-document-elements document))
           (pictures (remove-if-not (lambda (element) (eq :picture (docling:document-element-kind element))) elements))
           (codes (mapcar #'docling:document-diagnostic-code (docling:parsed-document-diagnostics document)))
           (relative "assets/picture-1.png")
           (asset (merge-pathnames relative output)))
      (unless (and (null codes) (= 1 (length pictures)))
        (error "Expected one clean picture document, got ~D picture(s), codes ~S."
               (length pictures) codes))
      (let ((picture (first pictures)))
        (multiple-value-bind (width height)
            (docling:write-picture-asset image-path asset (docling:document-element-location picture))
          (let ((markdown (docling:document-to-markdown document :picture-paths (list relative))))
            (with-open-file (out (merge-pathnames "document.md" output) :direction :output
                                 :if-exists :error :external-format :utf-8)
              (write-string markdown out))
            (format t "~&PICTURE~C~{~D~^,~}~C~(~A~)~C~D~C~D~C~{~A~^,~}~%"
                    #\Tab (docling:document-element-location picture) #\Tab
                    (or (docling:document-element-classification picture) :none)
                    #\Tab width #\Tab height #\Tab
                    (mapcar (lambda (element)
                              (format nil "~(~A~):~D"
                                      (docling:document-element-kind element)
                                      (or (docling:document-element-level element) 0)))
                            elements))))))))
