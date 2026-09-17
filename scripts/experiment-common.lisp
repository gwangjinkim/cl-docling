(load (merge-pathnames "load-mlx.lisp" *load-truename*))
(asdf:load-system "cl-docling/pipeline")

(defun experiment-object (&rest pairs)
  (let ((object (make-hash-table :test #'equal)))
    (loop for (key value) on pairs by #'cddr do (setf (gethash key object) value))
    object))

(defun experiment-write (file value &optional json)
  (ensure-directories-exist file)
  (with-open-file (out file :direction :output :if-exists :error :external-format :utf-8)
    (if json (yason:encode value out) (write-string value out))))

(defun experiment-generate (model cases fixtures output phase device protocol)
  (dolist (case cases)
    (let* ((name (gethash "name" case))
           (prefix (merge-pathnames (format nil "~A/~A" phase name) output)))
      (format t "~&~A: ~A~%" phase name)
      (finish-output)
      (multiple-value-bind (raw ids reason)
          (docling:generate-image model (merge-pathnames (format nil "~A.png" name) fixtures)
                                  :task (gethash "task" protocol)
                                  :max-new-tokens (gethash "max_new_tokens" protocol))
        (let* ((document (docling:parse-doctags raw :token-ids ids :stop-reason reason))
               (elements (docling:parsed-document-elements document))
               (tables (remove-if-not (lambda (e) (eq :table (docling:document-element-kind e))) elements))
               (diagnostics (docling:parsed-document-diagnostics document))
               (markdown (handler-case (docling:document-to-markdown document) (docling:document-error () nil))))
          (experiment-write (make-pathname :type "doctags" :defaults prefix) raw)
          (when markdown (experiment-write (make-pathname :type "md" :defaults prefix) markdown))
          (experiment-write
           (make-pathname :type "json" :defaults prefix)
           (experiment-object
            "case" name "split" (gethash "split" case) "phase" phase
            "device" (string-downcase device) "dtype" "float32"
            "task" (gethash "task" protocol) "max_new_tokens" (gethash "max_new_tokens" protocol)
            "tokens" ids "raw" raw "stop_reason" (string-downcase reason)
            "non_table_elements" (- (length elements) (length tables))
            "markdown_written" (if markdown yason:true yason:false)
            "diagnostics" (coerce (mapcar (lambda (d) (experiment-object
                                                      "code" (string-downcase (docling:document-diagnostic-code d))
                                                      "message" (docling:document-diagnostic-message d))) diagnostics) 'vector)
            "tables" (coerce (mapcar
                               (lambda (e)
                                 (let ((table (docling:document-element-table e)))
                                   (experiment-object
                                    "rows" (length (docling:document-table-rows table))
                                    "columns" (docling:document-table-column-count table)
                                    "renderable" (if (docling:document-table-renderable-p table) yason:true yason:false)
                                    "cells" (coerce (mapcar (lambda (c) (vector
                                                     (docling:table-cell-row-index c) (docling:table-cell-column-index c)
                                                     (docling:table-cell-row-span c) (docling:table-cell-column-span c)
                                                     (docling:table-cell-text c))) (docling:document-table-cells table)) 'vector))))
                               tables) 'vector)) t)
          (format t "~&~D tokens; ~A; ~D diagnostics.~%" (length ids) reason (length diagnostics)))))))
