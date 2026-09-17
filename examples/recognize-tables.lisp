;;; Two frozen synthetic PNGs -> native generation -> retained artifacts.
;;; Expected labels are never read by the model. Python is not used in this path.
(load (merge-pathnames "../scripts/load-mlx.lisp" *load-truename*))
(asdf:load-system "cl-docling/pipeline")

(defun table-report-object (&rest pairs)
  (let ((object (make-hash-table :test #'equal)))
    (loop for (key value) on pairs by #'cddr do (setf (gethash key object) value))
    object))

(defun table-report-write (file value &optional json)
  (with-open-file (out file :direction :output :if-exists :error :external-format :utf-8)
    (if json (yason:encode value out) (write-string value out))))

(defun table-report-grid (table)
  (table-report-object
   "rows" (length (docling:document-table-rows table)) "columns" (docling:document-table-column-count table)
   "renderable" (if (docling:document-table-renderable-p table) yason:true yason:false)
   "cells" (coerce (mapcar (lambda (cell)
                            (vector (docling:table-cell-row-index cell) (docling:table-cell-column-index cell)
                                    (docling:table-cell-row-span cell) (docling:table-cell-column-span cell)
                                    (docling:table-cell-text cell))) (docling:document-table-cells table)) 'vector)))

(let* ((arguments (uiop:command-line-arguments))
       (output (uiop:ensure-directory-pathname
                (or (first arguments) (error "Supply a fresh output directory."))))
       (checkpoint (or (uiop:getenv "DOCLING_MODEL") (asdf:system-relative-pathname "cl-docling" ".build/smoldocling/")))
       (device (ecase (intern (string-upcase (or (uiop:getenv "TB_DEVICE") "cpu")) :keyword) (:cpu :cpu) (:gpu :gpu))))
  (when (probe-file output) (error "Output already exists: ~A" output))
  (ensure-directories-exist output)
  (tb:with-resource (model (docling:load-document-model checkpoint :device device))
    (dolist (name '("grid" "merged"))
      (let ((png (asdf:system-relative-pathname "cl-docling" (format nil "tests/fixtures/table-pages/~A.png" name))))
        (format t "~&Generating ~A on ~A (512-token maximum).~%" name device)
        (finish-output)
        (multiple-value-bind (raw ids reason) (docling:generate-image model png :max-new-tokens 512)
          (table-report-write (merge-pathnames (format nil "~A.doctags" name) output) raw)
          (let* ((document (docling:parse-doctags raw :token-ids ids :stop-reason reason))
                 (elements (docling:parsed-document-elements document))
                 (tables (remove-if-not (lambda (element) (eq :table (docling:document-element-kind element))) elements))
                 (diagnostics (docling:parsed-document-diagnostics document))
                 (markdown (handler-case (docling:document-to-markdown document) (docling:document-error () nil))))
            (when markdown (table-report-write (merge-pathnames (format nil "~A.md" name) output) markdown))
            (table-report-write
             (merge-pathnames (format nil "~A.json" name) output)
             (table-report-object
              "case" name "device" (string-downcase device) "dtype" "float32"
              "task" "Convert this page to docling." "max_new_tokens" 512
              "tokens" ids "stop_reason" (string-downcase reason) "raw" raw
              "non_table_elements" (- (length elements) (length tables))
              "markdown_written" (if markdown yason:true yason:false)
              "diagnostics" (coerce (mapcar (lambda (d) (table-report-object
                                                        "code" (string-downcase (docling:document-diagnostic-code d))
                                                        "start" (docling:document-diagnostic-start d)
                                                        "message" (docling:document-diagnostic-message d))) diagnostics) 'vector)
              "tables" (coerce (mapcar (lambda (e) (table-report-grid (docling:document-element-table e))) tables) 'vector)) t)
            (format t "~&~A: ~D tokens, ~A; ~D table(s), ~D diagnostic(s); Markdown ~:[blocked~;written~].~%"
                    name (length ids) reason (length tables) (length diagnostics) markdown)))))))
