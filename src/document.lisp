(in-package #:cl-docling)

(define-condition document-error (error)
  ((message :initarg :message :reader document-error-message))
  (:report (lambda (condition stream)
             (write-string (document-error-message condition) stream))))

(defun invalid-document (control &rest arguments)
  (error 'document-error :message (apply #'format nil control arguments)))

(defstruct (document-diagnostic (:constructor %make-document-diagnostic))
  (code nil :read-only t) (message "" :read-only t) (start 0 :read-only t))

(defstruct (document-element (:constructor %make-document-element))
  (kind :unknown :read-only t) (tag "" :read-only t)
  (text "" :read-only t) (level nil :read-only t) (location nil :read-only t)
  (classification nil :read-only t)
  (children nil :read-only t) (start 0 :read-only t) (end 0 :read-only t)
  (table nil :read-only t))

(defstruct (table-cell (:constructor %make-table-cell))
  (tag "" :read-only t)
  (row-index 0 :read-only t) (column-index 0 :read-only t)
  (row-span 1 :read-only t) (column-span 1 :read-only t)
  (kind :data :read-only t) (text "" :read-only t)
  (start 0 :read-only t) (end 0 :read-only t))

(defstruct (document-table (:constructor %make-document-table))
  (rows nil :read-only t) (column-count 0 :read-only t) (renderable-p nil :read-only t))

(defstruct (parsed-document (:constructor %make-parsed-document))
  (raw "" :read-only t) (elements nil :read-only t) (diagnostics nil :read-only t)
  (page-number 1 :read-only t) (stop-reason :unknown :read-only t)
  (token-ids nil :read-only t))

(defun escape-document-text (text)
  "Render literal model text, not executable HTML or model-supplied Markdown."
  (with-output-to-string (out)
    (loop for character across text for index from 0 do
      (case character
        (#\& (write-string "&amp;" out))
        (#\< (write-string "&lt;" out))
        (#\> (write-string "&gt;" out))
        (otherwise
         (when (or (find character "\\`*_{}[]()#+-!|~=")
                   (and (char= character #\.) (plusp index)
                        (digit-char-p (char text (1- index)))))
           (write-char #\\ out))
         (write-char character out))))))

(defun picture-path-safe-p (path)
  (and (stringp path) (< 0 (length path) 256)
       (<= (length "assets/") (length path))
       (string= "assets/" path :end2 (length "assets/"))
       (> (length path) (length "assets/"))
       (not (search ".." path))
       (every (lambda (character)
                (or (alphanumericp character) (find character "-_.")))
              (subseq path (length "assets/")))))

(defun markdown-element (element &optional picture-path)
  (let ((text (escape-document-text
               (if (member (document-element-kind element) '(:heading :title))
                   (substitute #\Space #\Newline (substitute #\Newline #\Return (document-element-text element)))
                   (document-element-text element)))))
    (case (document-element-kind element)
      (:table (markdown-table (document-element-table element)))
      (:heading (format nil "~A ~A" (make-string (1+ (document-element-level element)) :initial-element #\#) text))
      (:title (format nil "# ~A" text))
      (:picture
       (unless (picture-path-safe-p picture-path)
         (invalid-document "Picture export requires one safe assets/ filename per picture."))
       (format nil "![Picture](~A)" picture-path))
      ((:text :page-footer :page-header :footnote) text)
      (:list-item (format nil "- ~A" text))
      ((:ordered-list :unordered-list)
       (with-output-to-string (out)
         (loop for child in (document-element-children element) for index from 1 do
           (when (> index 1) (terpri out))
           (let ((prefix (if (eq :ordered-list (document-element-kind element))
                             (format nil "~D. " index) "- ")))
             (if (eq :list-item (document-element-kind child))
                 (let ((lines (uiop:split-string (escape-document-text (document-element-text child))
                                                :separator '(#\Newline))))
                   (write-string prefix out)
                   (write-string (first lines) out)
                   (dolist (line (rest lines))
                     (format out "~%~A~A" (make-string (length prefix) :initial-element #\Space) line)))
                 (write-string "[Unsupported list content; see raw DocTags.]" out))))))
      (otherwise "[Unsupported content; see raw DocTags.]"))))

(defun document-to-markdown (document &key allow-partial picture-paths)
  "Return Markdown and diagnostics. Any diagnostic blocks export unless ALLOW-PARTIAL.
Partial export starts with a visible warning; PARSED-DOCUMENT-RAW remains canonical."
  (unless (parsed-document-p document) (invalid-document "Expected a parsed document."))
  (let* ((diagnostics (parsed-document-diagnostics document))
         (pictures (count :picture (parsed-document-elements document)
                          :key #'document-element-kind)))
    (unless (and (listp picture-paths) (= pictures (length picture-paths))
                 (every #'picture-path-safe-p picture-paths))
      (invalid-document "Expected exactly one safe assets/ path for every picture."))
    (when (and diagnostics (not allow-partial))
      (invalid-document "DocTags has ~D diagnostic(s); inspect them or explicitly allow partial export."
                        (length diagnostics)))
    (values
     (with-output-to-string (out)
       (when diagnostics
         (format out "> WARNING: Partial/unverified conversion (~D diagnostics). Retain and inspect raw DocTags.~%~%"
                 (length diagnostics)))
       (loop with paths = picture-paths
             for element in (parsed-document-elements document) for first = t then nil do
         (unless first (terpri out))
         (write-string (markdown-element element
                        (when (eq :picture (document-element-kind element)) (pop paths))) out)
         (terpri out)))
     diagnostics)))

(defun validate-document-pages (pages max-pages)
  "Validate a bounded proper list without traversing a cycle indefinitely."
  (unless pages (invalid-document "Expected at least one parsed page."))
  (let ((seen (make-hash-table :test #'eq)) (previous 0) (count 0))
    (loop for tail = pages then (cdr tail) while tail do
      (unless (consp tail) (invalid-document "Pages must be a proper list."))
      (when (gethash tail seen) (invalid-document "Cyclic page list."))
      (setf (gethash tail seen) t)
      (when (> (incf count) max-pages) (invalid-document "Page count exceeds ~D." max-pages))
      (let ((document (car tail)))
        (unless (parsed-document-p document) (invalid-document "Expected parsed pages only."))
        (let ((page (parsed-document-page-number document)))
          (unless (and (integerp page) (> page previous))
            (invalid-document "Page numbers must be positive and strictly increasing."))
          (setf previous page))))))

(defun documents-to-markdown (pages &key allow-partial (max-pages 256)
                                       (max-output-characters 8000000))
  "Compose ordered parsed pages with explicit page boundaries, without file I/O.
Return Markdown and an alist of (page-number . diagnostics), including clean pages.
Any diagnostic blocks strict export; partial output carries global/local warnings.
Budgets count pages and Lisp output characters, not bytes or total working memory."
  (dolist (bound (list max-pages max-output-characters))
    (unless (and (integerp bound) (plusp bound))
      (invalid-document "Expected positive integer composition bounds.")))
  (validate-document-pages pages max-pages)
  (let* ((reports (mapcar (lambda (document)
                            (cons (parsed-document-page-number document)
                                  (copy-list (parsed-document-diagnostics document)))) pages))
         (failed (remove-if-not #'cdr reports)))
    (when (and failed (not allow-partial))
      (invalid-document "Diagnostics on page(s) ~{~D~^, ~}; inspect them before export."
                        (mapcar #'car failed)))
    (values
     (with-output-to-string (out)
       (let ((used 0))
         (flet ((emit (text)
                  (when (> (incf used (length text)) max-output-characters)
                    (invalid-document "Combined Markdown exceeds ~D characters." max-output-characters))
                  (write-string text out)))
           (when failed
             (emit (format nil "> WARNING: Partial/unverified multi-page conversion (~D affected pages). Retain raw DocTags and page diagnostics.~%~%"
                           (length failed))))
           (loop for document in pages for first = t then nil do
             (unless first (emit (format nil "~%---~%~%")))
             (emit (format nil "# Page ~D~%~%" (parsed-document-page-number document)))
             (emit (document-to-markdown document :allow-partial allow-partial))))))
     reports)))
