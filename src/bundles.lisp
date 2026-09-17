(in-package #:cl-docling)

(export '(write-document-bundle))

(defun bundle-write-text (file text)
  (with-open-file (out file :direction :output :if-exists :error :external-format :utf-8)
    (write-string text out)))

(defun bundle-write-data (file data)
  (with-open-file (out file :direction :output :if-exists :error :external-format :utf-8)
    (with-standard-io-syntax
      ;; Logical schema values only: avoid SBCL's #A syntax for specialized strings.
      (let ((*print-pretty* nil) (*print-readably* nil))
        (write data :stream out) (terpri out)))))

(defun bundle-publish-manifest (directory manifest)
  ;; A closed staging file prevents a partially written final marker on Lisp/I/O errors.
  ;; The newly created private directory must not be concurrently modified.
  (let ((pending (merge-pathnames "manifest.pending.sexp" directory))
        (final (merge-pathnames "manifest.sexp" directory)))
    (bundle-write-data pending manifest)
    (when (probe-file final) (invalid-document "Bundle manifest already exists."))
    (rename-file pending final)))

(defun bundle-diagnostic-data (diagnostic)
  (list :code (document-diagnostic-code diagnostic)
        :start (document-diagnostic-start diagnostic)
        :message (document-diagnostic-message diagnostic)))

(defun write-document-bundle (pages output &key allow-partial (max-pages 256)
                                               (max-output-characters 8000000)
                                               (max-raw-characters 8000000)
                                               (max-token-ids 1000000))
  "Write explicit parsed pages to a new local directory; return pathname and status.
Status is :COMPLETE, :PARTIAL or :BLOCKED (diagnostics, no Markdown). All statuses
retain raw text, supplied IDs, stop reasons and diagnostics. MANIFEST.SEXP is last.
Argument/render failures precede directory creation; I/O failures may leave artifacts.
SBCL/POSIX only. No downloads, subprocesses, inference or automatic cleanup."
  (dolist (bound (list max-pages max-output-characters max-raw-characters max-token-ids))
    (unless (and (integerp bound) (plusp bound))
      (invalid-document "Expected positive integer bundle bounds.")))
  (validate-document-pages pages max-pages)
  (let ((raw-count 0) (id-count 0))
    (dolist (page pages)
      (when (> (incf raw-count (length (parsed-document-raw page))) max-raw-characters)
        (invalid-document "Bundle raw-character budget exceeded."))
      (when (> (incf id-count (length (parsed-document-token-ids page))) max-token-ids)
        (invalid-document "Bundle token-ID budget exceeded."))))
  (let* ((diagnostics-p (some #'parsed-document-diagnostics pages))
         (status (cond ((not diagnostics-p) :complete) (allow-partial :partial) (t :blocked)))
         (markdown (unless (eq status :blocked)
                     (documents-to-markdown pages :allow-partial allow-partial :max-pages max-pages
                                                 :max-output-characters max-output-characters)))
         (page-markdown (when markdown
                          (mapcar (lambda (page) (document-to-markdown page :allow-partial allow-partial)) pages)))
         (directory (merge-pathnames (uiop:ensure-directory-pathname output) (uiop:getcwd)))
         (entries nil))
    (when (wild-pathname-p directory) (invalid-document "Wildcard output paths are unsupported."))
    (unless (uiop:directory-exists-p (uiop:pathname-parent-directory-pathname directory))
      (invalid-document "Bundle parent directory must already exist."))
    ;; mkdir atomically refuses existing files/directories/symlinks, including empty ones.
    (handler-case (sb-posix:mkdir (namestring directory) #o700)
      (sb-posix:syscall-error (condition) (invalid-document "Cannot create new bundle directory: ~A" condition)))
    (setf directory (truename directory))
    (loop for page in pages for index from 0 do
      (let* ((number (parsed-document-page-number page))
             (raw-name (format nil "page-~D.doctags" number))
             (data-name (format nil "page-~D.sexp" number))
             (md-name (when markdown (format nil "page-~D.md" number))))
        (bundle-write-text (merge-pathnames raw-name directory) (parsed-document-raw page))
        (bundle-write-data (merge-pathnames data-name directory)
                           (list :page-number number :token-ids (parsed-document-token-ids page)
                                 :stop-reason (parsed-document-stop-reason page)
                                 :diagnostics (mapcar #'bundle-diagnostic-data (parsed-document-diagnostics page))))
        (when md-name (bundle-write-text (merge-pathnames md-name directory) (nth index page-markdown)))
        (push (list :page-number number :raw raw-name :metadata data-name :markdown md-name) entries)))
    (when markdown (bundle-write-text (merge-pathnames "document.md" directory) markdown))
    (bundle-publish-manifest directory
      (list :schema-version 1 :status status :markdown (when markdown "document.md")
            :pages (nreverse entries)))
    (values directory status)))
