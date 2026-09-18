(in-package #:cl-docling)
(export '(run-pdf-job))

(defun application-path-exists-p (path)
  ;; PROBE-FILE follows symlinks and misses dangling ones; never replace those.
  (handler-case (progn (sb-posix:lstat (string-right-trim "/" (namestring path))) t)
    (sb-posix:syscall-error (condition)
      (if (= (sb-posix:syscall-errno condition) sb-posix:enoent) nil (error condition)))))

(defun application-list-p (value maximum predicate)
  (loop repeat maximum
        do (when (null value) (return-from application-list-p t))
           (unless (and (consp value) (funcall predicate (car value)))
             (return-from application-list-p nil))
           (setf value (cdr value))
        finally (return (null value))))

(defun application-regular-file (file maximum)
  (unless (sb-posix:s-isreg (sb-posix:stat-mode (sb-posix:lstat (namestring file))))
    (invalid-document "Expected an ordinary job file, not a symlink: ~A" file))
  (pdf-file-size file maximum))

(defun application-read-data (file)
  "Read our bounded local records, never evaluate reader syntax or arbitrary dispatch."
  (application-regular-file file 2097152)
  (let ((text (uiop:read-file-string file)) (depth 0) (quoted nil) (escaped nil))
    ;; Bound nesting before invoking READ, including malformed metadata.
    (loop for c across text do
      (cond (escaped (setf escaped nil))
            ((and quoted (char= c #\\)) (setf escaped t))
            ((char= c #\") (setf quoted (not quoted)))
            (quoted)
            ((char= c #\() (when (> (incf depth) 8) (invalid-document "Job metadata nesting exceeded.")))
            ((char= c #\)) (when (minusp (decf depth)) (invalid-document "Unbalanced job metadata.")))))
    (unless (and (zerop depth) (not quoted)) (invalid-document "Incomplete job metadata."))
    (with-standard-io-syntax
      (let ((*read-eval* nil) (*readtable* (copy-readtable nil)))
        (dolist (c '(#\# #\' #\` #\, #\;))
          (set-macro-character c (lambda (stream character)
                                   (declare (ignore stream character))
                                   (invalid-document "Unsupported job reader syntax."))))
        (with-input-from-string (in text)
          (let ((value (read in)))
            (unless (eq :eof (read in nil :eof)) (invalid-document "Trailing job data."))
            value))))))

(defun application-files-equal-p (left right)
  (application-regular-file right 67108864)
  (pdf-file-size left 67108864)
  (with-open-file (a left :element-type '(unsigned-byte 8))
    (with-open-file (b right :element-type '(unsigned-byte 8))
      (unless (= (file-length a) (file-length b)) (return-from application-files-equal-p nil))
      (let ((x (make-array 65536 :element-type '(unsigned-byte 8)))
            (y (make-array 65536 :element-type '(unsigned-byte 8))))
        (loop for n = (read-sequence x a) for m = (read-sequence y b)
              always (and (= n m) (loop for i below n always (= (aref x i) (aref y i))))
              while (plusp n))))))

(defun application-attempt (root)
  (loop for n from 1 to 10000
        for directory = (merge-pathnames (format nil "attempt-~4,'0D/" n) root)
        unless (application-path-exists-p directory) do
          (sb-posix:mkdir (namestring directory) #o700)
          (return-from application-attempt directory))
  (invalid-document "Job attempt limit exceeded."))

(defun application-commit-data (data temporary final)
  (bundle-write-data temporary data)
  (when (application-path-exists-p final) (invalid-document "Refusing to replace job record: ~A" final))
  (rename-file temporary final))

(defun application-page-data (raw ids reason number maximum)
  (unless (and (stringp raw) (<= (length raw) 262144)
               (vectorp ids) (<= (length ids) maximum)
               (every (lambda (n) (typep n '(unsigned-byte 32))) ids)
               (member reason '(:eos :length)))
    (invalid-document "Invalid or oversized generated page record."))
  (list :schema 1 :number number :raw raw :ids (coerce ids 'list) :reason reason))

(defun application-parse-record (data number maximum)
  (unless (and (application-list-p data 10 (constantly t)) (= (length data) 10)
               (equal (loop for tail on data by #'cddr collect (car tail))
                      '(:schema :number :raw :ids :reason))
               (eql (getf data :schema) 1) (eql (getf data :number) number)
               (application-list-p (getf data :ids) maximum
                                   (lambda (n) (typep n '(unsigned-byte 32)))))
    (invalid-document "Invalid cached page schema or token IDs."))
  (let ((raw (getf data :raw)) (ids (coerce (getf data :ids) 'vector)) (reason (getf data :reason)))
    (application-page-data raw ids reason number maximum)
    (parse-doctags raw :token-ids ids :stop-reason reason :page-number number)))

(defun run-pdf-job (input output generator &key identity (pages '(1)) (dpi 144)
                                               (max-new-tokens 512)
                                               (task "Convert this page to docling.") resume)
  "Run or explicitly resume a bounded local job; return bundle, status, attempt report.
GENERATOR receives a PNG pathname and physical page number, returning raw/IDs/stop.
IDENTITY must describe every generation-affecting dependency; the native CLI hashes
its checkpoint/runtime. Callers must not mutate source/model/job files during a run.
Failed pages are retried only on explicit resume; committed pages are never replaced.
Statuses: :COMPLETE, :BLOCKED (parser diagnostics), :FAILED (page execution failure)."
  (unless (and (functionp generator) identity
               (application-list-p identity 512 (lambda (x) (and (stringp x) (<= 1 (length x) 4096))))
               (pdf-page-selection-p pages 16)
               (loop for (a b) on pages while b always (< a b))
               (typep dpi '(integer 1 300)) (typep max-new-tokens '(integer 1 4096))
               (stringp task) (<= 1 (length task) 4096))
    (invalid-document "Invalid job identity, generator, page order, DPI, token budget or task."))
  (let* ((source (truename input))
         (root (merge-pathnames (uiop:ensure-directory-pathname output) (uiop:getcwd)))
         (lock (merge-pathnames ".lock/" root))
         (request (list :schema 1 :identity identity :pages pages :dpi dpi
                        :max-new-tokens max-new-tokens :task task))
         (locked nil))
    (when (wild-pathname-p root) (invalid-document "Wildcard job paths are unsupported."))
    (pdf-file-size source 67108864)
    (if resume
        (unless (and (uiop:directory-exists-p root)
                     (sb-posix:s-isdir (sb-posix:stat-mode
                                        (sb-posix:lstat (string-right-trim "/" (namestring root))))))
          (invalid-document "Resume requires an existing ordinary job directory."))
        (sb-posix:mkdir (namestring root) #o700))
    (unwind-protect
         (progn
           ;; A stale lock after abrupt process death must be inspected by the operator.
           ;; Never guess ownership, remove another lock or restart another process.
           (sb-posix:mkdir (namestring lock) #o700)
           (setf locked t)
           (let ((snapshot (merge-pathnames "input.pdf" root)) (job (merge-pathnames "job.sexp" root)))
             (if resume
                 (unless (and (equal request (application-read-data job))
                              (application-files-equal-p source snapshot))
                   (invalid-document "Cannot resume: source, model/runtime identity or options changed."))
                 (progn (snapshot-pdf source snapshot 67108864)
                        (application-commit-data request (merge-pathnames "job.pending.sexp" root) job)))
             (let ((documents (make-hash-table)) (missing nil) (failures nil))
               ;; Validate all cached records before rasterizing or invoking a model.
               (dolist (number pages)
                 (let ((record (merge-pathnames (format nil "page-~D.sexp" number) root)))
                   (if (application-path-exists-p record)
                       (setf (gethash number documents)
                             (application-parse-record (application-read-data record) number max-new-tokens))
                       (push number missing))))
               (let* ((attempt (application-attempt root))
                      (report (merge-pathnames "report.sexp" attempt))
                      (rasters (when missing
                                 (rasterize-pdf snapshot (merge-pathnames "rasters/" attempt)
                                                :pages (nreverse missing) :dpi dpi))))
                 (dolist (page rasters)
                   (let ((number (pdf-page-number page)))
                     (handler-case
                         (multiple-value-bind (raw ids reason) (funcall generator (pdf-page-pathname page) number)
                           (let* ((data (application-page-data raw ids reason number max-new-tokens))
                                  (record (format nil "page-~D.sexp" number)))
                             ;; Commit raw generation first. A later parse/export failure must not erase it.
                             (application-commit-data data (merge-pathnames record attempt) (merge-pathnames record root))
                             (setf (gethash number documents) (application-parse-record data number max-new-tokens))))
                       (error (condition)
                         (push (list :page number :message (princ-to-string condition)) failures)))))
                 (multiple-value-bind (bundle status)
                     (if failures (values nil :failed)
                         (write-document-bundle (mapcar (lambda (n) (gethash n documents)) pages)
                                                (merge-pathnames "export/" attempt) :max-pages 16))
                   (application-commit-data
                    (list :schema 1 :status status :pages pages :failures (nreverse failures)
                          :bundle (when bundle "export/")
                          :completed-pages (remove-if-not (lambda (n) (gethash n documents)) pages))
                    (merge-pathnames "report.pending.sexp" attempt) report)
                   (values bundle status report))))))
      (when locked (sb-posix:rmdir (namestring lock))))))
