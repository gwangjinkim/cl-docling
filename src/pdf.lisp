(in-package #:cl-docling)

(define-condition pdf-error (error)
  ((message :initarg :message :reader pdf-error-message))
  (:report (lambda (condition stream) (write-string (pdf-error-message condition) stream))))

(defun invalid-pdf (control &rest arguments)
  (error 'pdf-error :message (apply #'format nil control arguments)))

(defstruct (pdf-page (:constructor %make-pdf-page))
  (number 1 :read-only t) (pathname nil :read-only t) (dpi 144 :read-only t)
  (rotation 0 :read-only t) (width 0 :read-only t) (height 0 :read-only t))

(defun pdf-check-bound (number maximum name)
  (unless (and (integerp number) (<= 1 number maximum))
    (invalid-pdf "~A must be an integer in [1,~D]." name maximum)))

(defun pdf-file-size (file maximum)
  (with-open-file (in file :element-type '(unsigned-byte 8))
    (let ((size (file-length in)))
      (when (> size maximum) (invalid-pdf "File ~A exceeds its ~D-byte budget." file maximum))
      size)))

(defun call-pdf-command (arguments directory stem deadline timeout-program)
  "Run a trusted local executable with argv, GNU timeout, and a POSIX file-size cap.
Every launched timeout process is waited on, including nonlocal exits. Its TERM
handler forwards cancellation; --kill-after bounds TERM-ignoring child lifetime."
  (let ((remaining (ceiling (- deadline (get-internal-real-time)) internal-time-units-per-second))
        (out (merge-pathnames (concatenate 'string stem ".out") directory))
        (err (merge-pathnames (concatenate 'string stem ".err") directory)))
    (unless (plusp remaining) (invalid-pdf "PDF transaction time budget exhausted."))
    ;; Explicit non-aborting close preserves diagnostics even on a nonlocal exit.
    ;; WITH-OPEN-FILE may delete a newly created output when unwinding with an error.
    (let ((stdout (open out :direction :output :if-exists :error))
          (stderr nil) (process nil) (joined nil) (status nil))
      (unwind-protect
           (progn
             (setf stderr (open err :direction :output :if-exists :error))
             (unwind-protect
                  (progn
                    (setf process
                          (uiop:launch-program
                           (append (list timeout-program "--signal=TERM" "--kill-after=1s"
                                         (format nil "~Ds" remaining)
                                         "/bin/sh" "-c" "ulimit -f 65536 || exit 125; exec \"$@\""
                                         "cl-docling-pdf" "/usr/bin/env" "LC_ALL=C") arguments)
                           :input nil :output stdout :error-output stderr))
                    (setf status (uiop:wait-process process) joined t))
               (when (and process (not joined))
                 ;; Do not SIGKILL timeout itself: it must forward TERM and own escalation.
                 (uiop:terminate-process process)
                 (uiop:wait-process process))))
        (when stderr (close stderr :abort nil))
        (close stdout :abort nil))
      (unless (eql status 0)
        (invalid-pdf "PDF tool ~A failed (exit ~A); inspect ~A. No successful page set was returned."
                     (first arguments) status directory)))
    (pdf-file-size out 1048576)
    (unless (zerop (pdf-file-size err 1048576))
      (invalid-pdf "PDF tool emitted diagnostics; inspect ~A. Output is not silently accepted." err))
    (uiop:read-file-string out)))

(defun pdf-words (line)
  (remove "" (uiop:split-string line :separator '(#\Space #\Tab #\Return)) :test #'string=))

(defun pdf-field (text prefix)
  (let ((matches (loop for line in (uiop:split-string text :separator '(#\Newline))
                       when (and (<= (length prefix) (length line))
                                 (string= prefix line :end2 (length prefix)))
                         collect (string-trim '(#\Space #\Tab #\Return) (subseq line (length prefix))))))
    (unless (= 1 (length matches)) (invalid-pdf "Expected one pdfinfo ~A field." prefix))
    (first matches)))

(defun pdf-decimal (text)
  "Strict nonnegative fixed-point decimal reader; never invoke the Lisp reader."
  (unless (and (<= 1 (length text) 20)
               (every (lambda (c) (or (digit-char-p c) (char= c #\.))) text)
               (<= (count #\. text) 1))
    (invalid-pdf "Invalid PDF dimension ~S." text))
  (let* ((point (position #\. text))
         (digits (remove #\. text)))
    (when (zerop (length digits)) (invalid-pdf "Empty PDF dimension."))
    (/ (parse-integer digits) (if point (expt 10 (- (length text) point 1)) 1))))

(defun pdf-paper-label-p (text)
  "Accept an optional parenthesized alphanumeric paper name, never dimensions."
  (and (stringp text) (<= 3 (length text) 34)
       (char= (char text 0) #\() (char= (char text (1- (length text))) #\))
       (every (lambda (c) (or (alphanumericp c) (char= c #\-)))
              (subseq text 1 (1- (length text))))))

(defun pdf-page-geometry (text page)
  (let ((size nil) (rotation nil))
    (dolist (line (uiop:split-string text :separator '(#\Newline)))
      (let ((words (pdf-words line)))
        (when (and (equal (first words) "Page")
                   (equal (second words) (write-to-string page)))
          (cond
            ((equal (third words) "size:")
             (unless (and (or (= 7 (length words))
                              (and (= 8 (length words)) (pdf-paper-label-p (eighth words))))
                          (equal (fifth words) "x") (equal (seventh words) "pts"))
               (invalid-pdf "Unsupported pdfinfo page-size syntax."))
             (when size (invalid-pdf "Duplicate page-size metadata."))
             (setf size (list (pdf-decimal (fourth words)) (pdf-decimal (sixth words)))))
            ((equal (third words) "rot:")
             (when rotation (invalid-pdf "Duplicate page rotation."))
             (setf rotation (parse-integer (fourth words))))))))
    (unless (and size (every #'plusp size) (member rotation '(0 90 180 270)))
      (invalid-pdf "Missing/unsupported page geometry or rotation for page ~D." page))
    (values (first size) (second size) rotation)))

(defun pdf-png-size (file max-side)
  "Read the fixed PNG IHDR only, before passing the file to the qualified decoder."
  (pdf-file-size file 33554432)
  (with-open-file (in file :element-type '(unsigned-byte 8))
    (let ((bytes (make-array 29 :element-type '(unsigned-byte 8))))
      (unless (and (= 29 (read-sequence bytes in))
                   (equalp (subseq bytes 0 16) #(137 80 78 71 13 10 26 10 0 0 0 13 73 72 68 82))
                   (= 8 (aref bytes 24)) (= 2 (aref bytes 25))
                   (zerop (aref bytes 26)) (zerop (aref bytes 27)) (zerop (aref bytes 28)))
        (invalid-pdf "Expected a non-interlaced 8-bit RGB Poppler PNG."))
      (flet ((u32 (offset) (loop for i from offset below (+ offset 4) for value = (aref bytes i)
                                then (+ (ash value 8) (aref bytes i)) finally (return value))))
        (let ((width (u32 16)) (height (u32 20)))
          (pdf-check-bound width max-side "PNG width")
          (pdf-check-bound height max-side "PNG height")
          (values width height))))))

(defun snapshot-pdf (source target maximum)
  (with-open-file (in source :element-type '(unsigned-byte 8))
    (when (> (file-length in) maximum) (invalid-pdf "PDF input exceeds byte budget."))
    (let ((header (make-array 5 :element-type '(unsigned-byte 8))))
      (unless (and (= 5 (read-sequence header in)) (equalp header #(37 80 68 70 45)))
        (invalid-pdf "Input must start with a PDF header.")))
    (file-position in 0)
    (with-open-file (out target :element-type '(unsigned-byte 8) :direction :output :if-exists :error)
      (let ((buffer (make-array 65536 :element-type '(unsigned-byte 8))) (total 0))
        (loop for count = (read-sequence buffer in) while (plusp count) do
          (when (> (incf total count) maximum) (invalid-pdf "PDF input grew beyond its byte budget."))
          (write-sequence buffer out :end count))))))

(defun pdf-page-selection-p (pages maximum)
  ;; Bounded traversal also rejects dotted/circular lists without calling LENGTH.
  (let ((seen nil) (tail pages))
    (loop repeat maximum do
      (when (null tail) (return-from pdf-page-selection-p (not (null seen))))
      (unless (and (consp tail) (integerp (car tail)) (<= 1 (car tail) 100000)
                   (not (member (car tail) seen)))
        (return-from pdf-page-selection-p nil))
      (push (car tail) seen)
      (setf tail (cdr tail)))
    (null tail)))

(defun rasterize-pdf (input output-directory &key (pages '(1)) (dpi 144) (max-pages 16)
                       (max-side 4096) (max-input-bytes 67108864) (timeout-seconds 60)
                       (pdfinfo "pdfinfo") (pdftoppm "pdftoppm") (timeout-program "timeout"))
  "Rasterize explicitly selected PDF pages using Poppler; no model or Python involved.
Requires SBCL/POSIX, Poppler and GNU timeout. OUTPUT-DIRECTORY must not exist;
its parent must exist. Failure artifacts are retained there for diagnosis, never
returned as successful pages. See docs/pdf.md for resource and trust boundaries."
  (handler-case
      (progn
        (pdf-check-bound dpi 300 "DPI")
        (pdf-check-bound max-pages 16 "Page count budget")
        (pdf-check-bound max-side 4096 "Raster side budget")
        (pdf-check-bound max-input-bytes 67108864 "Input byte budget")
        (pdf-check-bound timeout-seconds 120 "Time budget")
        (unless (pdf-page-selection-p pages max-pages)
          (invalid-pdf "PAGES must be a nonempty, bounded list of distinct positive page numbers."))
        (let* ((source (truename input))
               (directory (uiop:ensure-directory-pathname (merge-pathnames output-directory)))
               (snapshot (merge-pathnames "input.pdf" directory))
               (deadline (+ (get-internal-real-time) (* timeout-seconds internal-time-units-per-second)))
               (result nil))
          (pdf-file-size source max-input-bytes)
          ;; Atomic exclusive creation; do not reuse or recursively clear a user's directory.
          (sb-posix:mkdir (namestring directory) #o700)
          (setf directory (truename directory) snapshot (merge-pathnames "input.pdf" directory))
          (snapshot-pdf source snapshot max-input-bytes)
          (let* ((info (call-pdf-command (list pdfinfo (namestring snapshot)) directory "info" deadline timeout-program))
                 (page-count (parse-integer (pdf-field info "Pages:"))))
            (unless (string= "no" (pdf-field info "Encrypted:")) (invalid-pdf "Encrypted PDFs are unsupported."))
            (unless (every (lambda (n) (<= n page-count)) pages) (invalid-pdf "Requested page exceeds PDF page count.")))
          (dolist (page pages)
            (let* ((number (write-to-string page))
                   (stem (format nil "page-~D" page))
                   (prefix (merge-pathnames stem directory))
                   (png (merge-pathnames (concatenate 'string stem ".png") directory))
                   (info (call-pdf-command (list pdfinfo "-f" number "-l" number (namestring snapshot))
                                           directory (concatenate 'string stem "-info") deadline timeout-program)))
              (multiple-value-bind (points-width points-height rotation) (pdf-page-geometry info page)
                (when (member rotation '(90 270)) (rotatef points-width points-height))
                (let ((expected-width (ceiling (* points-width dpi) 72))
                      (expected-height (ceiling (* points-height dpi) 72)))
                  (pdf-check-bound expected-width max-side "Page raster width")
                  (pdf-check-bound expected-height max-side "Page raster height")
                  (call-pdf-command (list pdftoppm "-f" number "-l" number "-singlefile" "-png"
                                          "-r" (write-to-string dpi)
                                          "-W" (write-to-string max-side) "-H" (write-to-string max-side)
                                          (namestring snapshot) (namestring prefix))
                                    directory stem deadline timeout-program)
                  (multiple-value-bind (width height) (pdf-png-size png max-side)
                    ;; pdfinfo rounds points to decimals: allow at most one pixel of rounding.
                    (unless (and (<= (abs (- width expected-width)) 1) (<= (abs (- height expected-height)) 1))
                      (invalid-pdf "Raster size differs from PDF geometry; refusing possible clipping."))
                    (push (%make-pdf-page :number page :pathname png :dpi dpi :rotation rotation
                                          :width width :height height) result))))))
          (setf result (nreverse result))
          (with-open-file (out (merge-pathnames "manifest.sexp" directory) :direction :output :if-exists :error)
            (let ((*print-readably* t) (*print-pretty* t))
              (write (list :schema 1 :source (namestring source) :snapshot "input.pdf"
                           :rasterizer pdftoppm :metadata-tool pdfinfo :dpi dpi
                           :page-box :media :annotations :included :rotation :applied
                           :pages (mapcar (lambda (page)
                                            (list :number (pdf-page-number page)
                                                  :png (file-namestring (pdf-page-pathname page))
                                                  :width (pdf-page-width page) :height (pdf-page-height page)
                                                  :rotation (pdf-page-rotation page))) result)) :stream out)
              (terpri out)))
          result))
    (pdf-error (condition) (error condition))
    (error (condition) (invalid-pdf "PDF rasterization failed: ~A" condition))))

(export '(pdf-error pdf-error-message pdf-page pdf-page-number pdf-page-pathname
          pdf-page-dpi pdf-page-rotation pdf-page-width pdf-page-height rasterize-pdf))
