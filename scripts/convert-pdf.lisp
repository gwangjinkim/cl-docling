;;; Explicit local CLI. Help and syntax checks load no model/backend/Python.
(require :asdf)
(defparameter *conversion-root*
  (uiop:pathname-parent-directory-pathname (uiop:pathname-directory-pathname *load-truename*)))
(asdf:initialize-source-registry `(:source-registry (:directory ,*conversion-root*) :ignore-inherited-configuration))
(asdf:initialize-output-translations
 `(:output-translations (t (,(merge-pathnames ".build/fasl/" *conversion-root*) :implementation))
                       :ignore-inherited-configuration))
(asdf:load-system "cl-docling/application")

(defun conversion-identity (checkpoint device)
  ;; SHA-256 is explicit local setup tooling, not model execution. Keep logs.
  (let* ((engine (asdf:system-source-directory "cl-transformer-blocks"))
         (suffix #+darwin "dylib" #-darwin "so")
         (logs (merge-pathnames (format nil ".build/identity-~D-~D/" (get-universal-time) (random 1000000000)) *conversion-root*))
         (files (append
                 (uiop:directory-files checkpoint)
                 (uiop:directory-files (merge-pathnames "src/" *conversion-root*))
                 (uiop:directory-files (merge-pathnames "src/" engine))
                 (list (merge-pathnames "cl-docling.asd" *conversion-root*)
                       (merge-pathnames "cl-transformer-blocks.asd" engine)
                       (merge-pathnames "scripts/convert-pdf.lisp" *conversion-root*)
                       (pathname (or (uiop:getenv "TB_MLX_LIBRARY")
                                     (merge-pathnames (format nil ".build/native/libtb_mlx.~A" suffix) engine)))
                       (pathname (or (uiop:getenv "TB_TOKENIZER_LIBRARY")
                                     (merge-pathnames (format nil ".build/tokenizer/release/libtb_tokenizer.~A" suffix) engine)))
                       (pathname (or (uiop:getenv "DOCLING_IMAGE_LIBRARY")
                                     (merge-pathnames (format nil ".build/native/libdocling_images.~A" suffix) *conversion-root*)))))))
    (setf files (sort (remove-duplicates (mapcar #'truename files) :test #'equal) #'string< :key #'namestring))
    (dolist (file files) (docling::application-regular-file file 8589934592))
    (sb-posix:mkdir (namestring logs) #o700)
    (let* ((text (docling::call-pdf-command
                  (append '("shasum" "-a" "256" "--") (mapcar #'namestring files)) logs "sha256"
                  (+ (get-internal-real-time) (* 120 internal-time-units-per-second)) "timeout"))
           (lines (remove "" (uiop:split-string text :separator '(#\Newline)) :test #'equal)))
      (unless (and (= (length files) (length lines))
                   (every (lambda (line) (and (>= (length line) 66)
                                              (every (lambda (c) (digit-char-p c 16)) (subseq line 0 64)))) lines))
        (error "Unexpected SHA-256 tool output; inspect ~A" logs))
      (append (list "native-cli-v1" (string device) (lisp-implementation-version)
                    (software-type) (software-version) (namestring checkpoint))
              (loop for file in files for line in lines
                    collect (format nil "~A ~A" (subseq line 0 64) (namestring file)))))))

(let ((options (handler-case (docling:parse-conversion-arguments (uiop:command-line-arguments))
                 (error (c) (format *error-output* "~&Argument error: ~A~%" c) (uiop:quit 64)))))
  (when (eq options :help)
    (format t "Usage: bin/cl-docling --input FILE.pdf --output NEW-DIRECTORY [options]~%~
Options: --model DIRECTORY --pages 1,3-5 --dpi 144 --max-new-tokens 512~%~
         --device cpu|gpu --task TEXT --resume~%~
Default selection is PAGE 1 ONLY; at most 16 increasing pages per job.~%~
Requires prepared model/native libraries, Poppler, GNU timeout and shasum.~%~
Resume requires identical source bytes, options and checkpoint/source/direct-library hashes.~%~
Exit codes: 0 complete export, 2 blocked DocTags, 3 page failure, 1 setup/job error, 64 bad arguments.~%")
    (uiop:quit 0))
  (handler-case
      (progn
        (load (merge-pathnames "scripts/load-mlx.lisp" *conversion-root*))
        (asdf:load-system "cl-docling/pipeline")
        (let* ((checkpoint (truename (uiop:ensure-directory-pathname
                            (or (getf options :model) (uiop:getenv "DOCLING_MODEL")
                                (merge-pathnames ".build/smoldocling/" *conversion-root*)))))
               (device (getf options :device))
               (task (or (getf options :task) "Convert this page to docling."))
               (budget (getf options :max-new-tokens))
               (identity (conversion-identity checkpoint device))
               (backend nil) (model nil) (status nil))
          (unwind-protect
               (multiple-value-bind (bundle result report)
                   (docling:run-pdf-job (getf options :input) (getf options :output)
                     (lambda (image number)
                       (unless backend (setf backend (uiop:symbol-call :tb :make-backend :device device)))
                       (unless model (setf model (uiop:symbol-call :docling :load-document-model checkpoint :backend backend)))
                       (format t "~&Converting page ~D on ~A...~%" number device) (finish-output)
                       (uiop:symbol-call :docling :generate-image model image :task task :max-new-tokens budget))
                     :identity identity :pages (getf options :pages) :dpi (getf options :dpi)
                     :task task :max-new-tokens budget :resume (getf options :resume))
                 (setf status result)
                 (format t "~&Status: ~A (export status, not OCR accuracy).~%Report: ~A~%" result report)
                 (when bundle (format t "Bundle: ~A~%" bundle)))
            (when model (uiop:symbol-call :tb :dispose model))
            (when backend
              (unwind-protect
                   (let ((handles (getf (uiop:symbol-call :tb :backend-memory backend) :handles)))
                     (format t "~&Remaining tensor handles: ~D~%" handles)
                     (unless (zerop handles) (error "Native tensor handle leak.")))
                (uiop:symbol-call :tb :dispose backend))))
          (uiop:quit (ecase status (:complete 0) (:blocked 2) (:failed 3)))))
    (error (condition) (format *error-output* "~&Conversion failed: ~A~%" condition) (uiop:quit 1))))
