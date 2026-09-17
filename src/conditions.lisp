(in-package #:cl-docling)

(define-condition layout-error (error)
  ((message :initarg :message :reader layout-error-message))
  (:report (lambda (condition stream)
             (write-string (layout-error-message condition) stream))))

(defun invalid-layout (control &rest arguments)
  (error 'layout-error :message (apply #'format nil control arguments)))

(defun require-positive-integer (value name)
  (unless (and (integerp value) (plusp value))
    (invalid-layout "~A must be a positive integer, got ~S" name value))
  value)
