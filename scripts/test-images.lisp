(require :asdf)
(let ((root (uiop:pathname-parent-directory-pathname
             (uiop:pathname-directory-pathname *load-truename*))))
  (handler-case
      (progn (load (merge-pathnames "scripts/load-images.lisp" root))
             (asdf:load-system "yason")
             (load (merge-pathnames "tests/images.lisp" root)))
    (error (e) (format *error-output* "~&FAIL: ~A~%" e) (uiop:quit 1))))
