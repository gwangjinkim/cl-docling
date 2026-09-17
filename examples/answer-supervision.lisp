;;; Offline, illustrative token IDs. No tokenizer, model or Python required.
(require :asdf)
(let ((root (uiop:pathname-parent-directory-pathname (uiop:pathname-directory-pathname *load-truename*))))
  (asdf:initialize-source-registry `(:source-registry (:directory ,root) :ignore-inherited-configuration))
  (asdf:initialize-output-translations `(:output-translations (t ,(merge-pathnames ".build/" root)) :ignore-inherited-configuration))
  (asdf:load-system "cl-docling"))
(multiple-value-bind (ids labels attention)
    (docling:make-answer-supervision #(1 9 9 2 4 5 7) 4
                                     :vocab-size 10 :image-token-id 9 :eos-token-id 7 :pad-to 9)
  (format t "Input IDs: ~S~%Labels:    ~S~%Attention: ~S~%" ids labels attention)
  (format t "Prompt/image labels ignored; answer 4, 5 and real EOS 7 supervised.~%")
  (format t "Padding also uses 7, but its labels are -100 and attention is zero.~%")
  (format t "Labels are unshifted: causal loss must shift exactly once. No training has run.~%"))
