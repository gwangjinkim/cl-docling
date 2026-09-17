;;; Bounded real-checkpoint smoke: one preprocessed tile, two synthetic targets.
;;; Not a quality experiment or a full real-model gradient-parity gate.
(load (merge-pathnames "load-mlx.lisp" *load-truename*))
(let* ((device (ecase (intern (string-upcase (or (uiop:getenv "TB_DEVICE") "cpu")) :keyword)
                 (:cpu :cpu) (:gpu :gpu)))
       (checkpoint (uiop:ensure-directory-pathname (uiop:getenv "DOCLING_MODEL")))
       (fixture (uiop:ensure-directory-pathname (uiop:getenv "DOCLING_FIXTURE"))))
  (tb:with-resource (backend (tb:make-backend :device device))
    (let ((baseline (getf (tb:backend-memory backend) :handles)))
      (tb:with-resource (model (docling:load-document-model checkpoint :backend backend))
        (let ((data (tbk:load-weight-file (merge-pathnames "reference.safetensors" fixture) backend)))
          (unwind-protect
               (let* ((prompt (tb:tensor-array (gethash "input_ids" data)))
                      (n (array-dimension prompt 1))
                      (ids (make-array (list 1 (+ n 2))))
                      (labels (make-array (list 1 (+ n 2)) :initial-element -100)))
                 (dotimes (i n) (setf (aref ids 0 i) (round (aref prompt 0 i))))
                 ;; In-vocabulary synthetic labels: deliberately no document-quality claim.
                 (setf (aref ids 0 n) 3 (aref ids 0 (1+ n)) 4
                       (aref labels 0 n) 3 (aref labels 0 (1+ n)) 4)
                 (tb:with-resource (features (docling:document-image-features model (gethash "pixels" data)))
                   (docling:make-document-lora model :rank 2 :alpha 4)
                   (multiple-value-bind (loss gradients)
                       (docling:document-loss-and-gradients model ids features :labels labels)
                     (unwind-protect
                          (progn
                            (assert (= (length gradients) (length (docling:document-lora-parameters model))))
                            (assert (some (lambda (p)
                                            (let ((a (tb:tensor-array (cdr p))))
                                              (loop for i below (array-total-size a)
                                                    thereis (not (zerop (row-major-aref a i)))))) gradients))
                            (format t "~&Real ~A: ~D tokens, ~D finite LoRA gradients, loss ~,8F.~%"
                                    device (+ n 2) (length gradients) loss))
                       (mapc (lambda (p) (tb:dispose (cdr p))) gradients)))
                   (tb:with-resource (optimizer (tb:make-sgd :learning-rate 0.001))
                     (let ((before (docling:document-train-step model optimizer ids features :labels labels)))
                       (multiple-value-bind (after gradients)
                           (docling:document-loss-and-gradients model ids features :labels labels)
                         (unwind-protect
                              (progn (assert (< after before))
                                     (format t "Real ~A: one update ~,8F -> ~,8F.~%" device before after))
                           (mapc (lambda (p) (tb:dispose (cdr p))) gradients)))))))
            (tbk:dispose-weights data))))
      (assert (= baseline (getf (tb:backend-memory backend) :handles)))
      (format t "PASS: bounded real-checkpoint training smoke on ~A; all tensor handles released.~%" device))))
