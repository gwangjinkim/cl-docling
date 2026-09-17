;;; Fixed M5.6 experiment; shared generation helpers have no main entry point.
(load (merge-pathnames "experiment-common.lisp" *load-truename*))

(let* ((arguments (uiop:command-line-arguments))
       (fixtures (uiop:ensure-directory-pathname (first arguments)))
       (output (uiop:ensure-directory-pathname (second arguments)))
       (manifest (with-open-file (s (merge-pathnames "manifest.json" fixtures)) (yason:parse s)))
       (cases (gethash "cases" manifest))
       (protocol (gethash "protocol" manifest))
       (device (ecase (intern (string-upcase (or (uiop:getenv "TB_DEVICE") "gpu")) :keyword)
                 (:cpu :cpu) (:gpu :gpu)))
       (checkpoint (or (uiop:getenv "DOCLING_MODEL") (error "Set DOCLING_MODEL."))))
  (when (probe-file output) (error "Experiment output already exists: ~A" output))
  (ensure-directories-exist output)
  (dolist (case cases)
    (unless (eq (not (null (gethash "answer" case))) (string= "train" (gethash "split" case)))
      (error "Only training cases may supply an answer.")))
  (tb:with-resource (model (docling:load-document-model checkpoint :device device))
    (experiment-generate model cases fixtures output "base" device protocol)
    (let ((examples nil) (losses nil))
      (unwind-protect
           (progn
             (dolist (case cases)
               (when (string= "train" (gethash "split" case))
                 (let ((name (gethash "name" case)))
                   (format t "~&Preparing training features: ~A~%" name) (finish-output)
                   (multiple-value-bind (ids labels attention features tiles start)
                       (docling:prepare-image-training-example model
                         (merge-pathnames (format nil "~A.png" name) fixtures) (gethash "answer" case)
                         :task (gethash "task" protocol))
                     (push (list name ids labels attention features tiles start) examples)))))
             (setf examples (nreverse examples))
             (docling:make-document-lora model :rank (gethash "rank" protocol)
                                              :alpha (gethash "alpha" protocol) :seed (gethash "seed" protocol))
             (tb:with-resource (optimizer (tb:make-adamw
                                           :learning-rate (gethash "learning_rate" protocol)
                                           :beta1 (gethash "beta1" protocol) :beta2 (gethash "beta2" protocol)
                                           :epsilon (gethash "epsilon" protocol) :weight-decay (gethash "weight_decay" protocol)))
               (dotimes (epoch (gethash "epochs" protocol))
                 (dolist (example examples)
                   (destructuring-bind (name ids labels attention features tiles start) example
                     (let ((loss (docling:document-train-step model optimizer ids features
                                    :labels labels :attention-mask attention :tile-count tiles
                                    :max-grad-norm (gethash "max_grad_norm" protocol))))
                       (push (experiment-object "epoch" (1+ epoch) "page" name "loss" loss
                                                "tiles" tiles "prompt_tokens" start "positions" (array-dimension ids 1)) losses)
                       (format t "~&Epoch ~D ~A: loss ~,8F~%" (1+ epoch) name loss) (finish-output)))))
               (docling:save-document-adapter model (merge-pathnames "adapter/" output)))
             (experiment-write (merge-pathnames "losses.json" output) (coerce (nreverse losses) 'vector) t)
             (experiment-generate model cases fixtures output "adapted" device protocol))
        (dolist (example examples) (tb:dispose (fifth example)))))))
