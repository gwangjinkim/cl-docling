(in-package #:docling)
(export '(save-document-adapter load-document-adapter merge-document-lora))

(defparameter +document-lora-targets+ "model.text_model.layers.\\d+.self_attn.(q_proj|v_proj)")

(defun save-document-adapter (model destination)
  "Save standard FP32 PEFT adapter artifacts to a NEW local directory.
The base is the original absolute local checkpoint path; no model copy or upload."
  (require-document-lora model)
  (let ((config (tbk:lora-config (document-decoder model))))
    (setf (gethash "target_modules" config) +document-lora-targets+
          (gethash "inference_mode" config) 'yason:true)
    (tbk:with-backend ((document-backend model))
      (tbk:save-checkpoint destination config (document-lora-parameters model) nil :kind :adapter))))

(defun document-adapter-schema (model rank)
  (unless (typep rank '(integer 1)) (invalid-layout "Adapter rank must be positive"))
  (loop for (name . tensor) in (tb:named-parameters (document-decoder model))
        when (or (search ".self_attn.q_proj.weight" name) (search ".self_attn.v_proj.weight" name))
          append (destructuring-bind (out in) (tb:tensor-shape tensor)
                   (loop for factor in '("A" "B") for shape in (list (list rank in) (list out rank))
                         collect (cons (format nil "base_model.model.~A.lora_~A.weight"
                                              (subseq (document-weight-name name) 0 (- (length (document-weight-name name)) 7)) factor)
                                       shape)))))

(defun load-document-adapter (model directory)
  "Attach one standard, decoder-q/v-only FP32 PEFT adapter. Borrow no file state.
Validate local base identity and complete coverage; failed loads leave MODEL unchanged."
  (require-document model)
  (when (document-lora-p model) (invalid-layout "A document adapter is already active"))
  (let* ((directory (uiop:ensure-directory-pathname directory))
         (config (read-json-file (merge-pathnames "adapter_config.json" directory)))
         (file (merge-pathnames "adapter_model.safetensors" directory)))
    (unless (and (hash-table-p config) (equal (gethash "target_modules" config) +document-lora-targets+)
                 (member (gethash "use_rslora" config) '(nil yason:false)))
      (invalid-layout "Only the exact document decoder q/v target pattern and standard LoRA are qualified"))
    (validate-vision-header file (document-adapter-schema model (gethash "r" config)) :allow-decoder nil)
    (maphash (lambda (name descriptor)
               (unless (or (equal name "__metadata__") (equal (gethash "dtype" descriptor) "F32"))
                 (invalid-layout "Document adapter weights must be FP32")))
             (safetensors-header file))
    ;; Translate only the validated document namespace/selector. Engine validates
    ;; every remaining PEFT option, base identity/revision, shape and finite value.
    (setf (gethash "target_modules" config) #("q_proj" "v_proj"))
    (let ((weights (tbk:load-weight-file file (document-backend model))))
      (unwind-protect
           (progn
             (tbk:install-lora-parameters (document-decoder model) config
               (loop for name being the hash-keys of weights using (hash-value tensor)
                     collect (cons (concatenate 'string "base_model.model.model."
                                               (subseq name (length "base_model.model.model.text_model."))) tensor)))
             (setf (document-lora-p model) t) model)
        (tbk:dispose-weights weights)))))

(defun merge-document-lora (model)
  "Fold active q/v adapter into decoder base weights and remove it; invalidates caches.
Then SAVE-PRETRAINED exports an ordinary Idefics3 model. Reload before attaching another adapter."
  (require-document-lora model)
  (tb:merge-adapter (document-decoder model))
  (setf (document-lora-p model) nil)
  model)
