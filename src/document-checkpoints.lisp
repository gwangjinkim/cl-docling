(in-package #:docling)
(export '(save-document-training-checkpoint load-document-training-checkpoint))

(defun save-document-training-checkpoint (model optimizer destination)
  "Save native decoder adapter + optimizer state under NEW DESTINATION/decoder/.
Not a document PEFT export. Base, image features and data cursor are not copied.
The engine atomically publishes the complete decoder payload; serialize access."
  (require-document-lora model)
  (let ((destination (uiop:ensure-directory-pathname destination)))
    (when (probe-file destination) (invalid-layout "Training checkpoint destination already exists"))
    (tb:save-training-checkpoint (document-decoder model) optimizer (merge-pathnames "decoder/" destination))
    destination))

(defun require-fp32-checkpoint-file (file)
  (maphash (lambda (name descriptor)
             (unless (or (equal name "__metadata__") (equal (gethash "dtype" descriptor) "F32"))
               (invalid-layout "Native document checkpoint tensors must be FP32")))
           (safetensors-header file)))

(defun load-document-training-checkpoint (base-directory checkpoint optimizer &key backend (device :cpu))
  "Return a NEW owned document model and restored step count; borrow fresh OPTIMIZER.
Reload the unchanged local base, native decoder adapter and matching optimizer state.
Dispose OPTIMIZER before returned model/backend. Failure disposes the partial model.
No image features, data cursor, Python optimizer, scheduler or runtime RNG is restored."
  (let* ((directory (merge-pathnames "decoder/" (uiop:ensure-directory-pathname checkpoint)))
         (state (merge-pathnames "training_state.json" directory)))
    (dolist (name '("adapter_config.json" "adapter_model.safetensors" "training_state.json"))
      (unless (probe-file (merge-pathnames name directory))
        (invalid-layout "Incomplete native document training checkpoint: missing ~A" name)))
    (let ((config (read-json-file (merge-pathnames "adapter_config.json" directory)))
          (manifest (read-json-file state)))
      (unless (and (hash-table-p config) (hash-table-p manifest)
                   (equal (gethash "model_kind" manifest) "adapter")
                   (equalp (gethash "target_modules" config) #("q_proj" "v_proj"))
                   (member (gethash "use_rslora" config) '(nil yason:false)))
        (invalid-layout "Only standard q/v document decoder training checkpoints are supported"))
      (let ((model (load-document-model base-directory :backend backend :device device)) (success nil))
        (unwind-protect
             (progn
               (validate-vision-header (merge-pathnames "adapter_model.safetensors" directory)
                 (loop for (name . shape) in (document-adapter-schema model (gethash "r" config))
                       collect (cons (concatenate 'string "base_model.model.model."
                                                  (subseq name (length "base_model.model.model.text_model."))) shape))
                 :allow-decoder nil)
               (require-fp32-checkpoint-file (merge-pathnames "adapter_model.safetensors" directory))
               (when (probe-file (merge-pathnames "optimizer.safetensors" directory))
                 (require-fp32-checkpoint-file (merge-pathnames "optimizer.safetensors" directory)))
               (tb:load-adapter (document-decoder model) directory)
               (tbk:with-backend ((document-backend model))
                 (dolist (p (tb:trainable-parameters (document-decoder model))) (tbk:evaluate (cdr p))))
               (setf (document-lora-p model) t)
               (let ((step (tb:restore-training-checkpoint (document-decoder model) optimizer directory)))
                 (setf success t)
                 (values model step)))
          (unless success (tb:dispose model)))))))
