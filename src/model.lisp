(in-package #:docling)
(export '(document-model load-document-model document-image-features merge-image-embeddings forward-document))

(defun document-text-config (config)
  "Normalize audited inactive training/legacy metadata; reject active variants."
  (let* ((text (gethash "text_config" config)) (canonical (make-hash-table :test 'equal)))
    (unless (member (gethash "tie_word_embeddings" config 'yason:false) '(yason:true yason:false))
      (invalid-layout "Root tying policy must be a JSON boolean"))
    (dolist (key '("use_resampler" "qk_layer_norms"))
      (unless (member (gethash key text) '(nil yason:false))
        (invalid-layout "Unsupported text feature ~A" key)))
    (unless (zerop (gethash "neftune_noise_alpha" text 0))
      (invalid-layout "Training embedding noise is unsupported"))
    (unless (= (gethash "pixel_shuffle_factor" text (gethash "scale_factor" config 2))
               (gethash "scale_factor" config 2))
      (invalid-layout "Conflicting legacy pixel-shuffle factor"))
    (maphash (lambda (key value)
               (unless (member key '("_flash_attn_2_enabled" "neftune_noise_alpha" "perceiver_config"
                                    "pixel_shuffle_factor" "qk_layer_norms" "use_resampler"
                                    "transformers.js_config") :test #'equal)
                 (setf (gethash key canonical) value))) text)
    ;; The outer conditional LM owns the output head and tying policy. The nested
    ;; LlamaModel has no head; its own tie flag does not define the outer head.
    (setf (gethash "tie_word_embeddings" canonical) (gethash "tie_word_embeddings" config 'yason:false))
    (tbk:decoder-schema canonical)
    (unless (typep (gethash "image_token_id" config) `(integer 0 (,(gethash "vocab_size" canonical))))
      (invalid-layout "Image token is outside text vocabulary"))
    (when (and (gethash "vocab_size" config)
               (/= (gethash "vocab_size" config) (gethash "vocab_size" canonical)))
      (invalid-layout "Root and nested vocabularies disagree"))
    canonical))

(defun document-weight-name (decoder-name)
  (if (equal decoder-name "lm_head.weight") decoder-name
      (concatenate 'string "model.text_model." (subseq decoder-name 6))))

(defclass document-model ()
  ((vision :initarg :vision :reader document-vision)
   (decoder :initarg :decoder :reader document-decoder)
   (config :initarg :config :reader document-config)
   (assets :initarg :assets :reader document-assets)
   (eos-token-id :initarg :eos-token-id :reader document-eos-token-id)
   (backend :initarg :backend :reader document-backend)
   (owns-backend :initarg :owns-backend :reader document-owns-backend-p)
   (lora :initform nil :accessor document-lora-p)
   (disposed :initform nil :accessor document-disposed-p)))

(defun require-document (model)
  (when (document-disposed-p model) (invalid-layout "Document model is disposed")))

(defmethod tb:dispose ((model document-model))
  (unless (document-disposed-p model)
    (tb:dispose (document-vision model))
    (tb:dispose (document-decoder model))
    (when (document-owns-backend-p model) (tb:dispose (document-backend model)))
    (setf (document-disposed-p model) t)))

(defmethod tb:named-parameters ((model document-model))
  (require-document model)
  (sort (append (loop for name being the hash-keys of (vision-weights (document-vision model))
                     using (hash-value tensor) collect (cons name tensor))
                (loop for (name . tensor) in (tb:named-parameters (document-decoder model))
                      collect (cons (document-weight-name name) tensor))) #'string< :key #'car))

(defmethod tb:make-cache ((model document-model) &key (growth-step 256))
  (require-document model)
  (tb:make-cache (document-decoder model) :growth-step growth-step))

(defmethod tb:decode-tokens ((model document-model) ids &key (skip-special-tokens nil))
  (require-document model)
  (tb:decode-tokens (document-decoder model) ids :skip-special-tokens skip-special-tokens))

(defmethod tb:encode-text ((model document-model) text &key (add-special-tokens t))
  (require-document model)
  (tb:encode-text (document-decoder model) text :add-special-tokens add-special-tokens))

(defparameter +document-asset-names+
  '("tokenizer.json" "tokenizer_config.json" "special_tokens_map.json" "added_tokens.json"
    "vocab.json" "merges.txt" "generation_config.json" "chat_template.json" "chat_template.jinja"
    "preprocessor_config.json" "processor_config.json" "README.md"))

(defun snapshot-document-assets (directory)
  (loop for name in +document-asset-names+ for path = (merge-pathnames name directory)
        when (probe-file path)
          collect (cons name (with-open-file (s path :element-type '(unsigned-byte 8))
                               (let ((bytes (make-array (file-length s) :element-type '(unsigned-byte 8))))
                                 (unless (= (length bytes) (read-sequence bytes s))
                                   (invalid-layout "Incomplete asset ~A" name))
                                 bytes)))))

(defun load-document-model (directory &key backend (device :cpu))
  "Load a complete local Idefics3/Llama single-file checkpoint, with FP32 execution.
Backend is borrowed if provided. Snapshot tokenizer assets when present. No Python."
  (let* ((directory (uiop:ensure-directory-pathname directory))
         (config (read-json-file (merge-pathnames "config.json" directory)))
         (spec (parse-vision-spec config)) (text (document-text-config config))
         (vision-schema (vision-schema spec)) (decoder-schema (tbk:decoder-schema text))
         (schema (append vision-schema
                         (loop for (name . shape) in decoder-schema collect (cons (document-weight-name name) shape))))
         (file (merge-pathnames "model.safetensors" directory))
         (assets (snapshot-document-assets directory))
         (generation (cdr (assoc "generation_config.json" assets :test #'equal)))
         (eos (if generation
                  (gethash "eos_token_id" (yason:parse (babel:octets-to-string generation :encoding :utf-8)))
                  (gethash "eos_token_id" text 2))))
    (unless (or (null eos) (typep eos `(integer 0 (,(gethash "vocab_size" text)))))
      (invalid-layout "Only a scalar in-vocabulary generation EOS is supported"))
    ;; The first full-model contract accepts canonical (single-copy) tied heads.
    ;; Duplicated tied heads and sharded checkpoints are rejected before allocation.
    (validate-vision-header file schema :allow-decoder nil)
    (let* ((owned (null backend)) (backend (or backend (tb:make-backend :device device)))
           (weights nil) (vision nil) (decoder nil) (success nil))
      (unwind-protect
           (progn
             (setf weights (tbk:load-weight-file file backend))
             (setf decoder (tbk:make-decoder text
                            (loop for (name . shape) in decoder-schema
                                  collect (cons name (gethash (document-weight-name name) weights))) backend
                            :source-id (namestring (truename directory))))
             (when (probe-file (merge-pathnames "tokenizer.json" directory))
               (tb:attach-tokenizer decoder directory))
             (setf vision (make-instance 'vision-model :spec spec :backend backend :owns-backend nil
                                          :weights (make-hash-table :test 'equal)))
             (dolist (entry vision-schema)
               (setf (gethash (car entry) (vision-weights vision)) (gethash (car entry) weights))
               (remhash (car entry) weights))
             (let ((model (make-instance 'document-model :vision vision :decoder decoder :config config
                                         :backend backend :owns-backend owned :assets assets :eos-token-id eos)))
               (setf success t) model))
        (when weights (tbk:dispose-weights weights))
        (unless success
          (tb:dispose vision) (tb:dispose decoder)
          (when owned (tb:dispose backend)))))))

(defmethod tb:save-pretrained ((model document-model) destination &key max-shard-size)
  "Export FP32 canonical weights and snapshotted assets to a NEW local directory.
No remote publication, model update or training is performed. Single-file only."
  (require-document model)
  (when max-shard-size (invalid-layout "Sharded document checkpoints are not yet qualified"))
  (when (document-lora-p model)
    (invalid-layout "Merge document LoRA before full-model export, or use SAVE-DOCUMENT-ADAPTER"))
  (let ((config (yason:parse (with-output-to-string (s) (yason:encode (document-config model) s))
                            :json-arrays-as-vectors t :json-booleans-as-symbols t)))
    (dolist (part (list config (gethash "vision_config" config) (gethash "text_config" config)))
      (setf (gethash "dtype" part) "float32")
      (when (gethash "torch_dtype" part) (setf (gethash "torch_dtype" part) "float32")))
    (tbk:with-backend ((document-backend model))
      (tbk:save-checkpoint destination config (tb:named-parameters model) (document-assets model)))))

(defun document-image-features (model pixels &key patch-mask)
  "Return owned visual features, one fixed-size preprocessed tile per batch row."
  (require-document model)
  (encode-vision (document-vision model) pixels :patch-mask patch-mask))

(defun merge-image-embeddings (model ids features &key (tile-count 1))
  "Replace exact image slots with FEATURES. Multiple tiles require a single text row.
Return owned embeddings. Integer gathers preserve selected float values exactly."
  (require-document model)
  (unless (and (typep tile-count '(integer 1 17))
               (or (= tile-count 1) (and (arrayp ids) (= (array-rank ids) 2) (= (array-dimension ids 0) 1))))
    (invalid-layout "Multiple tiles require one text row and an explicit count in 1..17"))
  (tb:with-resource (text (tbk:embed-tokens (document-decoder model) ids))
    (let* ((shape (tb:tensor-shape text)) (batch (first shape)) (steps (second shape)) (hidden (third shape))
           (count (* tile-count (image-token-count (vision-spec-layout (model-vision-spec (document-vision model))))))
           (token (gethash "image_token_id" (document-config model)))
           (indices (make-array (list batch steps) :element-type '(signed-byte 32))))
      (unless (and (eq :float32 (tb:tensor-dtype features))
                   (equal (tb:tensor-shape features) (list batch count hidden)))
        (invalid-layout "Expected ~D image features per batch row (~D tiles)" count tile-count))
      (dotimes (b batch)
        (let ((image-index 0))
          (dotimes (i steps)
            (setf (aref indices b i)
                  (if (= (aref ids b i) token)
                      (prog1 (+ (* batch steps) (* b count) image-index) (incf image-index))
                      (+ (* b steps) i))))
          (unless (= count image-index)
            (invalid-layout "Row ~D has ~D image slots; expected ~D" b image-index count))))
      (tbk:with-backend ((document-backend model))
        (tbk:retain
         (tbk:take-indices
          (tbk:concatenate-tensors (tbk:reshape text (list (* batch steps) hidden))
                                   (tbk:reshape features (list (* batch count) hidden)) 0)
          (tb:tensor-from-array (document-backend model) indices :dtype :int32)))))))

(defun forward-document (model ids &key pixels patch-mask image-features cache attention-mask (tile-count 1))
  "Return owned logits. Insert images only at cache offset zero.
PIXELS and IMAGE-FEATURES are exclusive. Cached continuation takes only token IDs.
Native causal/right-padded decoder semantics apply; no image processing is done."
  (require-document model)
  (when (or (and pixels image-features) (and patch-mask (not pixels)))
    (invalid-layout "Pixels/features are exclusive; patch-mask requires pixels"))
  (unless (and (typep tile-count '(integer 1 17)) (or (= tile-count 1) image-features))
    (invalid-layout "Multiple tiles require explicit image features"))
  (let ((continuation (and cache (plusp (tb:cache-length cache)))))
    (when (and continuation (or pixels image-features))
      (invalid-layout "Images must not be reinserted during cached decode"))
    (cond ((or pixels image-features)
           (let ((features (or image-features (document-image-features model pixels :patch-mask patch-mask))))
             (unwind-protect
                  (tb:with-resource (merged (merge-image-embeddings model ids features :tile-count tile-count))
                    (tbk:forward-embeddings (document-decoder model) merged :cache cache :attention-mask attention-mask))
               (unless image-features (tb:dispose features)))))
          (t
           (when (and (not continuation) (arrayp ids)
                      (loop for i below (array-total-size ids)
                            thereis (eql (row-major-aref ids i) (gethash "image_token_id" (document-config model)))))
             (invalid-layout "Image slots need exactly matching image features"))
           (tb:forward (document-decoder model) ids :cache cache :attention-mask attention-mask)))))
