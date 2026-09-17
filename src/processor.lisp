;;; Pinned Idefics3 prompt behavior adapted from Transformers (Apache-2.0).
;;; See THIRD-PARTY.md. No general Jinja interpreter or Python fallback.
(in-package #:docling)
(export '(prepare-image-input document-tile-features generate-image))

(defparameter +smoldocling-chat-template+
  (format nil "<|im_start|>{% for message in messages %}{{message['role'] | capitalize}}{% if message['content'][0]['type'] == 'image' %}{{':'}}{% else %}{{': '}}{% endif %}{% for line in message['content'] %}{% if line['type'] == 'text' %}{{line['text']}}{% elif line['type'] == 'image' %}{{ '<image>' }}{% endif %}{% endfor %}<end_of_utterance>~%{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"))

(defun document-json-asset (model name)
  (let ((bytes (cdr (assoc name (document-assets model) :test #'equal))))
    (unless bytes (invalid-layout "Missing required snapshotted asset ~A" name))
    (let ((*read-default-float-format* 'double-float))
      (let ((object (handler-case
                        (yason:parse (babel:octets-to-string bytes :encoding :utf-8)
                                     :json-arrays-as-vectors t :json-booleans-as-symbols t)
                      (error () (invalid-layout "Invalid JSON asset ~A" name)))))
        (unless (hash-table-p object) (invalid-layout "Expected an object in asset ~A" name))
        object))))

(defun require-json-policy (object expected label)
  (unless (and (hash-table-p object) (= (hash-table-count object) (length expected))
               (every (lambda (entry) (equalp (gethash (car entry) object :missing) (cdr entry))) expected))
    (invalid-layout "Unsupported ~A; expected the pinned SmolDocling policy" label)))

(defun document-image-processor (model)
  "Validate snapshotted processor assets, not mutable files in the source directory."
  (require-document model)
  (let* ((config (document-json-asset model "preprocessor_config.json"))
         (layout (vision-spec-layout (model-vision-spec (document-vision model)))))
    (require-json-policy (gethash "size" config) '(("longest_edge" . 2048)) "resize size")
    (require-json-policy (gethash "max_image_size" config) '(("longest_edge" . 512)) "tile size")
    (require-json-policy config
      (append '(("do_convert_rgb" . yason:true) ("do_image_splitting" . yason:true)
                ("do_normalize" . yason:true) ("do_pad" . yason:true) ("do_rescale" . yason:true)
                ("do_resize" . yason:true) ("image_mean" . #(0.5d0 0.5d0 0.5d0))
                ("image_std" . #(0.5d0 0.5d0 0.5d0)) ("resample" . 1)
                ("image_processor_type" . "Idefics3ImageProcessor") ("processor_class" . "Idefics3Processor"))
              (list (cons "size" (gethash "size" config)) (cons "max_image_size" (gethash "max_image_size" config))
                    (cons "rescale_factor" (/ 1d0 255d0)))) "image processor")
    (require-json-policy (document-json-asset model "processor_config.json")
                        '(("image_seq_len" . 64) ("processor_class" . "Idefics3Processor")) "processor")
    (require-json-policy (document-json-asset model "chat_template.json")
                        (list (cons "chat_template" +smoldocling-chat-template+)) "chat template")
    (let ((tokenizer (document-json-asset model "tokenizer_config.json")))
      (dolist (entry (list '("add_prefix_space" . yason:false)
                           '("clean_up_tokenization_spaces" . yason:false)
                           '("tokenizer_class" . "GPT2Tokenizer")
                           '("padding_side" . "right") '("truncation_side" . "right")
                           (cons "chat_template" +smoldocling-chat-template+)))
        (unless (equalp (gethash (car entry) tokenizer :missing) (cdr entry))
          (invalid-layout "Unsupported tokenizer setting ~A" (car entry))))
      (dolist (key '("add_bos_token" "add_eos_token"))
        (when (gethash key tokenizer) (invalid-layout "Unqualified tokenizer setting ~A" key))))
    (when (assoc "chat_template.jinja" (document-assets model) :test #'equal)
      (invalid-layout "Alternative chat_template.jinja asset is not qualified"))
    (unless (and (= 512 (vision-layout-image-size layout)) (= 64 (image-token-count layout)))
      (invalid-layout "Processor tile/token sizes disagree with the model"))
    (let ((tokens (tb:encode-text model "<image>" :add-special-tokens nil)))
      (unless (and (= 1 (length tokens)) (= (aref tokens 0) (gethash "image_token_id" (document-config model))))
        (invalid-layout "Tokenizer image ID disagrees with model")))
    (make-image-processor)))

(defun validate-image-task (task)
  (unless (and (stringp task) (<= (length task) 32768) (not (find #\Null task))
               (notany (lambda (marker) (search marker task))
                       '("<image>" "<fake_token_around_image>" "<global-img>" "<row_"
                         "<|im_start|>" "<|im_end|>" "<end_of_utterance>")))
    (invalid-layout "Task must be bounded text without reserved image/chat markers or NUL")))

(defun prepare-image-input (model file &key (task "Convert this page to docling."))
  "One PNG -> host (1 time) IDs, (1 tiles 3 H W) FP32 pixels, tile count, prompt string.
Owns returned Lisp arrays. Exact pinned single-user, image-first chat template only."
  (validate-image-task task)
  (let ((processor (document-image-processor model)))
    (multiple-value-bind (pixels mask grids) (preprocess-images processor (list file))
      (declare (ignore mask)) ; one image: every tile is real and every pixel valid
      (let* ((count (array-dimension pixels 1))
             (prompt (format nil "<|im_start|>User:~A~A<end_of_utterance>~%Assistant:"
                             (image-prompt processor (caar grids) (cadar grids)) task))
             (tokens (tb:encode-text model prompt :add-special-tokens t))
             (ids (make-array (list 1 (length tokens)) :element-type '(unsigned-byte 32))))
        (unless (= (* count 64) (count (gethash "image_token_id" (document-config model)) tokens))
          (invalid-layout "Prompt image IDs disagree with tile count"))
        (when (> (length tokens) (gethash "max_position_embeddings" (tb:model-config (document-decoder model))))
          (invalid-layout "Image prompt exceeds model context"))
        (dotimes (i (length tokens)) (setf (aref ids 0 i) (aref tokens i)))
        (values ids pixels count prompt)))))

(defun document-tile-features (model pixels)
  "Single image, host FP32 (1 tiles 3 H W) -> owned (1 tiles*tokens hidden) native features.
Encode tiles sequentially and evaluate each, bounding simultaneous encoder work."
  (require-document model)
  (let* ((side (vision-layout-image-size (vision-spec-layout (model-vision-spec (document-vision model)))))
         (count (and (arrayp pixels) (= 5 (array-rank pixels)) (array-dimension pixels 1))))
    (unless (and count (<= 1 count 17) (equal (array-dimensions pixels) (list 1 count 3 side side))
                 (equal (array-element-type pixels) 'single-float))
      (invalid-layout "Expected FP32 pixels (1 tiles 3 ~D ~D), 1..17 unpadded tiles" side side))
    (let ((features nil))
      (unwind-protect
           (progn
             (dotimes (i count)
               (let ((tile (make-array (list 1 3 side side) :element-type 'single-float
                                      :displaced-to pixels :displaced-index-offset (* i 3 side side))))
                 (tb:with-resource (native (tb:tensor-from-array (document-backend model) tile))
                   (let ((feature (document-image-features model native)))
                     (push feature features)
                     (tbk:with-backend ((document-backend model)) (tbk:evaluate feature))))))
             (tbk:with-backend ((document-backend model))
               (tbk:retain (reduce (lambda (left right) (tbk:concatenate-tensors left right 1)) (reverse features)))))
        (mapc #'tb:dispose features)))))

(defun generate-image (model file &key (task "Convert this page to docling.") (max-new-tokens 128)
                                      (eos-token-id (document-eos-token-id model)))
  "One PNG -> raw DocTags string, new ID vector, :EOS/:LENGTH. No Markdown repair.
Greedy FP32; every image tile is encoded once, before cached generation."
  (require-document model)
  (unless (and (typep max-new-tokens '(integer 0))
               (or (null eos-token-id) (typep eos-token-id `(integer 0 (,(gethash "vocab_size" (tb:model-config (document-decoder model))))))))
    (invalid-layout "Invalid token budget or EOS"))
  (multiple-value-bind (ids pixels count) (prepare-image-input model file :task task)
    (when (> (+ (array-dimension ids 1) max-new-tokens)
             (gethash "max_position_embeddings" (tb:model-config (document-decoder model))))
      (invalid-layout "Prompt plus generation budget exceeds context"))
    (when (zerop max-new-tokens) (return-from generate-image (values "" #() :length)))
    (tb:with-resource (features (document-tile-features model pixels))
      (multiple-value-bind (tokens reason)
          (generate-document model ids nil :image-features features :tile-count count
                              :max-new-tokens max-new-tokens :eos-token-id eos-token-id)
        (values (tb:decode-tokens model tokens) tokens reason)))))
