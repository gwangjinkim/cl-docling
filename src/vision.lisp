(in-package #:docling)

(defclass vision-model ()
  ((spec :initarg :spec :reader model-vision-spec)
   (backend :initarg :backend :reader vision-backend)
   (owns-backend :initarg :owns-backend :reader owns-vision-backend-p)
   (weights :initarg :weights :reader vision-weights)
   (disposed :initform nil :accessor vision-disposed-p)))

(defmethod tb:dispose ((model vision-model))
  (unless (vision-disposed-p model)
    (tbk:dispose-weights (vision-weights model))
    (when (owns-vision-backend-p model) (tb:dispose (vision-backend model)))
    (setf (vision-disposed-p model) t)))

(defun vision-weight-count (model) (hash-table-count (vision-weights model)))

(defun load-vision-model (directory &key backend (device :cpu))
  "Load an Idefics3 vision tower/connector from local safetensors, promoted to FP32.
Accepts vision.safetensors or a single standard model.safetensors (decoder weights
are discarded). BACKEND, if supplied, is borrowed; otherwise the model owns it."
  (let* ((directory (uiop:ensure-directory-pathname directory))
         (spec (parse-vision-spec (read-json-file (merge-pathnames "config.json" directory))))
         (schema (vision-schema spec))
         (file (or (probe-file (merge-pathnames "vision.safetensors" directory))
                   (probe-file (merge-pathnames "model.safetensors" directory))
                   (invalid-layout "No single-file vision/model safetensors checkpoint"))))
    (validate-vision-header file schema)
    (let* ((owned (null backend)) (backend (or backend (tb:make-backend :device device)))
           (weights nil) (success nil))
      (unwind-protect
           (progn
             (setf weights (tbk:load-weight-file file backend))
             (dolist (name (loop for name being the hash-keys of weights
                                unless (assoc name schema :test #'equal) collect name))
               (tb:dispose (gethash name weights)) (remhash name weights))
             (dolist (entry schema)
               (unless (equal (tb:tensor-shape (gethash (car entry) weights)) (cdr entry))
                 (invalid-layout "Loaded weight shape differs: ~A" (car entry))))
             (setf success t)
             (make-instance 'vision-model :spec spec :weights weights :backend backend :owns-backend owned))
        (unless success
          (when weights (tbk:dispose-weights weights))
          (when owned (tb:dispose backend)))))))

(defun vision-weight (model name)
  (or (gethash name (vision-weights model)) (invalid-layout "Missing vision weight ~A" name)))

(defmethod patchify-image ((pixels tb:tensor) patch-size)
  (destructuring-bind (batch channels height width) (tb:tensor-shape pixels)
    (require-positive-integer patch-size 'patch-size)
    (unless (and (zerop (mod height patch-size)) (zerop (mod width patch-size)))
      (invalid-layout "Pixel dimensions must divide into patches"))
    (let ((rows (/ height patch-size)) (columns (/ width patch-size)))
      (tbk:reshape
       (tbk:permute (tbk:reshape pixels (list batch channels rows patch-size columns patch-size))
                    '(0 2 4 1 3 5))
       (list batch (* rows columns) (* channels patch-size patch-size))))))

(defmethod pixel-shuffle-features ((features tb:tensor) scale-factor)
  (destructuring-bind (batch patches channels) (tb:tensor-shape features)
    (require-positive-integer scale-factor 'scale-factor)
    (let ((side (isqrt patches)))
      (unless (and (= (* side side) patches) (zerop (mod side scale-factor)))
        (invalid-layout "Invalid square patch grid for pixel shuffle"))
      (let ((out-side (/ side scale-factor)))
        (tbk:reshape
         (tbk:permute
          (tbk:reshape
           (tbk:permute (tbk:reshape features (list batch side out-side (* channels scale-factor)))
                        '(0 2 1 3))
           (list batch out-side out-side (* channels scale-factor scale-factor)))
          '(0 2 1 3))
         (list batch (* out-side out-side) (* channels scale-factor scale-factor)))))))

(defun vision-mask-and-positions (mask batch side)
  "Accept nonempty top-left rectangular patch masks; return host indices/additive mask."
  (unless (and (arrayp mask) (equal (array-dimensions mask) (list batch side side)))
    (invalid-layout "Patch mask must have shape (~D ~D ~D)" batch side side))
  (let ((positions (make-array (list batch (* side side)) :element-type '(signed-byte 32) :initial-element 0))
        (attention (make-array (list batch 1 1 (* side side)) :element-type 'single-float)))
    (dotimes (b batch)
      (dotimes (y side)
        (dotimes (x side)
          (unless (member (aref mask b y x) '(0 1) :test #'eql)
            ;; Reference masks arrive as float32; permit exact 0.0/1.0, not arbitrary nonzero values.
            (unless (and (realp (aref mask b y x)) (or (= 0 (aref mask b y x)) (= 1 (aref mask b y x))))
              (invalid-layout "Patch mask is not binary")))))
      (let ((height (loop for y below side count (= 1 (aref mask b y 0))))
            (width (loop for x below side count (= 1 (aref mask b 0 x)))))
        (unless (and (plusp height) (plusp width)) (invalid-layout "Empty image mask"))
        (dotimes (y side)
          (dotimes (x side)
            (let* ((valid (and (< y height) (< x width))) (index (+ (* y side) x)))
              (unless (eq valid (= 1 (aref mask b y x))) (invalid-layout "Patch mask must be a top-left rectangle"))
              (setf (aref attention b 0 0 index) (if valid 0.0 most-negative-single-float))
              (when valid
                (flet ((bucket (coordinate extent)
                         (let ((fraction (* (float coordinate 1.0) (/ 1.0 extent))))
                           (loop for i from 1 below side count (>= fraction (/ (float i 1.0) side))))))
                  (setf (aref positions b index) (+ (* side (bucket y height)) (bucket x width))))))))))
    (values positions attention)))

(defun vision-linear (model input prefix)
  (tbk:add (tbk:linear input (vision-weight model (concatenate 'string prefix ".weight")))
           (vision-weight model (concatenate 'string prefix ".bias"))))

(defun vision-normalize (model input prefix)
  (tbk:layer-norm input (vision-weight model (concatenate 'string prefix ".weight"))
                 (vision-weight model (concatenate 'string prefix ".bias"))
                 (vision-spec-epsilon (model-vision-spec model))))

(defun vision-block-graph (model input mask index)
  (let* ((layout (vision-spec-layout (model-vision-spec model)))
         (heads (vision-layout-num-heads layout)) (width (vision-layout-hidden-size layout))
         (head-width (/ width heads)) (shape (tb:tensor-shape input))
         (prefix (format nil "model.vision_model.encoder.layers.~D." index))
         (normalized (vision-normalize model input (concatenate 'string prefix "layer_norm1"))))
    (flet ((projection (name)
             (tbk:permute
              (tbk:reshape (vision-linear model normalized (concatenate 'string prefix "self_attn." name))
                           (list (first shape) (second shape) heads head-width)) '(0 2 1 3))))
      (let* ((attended (tbk:attention (projection "q_proj") (projection "k_proj") (projection "v_proj")
                                      (/ 1.0 (sqrt head-width)) :mode :bidirectional :mask mask))
             (merged (tbk:reshape (tbk:permute attended '(0 2 1 3)) shape))
             (residual (tbk:add input (vision-linear model merged (concatenate 'string prefix "self_attn.out_proj"))))
             (ffn-input (vision-normalize model residual (concatenate 'string prefix "layer_norm2")))
             (expanded (tbk:gelu (vision-linear model ffn-input (concatenate 'string prefix "mlp.fc1")) :approximate t)))
        (tbk:add residual (vision-linear model expanded (concatenate 'string prefix "mlp.fc2")))))))

(defun vision-block (model input mask index)
  "Build a block with bounded temporary handles; return a caller-scoped result.
The lazy MLX graph keeps its dependencies alive, not the Lisp temporary handles."
  (tb:with-resource
      (result (tbk:with-backend ((vision-backend model))
                (tbk:retain (vision-block-graph model input mask index))))
    ;; A same-shape native alias joins the enclosing encode-vision scope. The
    ;; independently retained bridge handle is disposed even if aliasing fails.
    (tbk:reshape result (tb:tensor-shape result))))

(defun encode-vision (model pixels &key patch-mask (output :connector))
  "Encode preprocessed FP32 NCHW tensors; return an owned native output tensor.
Fixed square tiles matching config only. PATCH-MASK is a host (batch rows columns)
binary top-left rectangle. OUTPUT selects :PATCH-PROJECTION, :EMBEDDINGS,
:FIRST-LAYER, :VISION, :SHUFFLE or :CONNECTOR. No image decoding/normalization occurs."
  (when (vision-disposed-p model) (invalid-layout "Vision model has been disposed"))
  (unless (member output '(:patch-projection :embeddings :first-layer :vision :shuffle :connector))
    (invalid-layout "Unknown vision output ~S" output))
  (let* ((spec (model-vision-spec model)) (layout (vision-spec-layout spec))
         (size (vision-layout-image-size layout)) (patch (vision-layout-patch-size layout))
         (side (/ size patch)) (shape (tb:tensor-shape pixels))
         (batch (first shape)) (backend (vision-backend model)))
    (unless (and (equal (rest shape) (list (vision-layout-channels layout) size size))
                 (typep batch '(integer 1)) (eq (tb:tensor-dtype pixels) :float32))
      (invalid-layout "Expected FP32 (batch ~D ~D ~D) pixels" (vision-layout-channels layout) size size))
    (multiple-value-bind (positions mask)
        (vision-mask-and-positions (or patch-mask (make-array (list batch side side) :initial-element 1)) batch side)
      (tbk:with-backend (backend)
        (tbk:evaluate pixels)
        (flet ((done (tensor) (return-from encode-vision (tbk:retain (tbk:evaluate tensor)))))
          (let* ((patches (patchify-image pixels patch))
                 (kernel (tbk:reshape (vision-weight model "model.vision_model.embeddings.patch_embedding.weight")
                                      (list (vision-layout-hidden-size layout) (* (vision-layout-channels layout) patch patch))))
                 (hidden (tbk:add (tbk:linear patches kernel)
                                  (vision-weight model "model.vision_model.embeddings.patch_embedding.bias"))))
            (when (eq output :patch-projection) (done hidden))
            (setf hidden (tbk:add hidden (tbk:take-indices
                                         (vision-weight model "model.vision_model.embeddings.position_embedding.weight")
                                         (tb:tensor-from-array backend positions :dtype :int32))))
            (when (eq output :embeddings) (done hidden))
            (let ((mask (tb:tensor-from-array backend mask)))
              (dotimes (index (vision-spec-layers spec))
                (setf hidden (vision-block model hidden mask index))
                (when (and (zerop index) (eq output :first-layer)) (done hidden))))
            (setf hidden (vision-normalize model hidden "model.vision_model.post_layernorm"))
            (when (eq output :vision) (done hidden))
            (setf hidden (pixel-shuffle-features hidden (vision-layout-scale-factor layout)))
            (when (eq output :shuffle) (done hidden))
            (done (tbk:linear hidden (vision-weight model "model.connector.modality_projection.proj.weight")))))))))
