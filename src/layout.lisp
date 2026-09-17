(in-package #:cl-docling)

(defstruct (vision-layout (:constructor %make-vision-layout))
  "Validated square-tile geometry, not a complete Hugging Face model config."
  (image-size 512 :read-only t)
  (patch-size 16 :read-only t)
  (channels 3 :read-only t)
  (hidden-size 768 :read-only t)
  (num-heads 12 :read-only t)
  (scale-factor 4 :read-only t)
  (text-hidden-size 576 :read-only t))

(defun make-vision-layout (&key (image-size 512) (patch-size 16) (channels 3)
                              (hidden-size 768) (num-heads 12) (scale-factor 4)
                              (text-hidden-size 576))
  "Validate tile geometry; defaults describe the SmolDocling preview layout."
  (loop for value in (list image-size patch-size channels hidden-size num-heads
                          scale-factor text-hidden-size)
        for name in '(image-size patch-size channels hidden-size num-heads
                      scale-factor text-hidden-size)
        do (require-positive-integer value name))
  (unless (zerop (mod image-size patch-size))
    (invalid-layout "Image size ~D is not divisible by patch size ~D" image-size patch-size))
  (unless (zerop (mod (/ image-size patch-size) scale-factor))
    (invalid-layout "Patch grid side must be divisible by shuffle factor ~D" scale-factor))
  (unless (zerop (mod hidden-size num-heads))
    (invalid-layout "Vision hidden size ~D is not divisible by ~D heads" hidden-size num-heads))
  (%make-vision-layout :image-size image-size :patch-size patch-size :channels channels
                      :hidden-size hidden-size :num-heads num-heads :scale-factor scale-factor
                      :text-hidden-size text-hidden-size))

(defun patch-count (layout)
  (expt (/ (vision-layout-image-size layout) (vision-layout-patch-size layout)) 2))

(defun image-token-count (layout)
  "Visual tokens per processed tile, excluding prompt/boundary tokens."
  (/ (patch-count layout) (expt (vision-layout-scale-factor layout) 2)))

(defun connector-input-width (layout)
  (* (vision-layout-hidden-size layout) (expt (vision-layout-scale-factor layout) 2)))
