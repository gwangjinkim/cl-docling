(in-package #:cl-docling)

(defun require-array-shape (array rank)
  (unless (= rank (array-rank array))
    (invalid-layout "Expected rank ~D, got shape ~S" rank (array-dimensions array)))
  (dolist (dimension (array-dimensions array))
    (require-positive-integer dimension 'array-dimension)))

(defgeneric patchify-image (pixels patch-size)
  (:documentation
   "Extract non-overlapping patches from NCHW pixels into (batch patches C*P*P).
The array method is a host reference, not an MLX kernel. It preserves element type,
allocates independent output, and neither normalizes pixels nor applies weights."))

(defmethod patchify-image ((pixels array) patch-size)
  (require-array-shape pixels 4)
  (require-positive-integer patch-size 'patch-size)
  (destructuring-bind (batch channels height width) (array-dimensions pixels)
    (unless (and (zerop (mod height patch-size)) (zerop (mod width patch-size)))
      (invalid-layout "Image dimensions ~Dx~D must be divisible by patch size ~D"
                      height width patch-size))
    (let* ((rows (/ height patch-size))
           (columns (/ width patch-size))
           (result (make-array (list batch (* rows columns) (* channels patch-size patch-size))
                               :element-type (array-element-type pixels))))
      (dotimes (b batch)
        (dotimes (y rows)
          (dotimes (x columns)
            (dotimes (c channels)
              (dotimes (dy patch-size)
                (dotimes (dx patch-size)
                  (setf (aref result b (+ (* y columns) x)
                              (+ (* c patch-size patch-size) (* dy patch-size) dx))
                        (aref pixels b c (+ (* y patch-size) dy) (+ (* x patch-size) dx)))))))))
      result)))

(defgeneric pixel-shuffle-features (features scale-factor)
  (:documentation
   "Compress square-grid (batch patches channels) features spatially.
Each local R*R group becomes channels ordered (local-row local-column channel).
The array method is an allocating host reference for Idefics3, without projection."))

(defmethod pixel-shuffle-features ((features array) scale-factor)
  (require-array-shape features 3)
  (require-positive-integer scale-factor 'scale-factor)
  (destructuring-bind (batch patches channels) (array-dimensions features)
    (let ((side (isqrt patches)))
      (unless (and (= patches (* side side)) (zerop (mod side scale-factor)))
        (invalid-layout "~D patches must form a square with side divisible by ~D"
                        patches scale-factor))
      (let* ((out-side (/ side scale-factor))
             (out-width (* channels scale-factor scale-factor))
             (result (make-array (list batch (* out-side out-side) out-width)
                                 :element-type (array-element-type features))))
        (dotimes (b batch)
          (dotimes (y out-side)
            (dotimes (x out-side)
              (dotimes (dy scale-factor)
                (dotimes (dx scale-factor)
                  (dotimes (c channels)
                    (setf (aref result b (+ (* y out-side) x)
                                (+ (* (+ (* dy scale-factor) dx) channels) c))
                          (aref features b
                                (+ (* (+ (* y scale-factor) dy) side) (* x scale-factor) dx)
                                c))))))))
        result))))
