;;; Controlled lifetime probe. Restores the original function even on failure.
(load (merge-pathnames "experiment-common.lisp" *load-truename*))

(let* ((output (first (uiop:command-line-arguments)))
       (original (symbol-function 'docling::vision-block))
       ;; Preserve the historical unscoped baseline after production adopts scopes.
       (graph (if (fboundp 'docling::vision-block-graph)
                  (symbol-function 'docling::vision-block-graph) original))
       (original-transfer (symbol-function 'tb:tensor-from-array))
       (transfer-seconds 0d0)
       (results (make-hash-table :test #'equal))
       (reference nil))
  (when (probe-file output) (error "Output must be new."))
  (tb:with-resource (backend (tb:make-backend :device :gpu))
    (tb:with-resource (model (docling:load-document-model (uiop:getenv "DOCLING_MODEL") :backend backend))
      (multiple-value-bind (ids pixels tiles)
          (docling:prepare-image-input model (asdf:system-relative-pathname "cl-docling" "tests/fixtures/table-pages/grid.png"))
        (declare (ignore ids tiles))
        (unwind-protect
             (dolist (mode '("original" "block-scoped"))
               (setf (symbol-function 'tb:tensor-from-array)
                     (lambda (backend data &rest args)
                       (let ((start (get-internal-real-time)))
                         (prog1 (apply original-transfer backend data args)
                           (incf transfer-seconds (/ (- (get-internal-real-time) start)
                                                    (float internal-time-units-per-second 1d0)))))))
               (setf (symbol-function 'docling::vision-block)
                     (if (equal mode "original") graph
                         (lambda (vision input mask index)
                           (tb:with-resource
                               (result (tbk:with-backend ((docling::vision-backend vision))
                                         (tbk:retain (funcall graph vision input mask index))))
                             ;; Register a shared-data alias in the caller's scope.
                             (tbk:reshape result (tb:tensor-shape result))))))
               (let ((samples nil) (maximum 0.0))
                 (dotimes (repeat 4)
                   (tb:reset-backend-peak-memory backend)
                   (setf transfer-seconds 0d0)
                   (let ((start (get-internal-real-time)))
                     (tb:with-resource (value (docling:document-tile-features model pixels))
                       (tbk:with-backend (backend) (tbk:evaluate value))
                       (let* ((elapsed (/ (- (get-internal-real-time) start)
                                          (float internal-time-units-per-second 1d0)))
                              (actual (tb:tensor-array value)) (memory (tb:backend-memory backend)))
                         (if reference
                             (dotimes (i (array-total-size actual))
                               (let* ((x (row-major-aref actual i)) (y (row-major-aref reference i))
                                      (delta (abs (- x y))))
                                 (setf maximum (max maximum delta))
                                 (unless (<= delta (+ 1e-3 (* 3e-4 (abs y)))) (error "Feature mismatch"))))
                             (setf reference actual))
                         (push (experiment-object "seconds" elapsed "host_to_native_seconds" transfer-seconds
                                                  "mlx_peak_bytes" (getf memory :peak)) samples)
                         (format t "~&~A repeat ~D: ~,4F s transfer ~,4F s peak ~D~%" mode repeat elapsed transfer-seconds (getf memory :peak))
                         (finish-output)))))
                 (setf (gethash mode results)
                       (experiment-object "warmup" (car (last samples)) "samples" (coerce (reverse (butlast samples)) 'vector)
                                          "maximum_absolute_difference" maximum
                                          "post_request_handles" (getf (tb:backend-memory backend) :handles)))))
          (setf (symbol-function 'docling::vision-block) original
                (symbol-function 'tb:tensor-from-array) original-transfer)))))
  (experiment-write output (experiment-object "device" "gpu" "dtype" "float32" "tiles" 13
                                              "scope" "Vision lifetime probe, not page inference" "results" results) t))
