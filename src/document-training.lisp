(in-package #:docling)
(export '(make-document-lora document-lora-parameters document-loss-and-gradients document-train-step))

(defun make-document-lora (model &key (rank 8) (alpha 16) (seed 17))
  "Attach zero-dropout query/value decoder LoRA. Vision, connector and base stay frozen."
  (require-document model)
  (tb:make-lora (document-decoder model) :rank rank :alpha alpha :seed seed
                                      :targets '("q_proj" "v_proj"))
  (setf (document-lora-p model) t)
  model)

(defun require-document-lora (model)
  (require-document model)
  (unless (document-lora-p model) (invalid-layout "Attach a document LoRA adapter before training")))

(defun document-adapter-names (parameters)
  ;; Engine decoder uses canonical Llama names; document PEFT names nest text_model.
  (loop for (name . tensor) in parameters
        collect (cons (concatenate 'string "base_model.model.model.text_model."
                                   (subseq name (length "base_model.model.model."))) tensor)))

(defun document-lora-parameters (model)
  "Return document-namespaced borrowed adapter tensors. Do not dispose/mutate them."
  (require-document-lora model)
  (document-adapter-names (tb:trainable-parameters (document-decoder model))))

(defun call-with-document-training-input (model ids features labels attention-mask tile-count function)
  (require-document-lora model)
  (unless (and (arrayp ids) (= (array-rank ids) 2) (arrayp labels)
               (equal (array-dimensions ids) (array-dimensions labels))
               (or (null attention-mask)
                   (and (arrayp attention-mask) (equal (array-dimensions ids) (array-dimensions attention-mask)))))
    (invalid-layout "Training IDs, explicit labels and optional attention must have matching rank-two shapes"))
  (let ((image-id (gethash "image_token_id" (document-config model))))
    (dotimes (i (array-total-size ids))
      (when (or (eql image-id (row-major-aref labels i))
                (and (eql image-id (row-major-aref ids i))
                     (or (not (eql -100 (row-major-aref labels i)))
                         (and attention-mask (not (eql 1 (row-major-aref attention-mask i)))))))
        (invalid-layout "Image slots must be attended and unsupervised; image IDs cannot be targets"))))
  (tb:with-resource (merged (merge-image-embeddings model ids features :tile-count tile-count))
    (funcall function (document-decoder model) merged)))

(defun document-loss-and-gradients (model ids features &key labels attention-mask (tile-count 1))
  "Mean answer loss and owned document-namespaced LoRA gradients from frozen FEATURES.
FEATURES are borrowed native image features; labels are unshifted with -100 ignores."
  (call-with-document-training-input model ids features labels attention-mask tile-count
    (lambda (decoder merged)
      (multiple-value-bind (loss gradients)
          (tbk:loss-and-gradients-embeddings decoder merged :labels labels :attention-mask attention-mask)
        (values loss (document-adapter-names gradients))))))

(defun document-train-step (model optimizer ids features &key labels attention-mask (tile-count 1) max-grad-norm)
  "Update only decoder q/v LoRA; return pre-update loss. Borrow fixed image FEATURES.
Caller owns optimizer/features. Existing decoder caches become stale after updates."
  (call-with-document-training-input model ids features labels attention-mask tile-count
    (lambda (decoder merged)
      (tbk:train-step-embeddings decoder optimizer merged :labels labels :attention-mask attention-mask
                                :max-grad-norm max-grad-norm))))
