(in-package #:docling)
(export 'generate-document)

(defun generate-document (model prompt pixels &key patch-mask (max-new-tokens 128)
                                                 image-features (tile-count 1)
                                                 (eos-token-id (document-eos-token-id model)))
  "Greedy single-row generation. Return NEW token vector and :EOS or :LENGTH.
PROMPT is pre-expanded (1 time) IDs. PIXELS is one preprocessed tile, or NIL with
explicit IMAGE-FEATURES and TILE-COUNT for a multi-tile single-row request.
Use TB:DECODE-TOKENS for raw output; no DocTags parsing or image preprocessing.
EOS NIL disables stopping; the published root pad_token_id is never used."
  (require-document model)
  (unless (and (arrayp prompt) (= 2 (array-rank prompt)) (= 1 (array-dimension prompt 0))
               (plusp (array-dimension prompt 1)) (typep max-new-tokens '(integer 0))
               (or (null eos-token-id)
                   (typep eos-token-id `(integer 0 (,(gethash "vocab_size" (tb:model-config (document-decoder model))))))))
    (invalid-layout "Invalid single-sequence prompt, token budget, or EOS"))
  (when (> (+ (array-dimension prompt 1) max-new-tokens)
           (gethash "max_position_embeddings" (tb:model-config (document-decoder model))))
    (invalid-layout "Prompt plus generation budget exceeds context"))
  (let ((tokens (make-array 0 :element-type '(unsigned-byte 32) :adjustable t :fill-pointer 0)))
    (when (zerop max-new-tokens) (return-from generate-document (values tokens :length)))
    (tb:with-resource (cache (tb:make-cache model))
      (let ((next prompt))
        (dotimes (step max-new-tokens)
          (tb:with-resource (logits (forward-document model next :cache cache
                                      :pixels (when (zerop step) pixels)
                                      :image-features (when (zerop step) image-features)
                                      :tile-count (if (zerop step) tile-count 1)
                                      :patch-mask (when (zerop step) patch-mask)))
            (let ((token (tbk:with-backend ((document-backend model)) (tbk:greedy-token logits))))
              (vector-push-extend token tokens)
              (when (eql token eos-token-id) (return-from generate-document (values tokens :eos)))
              (setf next (make-array '(1 1) :initial-element token)))))))
    (values tokens :length)))
