;;; Pinned assistant chat policy from Transformers (Apache-2.0); see THIRD-PARTY.md.
(in-package #:docling)
(export '(prepare-image-training-input prepare-image-training-example))

(defun prepare-image-training-input (model file answer &key (task "Convert this page to docling.")
                                                       pad-to (pad-token-id (document-eos-token-id model)))
  "One PNG and assistant answer -> owned IDs, unshifted labels, attention mask,
pixels, tile count and answer-start. Host arrays only; no gradients or updates.
ANSWER excludes chat/EOS markers. Final template newline after EOS is not trained."
  (require-document model)
  (validate-image-task answer)
  (when (doctag-blank-p answer) (invalid-layout "An assistant answer must contain non-whitespace text."))
  (multiple-value-bind (prompt-ids pixels tiles prompt) (prepare-image-input model file :task task)
    (let* ((eos (document-eos-token-id model))
           (encoded-eos (tb:encode-text model "<end_of_utterance>" :add-special-tokens nil))
           ;; Tokenize the complete teacher-forced transcript, never concatenate
           ;; separately encoded answer tokens across a possible BPE boundary.
           (tokens (tb:encode-text model (concatenate 'string prompt " " answer "<end_of_utterance>")
                                   :add-special-tokens t))
           (start (array-dimension prompt-ids 1)))
      (unless (and (= 1 (length encoded-eos)) (eql eos (aref encoded-eos 0)))
        (invalid-layout "Saved EOS disagrees with the pinned assistant terminator."))
      (unless (and (< start (length tokens))
                   (loop for i below start always (= (aref prompt-ids 0 i) (aref tokens i))))
        (invalid-layout "Full tokenization does not preserve the prompt prefix; answer boundary is ambiguous."))
      (multiple-value-bind (ids labels attention)
          (make-answer-supervision tokens start
                                   :vocab-size (gethash "vocab_size" (tb:model-config (document-decoder model)))
                                   :image-token-id (gethash "image_token_id" (document-config model))
                                   :eos-token-id eos :pad-token-id pad-token-id :pad-to pad-to
                                   :max-length (min 8192 (gethash "max_position_embeddings"
                                                                  (tb:model-config (document-decoder model)))))
        (values ids labels attention pixels tiles start)))))

(defun prepare-image-training-example (model file answer &key (task "Convert this page to docling.")
                                                         pad-to (pad-token-id (document-eos-token-id model)))
  "One PNG/answer -> owned IDs, labels, attention, native FEATURES, tile count, answer-start.
Encode every visual tile once. Reuse FEATURES while vision/connector remain unchanged;
dispose them before MODEL/backend. No adapter attachment or optimizer update."
  (multiple-value-bind (ids labels attention pixels tiles start)
      (prepare-image-training-input model file answer :task task :pad-to pad-to :pad-token-id pad-token-id)
    (values ids labels attention (document-tile-features model pixels) tiles start)))
