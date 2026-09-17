(in-package #:cl-docling)

;; An internal bounded token tree. Source spans refer to characters, not UTF-8 bytes.
(defstruct (doctag-token (:constructor make-doctag-token (tag start end)))
  tag start end children (closed-p nil) (malformed-p nil))

(defun doctag-space-p (character)
  (find character '(#\Space #\Tab #\Newline #\Return #\Page)))

(defun doctag-blank-p (text)
  (every #'doctag-space-p text))

(defun doctag-prefix-p (prefix text)
  (and (<= (length prefix) (length text))
       (string= prefix text :end2 (length prefix))))

(defun doctag-heading-level (tag)
  (let ((prefix "section_header_level_"))
    (when (and (doctag-prefix-p prefix tag) (= (length tag) (1+ (length prefix))))
      (let ((digit (digit-char-p (char tag (length prefix)))))
        (and digit (<= 1 digit 5) digit)))))

(defun otsl-marker-p (tag)
  (member tag '("fcel" "ecel" "ched" "rhed" "srow" "lcel" "ucel" "xcel" "nl") :test #'equal))

(defun scan-doctags (raw diagnose max-nodes max-depth)
  "One forward scan; unknown tags remain in a bounded tree rather than disappearing."
  (let* ((size (length raw)) (root (make-doctag-token "#root" 0 size))
         (stack (list root)) (count 0) (position 0))
    (labels ((problem (code start message)
               ;; Preserve scanner failures on containing tables for marked-partial export.
               (dolist (node stack) (setf (doctag-token-malformed-p node) t))
               (funcall diagnose code start message))
             (add (tag start end &optional atomic)
               (when (> (incf count) max-nodes) (invalid-document "DocTags node budget exceeded."))
               (let ((node (make-doctag-token tag start end)))
                 (push node (doctag-token-children (first stack)))
                 (if atomic (setf (doctag-token-closed-p node) t)
                     (progn
                       (when (> (length stack) max-depth) (invalid-document "DocTags depth budget exceeded."))
                       (push node stack)))
                 node))
             (finish (end closed-p)
               (let ((node (pop stack)))
                 (setf (doctag-token-end node) end
                       (doctag-token-closed-p node) closed-p
                       (doctag-token-children node) (nreverse (doctag-token-children node))))))
      (loop while (< position size) do
        (if (char/= #\< (char raw position))
            (let ((end (or (position #\< raw :start position) size)))
              (add nil position end t)
              (setf position end))
            (let ((end (position #\> raw :start (1+ position))))
              (unless end
                (problem :malformed-token position "Unterminated token.")
                (add nil position size t)
                (setf position size)
                (return))
              (let* ((tag (subseq raw (1+ position) end))
                     (closing (and (plusp (length tag)) (char= #\/ (char tag 0)))))
                (cond
                  (closing
                   (let* ((name (subseq tag 1))
                          (match (find name (butlast stack) :key #'doctag-token-tag :test #'string=)))
                     (unless (and match (eq match (first stack)))
                       (problem :mismatched-close position "Unexpected or mismatched closing tag."))
                     (when match
                       (loop until (eq match (first stack)) do
                         (problem :unclosed-tag (doctag-token-start (first stack)) "Missing closing tag.")
                         (finish position nil))
                       (finish (1+ end) t))))
                  (t
                   (when (or (zerop (length tag))
                             (some (lambda (c) (or (doctag-space-p c) (find c "<>/"))) tag))
                     (problem :malformed-token position "Unsupported token syntax; attributes and XML are not supported."))
                   (add tag position (1+ end)
                        (or (doctag-prefix-p "loc_" tag) (string= "end_of_utterance" tag)
                            (otsl-marker-p tag)))))
              (setf position (1+ end))))))
      (loop while (rest stack) do
        (problem :unclosed-tag (doctag-token-start (first stack)) "Missing closing tag.")
        (finish size nil))
      (nreverse (doctag-token-children root)))))

(defun doctag-location-value (tag)
  (let ((digits (subseq tag 4)))
    (when (and (<= 1 (length digits) 3) (every #'digit-char-p digits))
      (let ((number (parse-integer digits)))
        (and (< number 500) number)))))

(defun doctag-entity-p (text)
  ;; DocTags is not XML: do not guess whether entity-looking text needs decoding.
  (loop for i below (1- (length text))
        thereis (and (char= #\& (char text i))
                     (or (alphanumericp (char text (1+ i))) (char= #\# (char text (1+ i)))))))

(defun doctag-leaf-content (node raw diagnose)
  (let ((locations nil) (content-seen nil) (bad-location nil))
    (let ((text
            (with-output-to-string (out)
              (dolist (child (doctag-token-children node))
                (let ((tag (doctag-token-tag child)))
                  (cond
                    ((null tag)
                     (let ((part (subseq raw (doctag-token-start child) (doctag-token-end child))))
                       (unless (doctag-blank-p part) (setf content-seen t))
                       (write-string part out)))
                    ((doctag-prefix-p "loc_" tag)
                     (when content-seen (setf bad-location t))
                     (let ((value (doctag-location-value tag)))
                       (unless value (setf bad-location t))
                       (push value locations)))
                    (t (funcall diagnose :unsupported-structure (doctag-token-start child)
                                "Nested markup in a text element is unsupported; retain raw DocTags."))))))))
      (setf locations (nreverse locations))
      (when (or bad-location
                (and locations
                     (or (/= 4 (length locations))
                         (not (every #'integerp locations))
                         (> (first locations) (third locations))
                         (> (second locations) (fourth locations)))))
        (funcall diagnose :location (doctag-token-start node)
                 "Expected four leading location tokens in [0,499], with left<=right and top<=bottom."))
      (when (doctag-entity-p text)
        (funcall diagnose :ambiguous-entity (doctag-token-start node)
                 "Entity-like text is preserved literally, not XML-decoded."))
      (when (some (lambda (c) (and (< (char-code c) 32) (not (doctag-space-p c)))) text)
        (funcall diagnose :control-character (doctag-token-start node) "Control character in model text."))
      (values text locations))))

(defun build-document-element (node raw diagnose)
  (when (equal "otsl" (doctag-token-tag node))
    (return-from build-document-element (parse-otsl-table node raw diagnose)))
  (let* ((tag (doctag-token-tag node))
         (level (and tag (doctag-heading-level tag)))
         (kind (cond (level :heading)
                     ((equal tag "text") :text) ((equal tag "title") :title)
                     ((equal tag "page_footer") :page-footer)
                     ((equal tag "list_item") :list-item)
                     ((equal tag "ordered_list") :ordered-list)
                     ((equal tag "unordered_list") :unordered-list)
                     (t :unknown))))
    (multiple-value-bind (text location children)
        (cond
          ((eq kind :unknown)
           (funcall diagnose :unsupported-tag (doctag-token-start node) "Unsupported content retained in raw DocTags.")
           (values "" nil nil))
          ((member kind '(:ordered-list :unordered-list))
           (values "" nil
                   (loop for child in (doctag-token-children node)
                         unless (and (null (doctag-token-tag child))
                                     (doctag-blank-p (subseq raw (doctag-token-start child) (doctag-token-end child))))
                           collect (progn
                                     (unless (equal "list_item" (doctag-token-tag child))
                                       (funcall diagnose :unsupported-structure (doctag-token-start child)
                                                "Lists support direct list_item children only."))
                                     (build-document-element child raw diagnose)))))
          (t (doctag-leaf-content node raw diagnose)))
      (%make-document-element :kind kind :tag (or tag "#text") :level level
                              :text text :location location :children children
                              :start (doctag-token-start node) :end (doctag-token-end node)))))

(defun parse-doctags (raw &key (page-number 1) (stop-reason :unknown) token-ids
                             (max-characters 2000000) (max-nodes 100000) (max-depth 64))
  "Parse the supported single-page DocTags subset without discarding raw output.
Malformed/unsupported content becomes diagnostics; invalid arguments/resource limits
signal DOCUMENT-ERROR. Offsets are zero-based, end-exclusive character positions."
  (dolist (number (list page-number max-characters max-nodes max-depth))
    (unless (and (integerp number) (plusp number)) (invalid-document "Expected positive integer bounds/page number.")))
  (unless (<= max-depth 128) (invalid-document "Maximum supported parser depth is 128."))
  (unless (and (stringp raw) (<= (length raw) max-characters))
    (invalid-document "Expected a string within the DocTags character budget."))
  (unless (member stop-reason '(:unknown :eos :length)) (invalid-document "Unknown generation stop reason."))
  (unless (or (null token-ids)
              (and (vectorp token-ids) (not (stringp token-ids))
                   (every (lambda (id) (and (integerp id) (>= id 0))) token-ids)))
    (invalid-document "Token IDs must be a vector of nonnegative integers."))
  (let ((diagnostics nil) (raw (copy-seq raw)))
    (flet ((diagnose (code start message)
             (push (%make-document-diagnostic :code code :start start :message message) diagnostics)))
      (when (eq stop-reason :length)
        (diagnose :generation-truncated (length raw) "Generation stopped at its token budget, not EOS."))
      (let* ((nodes (scan-doctags raw #'diagnose max-nodes max-depth))
             (wrappers (remove-if-not (lambda (node) (equal "doctag" (doctag-token-tag node))) nodes))
             (wrapper (first wrappers)) (end-seen nil) (elements nil))
        (unless (= 1 (length wrappers))
          (diagnose :document-wrapper 0 "Expected exactly one outer doctag wrapper."))
        (dolist (node nodes)
          (cond
            ((equal "doctag" (doctag-token-tag node))
             (dolist (child (doctag-token-children node))
               (unless (and (null (doctag-token-tag child))
                            (doctag-blank-p (subseq raw (doctag-token-start child) (doctag-token-end child))))
                 (push (build-document-element child raw #'diagnose) elements))))
            ((equal "end_of_utterance" (doctag-token-tag node))
             (when (or end-seen (null wrapper) (not (doctag-token-closed-p wrapper))
                       (< (doctag-token-start node) (doctag-token-end wrapper)))
               (diagnose :end-marker (doctag-token-start node) "Misplaced or duplicate end-of-utterance marker."))
             (setf end-seen t))
            ((and (null (doctag-token-tag node))
                  (doctag-blank-p (subseq raw (doctag-token-start node) (doctag-token-end node)))))
            (t (diagnose :outside-content (doctag-token-start node) "Content outside the document wrapper."))))
        (%make-parsed-document :raw raw :elements (nreverse elements)
                               :diagnostics (nreverse diagnostics) :page-number page-number
                               :stop-reason stop-reason :token-ids (and token-ids (copy-seq token-ids)))))))
