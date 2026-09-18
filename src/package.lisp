(defpackage #:cl-docling
  (:use #:cl)
  (:nicknames #:docling)
  (:export #:layout-error #:layout-error-message
           #:vision-layout #:make-vision-layout
           #:vision-layout-image-size #:vision-layout-patch-size
           #:vision-layout-channels #:vision-layout-hidden-size
           #:vision-layout-num-heads #:vision-layout-scale-factor
           #:vision-layout-text-hidden-size
           #:patch-count #:image-token-count #:connector-input-width
           #:patchify-image #:pixel-shuffle-features
           #:document-error #:document-error-message
           #:parse-doctags #:document-to-markdown #:documents-to-markdown
           #:parsed-document #:parsed-document-raw #:parsed-document-elements
           #:parsed-document-diagnostics #:parsed-document-page-number
           #:parsed-document-stop-reason #:parsed-document-token-ids
           #:document-element #:document-element-kind #:document-element-tag
           #:document-element-text #:document-element-level #:document-element-location
           #:document-element-classification
           #:document-element-children #:document-element-start #:document-element-end
           #:document-diagnostic #:document-diagnostic-code
           #:document-diagnostic-message #:document-diagnostic-start
           #:document-element-table #:document-table #:document-table-rows
           #:document-table-column-count #:document-table-renderable-p
           #:table-cell #:table-cell-kind #:table-cell-text #:table-cell-start #:table-cell-end
           #:table-cell-tag #:table-cell-row-index #:table-cell-column-index
           #:table-cell-row-span #:table-cell-column-span #:document-table-cells
           #:make-answer-supervision))
