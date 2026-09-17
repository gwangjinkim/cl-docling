(asdf:defsystem "cl-docling"
  :description "Common Lisp foundations for native Docling vision-language models"
  :version "0.25.0"
  :author "Gwang-Jin Kim"
  :license "MIT"
  :serial t
  :components ((:file "src/package")
               (:file "src/conditions")
               (:file "src/layout")
               (:file "src/geometry")
               (:file "src/document")
               (:file "src/doctags")
               (:file "src/tables")
               (:file "src/supervision"))
  :in-order-to ((asdf:test-op (asdf:test-op "cl-docling/tests"))))

(asdf:defsystem "cl-docling/mlx"
  :description "Native Idefics3 vision, Llama decoder, generation and interchange"
  :depends-on ("cl-docling" (:version "cl-transformer-blocks" "0.41.1")
               (:version "cl-transformer-blocks/kernels" "0.4.0") "yason" "babel")
  :serial t
  :components ((:file "src/vision-config") (:file "src/vision")
               (:file "src/model") (:file "src/generation") (:file "src/document-training")
               (:file "src/document-adapters") (:file "src/document-checkpoints")))

(asdf:defsystem "cl-docling/images"
  :description "Native PNG and explicit PIL-compatible Idefics3 preprocessing"
  :depends-on ("cl-docling" "cffi")
  :serial t
  :components ((:file "src/images")))

(asdf:defsystem "cl-docling/pipeline"
  :description "Qualified single-PNG SmolDocling input, answer supervision and native generation"
  :depends-on ("cl-docling/images" "cl-docling/mlx")
  :serial t
  :components ((:file "src/processor") (:file "src/training-inputs")))

(asdf:defsystem "cl-docling/tests"
  :depends-on ("cl-docling")
  :serial t
  :components ((:file "tests/suite") (:file "tests/documents") (:file "tests/multipage-suite") (:file "tests/tables-suite")
               (:file "tests/supervision-suite"))
  :perform (asdf:test-op (operation system)
             (declare (ignore operation system))
             (uiop:symbol-call :cl-docling-tests :run-tests)))

(asdf:defsystem "cl-docling/pdf"
  :description "Bounded Poppler PDF rasterization on SBCL/POSIX; GNU timeout required"
  :depends-on ("cl-docling" "sb-posix")
  :components ((:file "src/pdf"))
  :in-order-to ((asdf:test-op (asdf:test-op "cl-docling/pdf-tests"))))

(asdf:defsystem "cl-docling/pdf-tests"
  :depends-on ("cl-docling/pdf")
  :components ((:file "tests/pdf-suite"))
  :perform (asdf:test-op (operation system)
             (declare (ignore operation system))
             (uiop:symbol-call :cl-docling-pdf-tests :run-tests)))

(asdf:defsystem "cl-docling/bundles"
  :description "Explicit local document artifact bundles on SBCL/POSIX"
  :depends-on ("cl-docling" "sb-posix")
  :components ((:file "src/bundles"))
  :in-order-to ((asdf:test-op (asdf:test-op "cl-docling/bundle-tests"))))

(asdf:defsystem "cl-docling/bundle-tests"
  :depends-on ("cl-docling/bundles")
  :components ((:file "tests/bundle-suite"))
  :perform (asdf:test-op (operation system)
             (declare (ignore operation system))
             (uiop:symbol-call :cl-docling-bundle-tests :run-tests)))
