SBCL ?= sbcl
PYTHON ?= python3
.DEFAULT_GOAL := test
.PHONY: test-book-picture
test-book-picture:
	$(PYTHON) scripts/test_book_picture.py

.PHONY: test-book-picture-export
test-book-picture-export:
	$(PYTHON) scripts/test_picture_score.py
	$(PYTHON) scripts/test_book_picture_export.py

.PHONY: test-book-final-guard
test-book-final-guard:
	$(PYTHON) scripts/test_book_final_guard.py

.PHONY: test-book-finetune
test-book-finetune:
	$(PYTHON) scripts/test_book_finetune.py

.PHONY: test-book-final-reference
test-book-final-reference:
	$(PYTHON) scripts/test_book_final_reference.py

.PHONY: test-book-final-evaluation
test-book-final-evaluation:
	$(PYTHON) scripts/test_book_final_evaluation.py

.PHONY: replay-book-finetune
replay-book-finetune:
	$(PYTHON) scripts/replay_book_finetune.py --inputs "$(FINETUNE_INPUTS)" \
	  --checkpoint "$(FINETUNE_CHECKPOINT)" --run "$(FINETUNE_RUN)" \
	  --output "$(FINETUNE_REPLAY)"

.PHONY: test-book-budget
test-book-budget:
	$(PYTHON) scripts/test_book_budget.py

.PHONY: test-book-score
test-book-score:
	$(PYTHON) scripts/test_book_score.py

.PHONY: test-book-splits
test-book-splits:
	$(PYTHON) scripts/test_book_splits.py

.PHONY: test-book-gradients
test-book-gradients:
	$(PYTHON) scripts/test_book_gradients.py

.PHONY: test-book-context
test-book-context:
	$(PYTHON) scripts/test_book_context.py

.PHONY: test-book-pilot
test-book-pilot:
	$(PYTHON) scripts/test_book_pilot.py

.PHONY: test-corpus-preflight
test-corpus-preflight:
	$(PYTHON) scripts/test_corpus_preflight.py

.PHONY: test-parser-replay
test-parser-replay:
	$(PYTHON) scripts/test_parser_replay.py

.PHONY: test-dpbench
test-dpbench:
	$(PYTHON) scripts/test_dpbench.py

.PHONY: test-dataset-geometry
test-dataset-geometry:
	$(PYTHON) scripts/test_dataset_geometry.py

.PHONY: test-dataset-audit
test-dataset-audit:
	$(PYTHON) scripts/test_dataset_audit.py

.PHONY: test-application
test-application:
	$(SBCL) --noinform --no-sysinit --no-userinit --script scripts/test-application.lisp

.PHONY: test-public-install
test-public-install:
	$(PYTHON) scripts/test_public_install.py

.PHONY: test-coverage
test-coverage:
	$(PYTHON) scripts/test_coverage.py

.PHONY: test-bundles
test-bundles:
	$(SBCL) --noinform --no-sysinit --no-userinit --script scripts/test-bundles.lisp

.PHONY: test-bundle-reference example-public-bundle
test-bundle-reference:
	$(PYTHON) scripts/test_bundle_reference.py
example-public-bundle:
	$(SBCL) --noinform --no-sysinit --no-userinit --script examples/public-document-bundle.lisp $(BUNDLE_OUTPUT)

.PHONY: test-footers
test-footers:
	$(PYTHON) scripts/test_footers.py

.PHONY: example-public-markdown
example-public-markdown:
	$(SBCL) --noinform --no-sysinit --no-userinit --script examples/public-page-markdown.lisp $(PUBLIC_MARKDOWN_OUTPUT)

.PHONY: test-public-pdf
test-public-pdf:
	$(PYTHON) scripts/test_public_pdf.py

INSTALL_REPORT ?= .build/install-check.json
.PHONY: test-install
test-install:
	$(PYTHON) scripts/test_install.py

.PHONY: check-native-install
check-native-install:
	$(SBCL) --noinform --no-sysinit --no-userinit --script scripts/check-native-install.lisp "$(INSTALL_REPORT)"

.PHONY: test-adapter-portability
test-adapter-portability:
	$(PYTHON) scripts/test_adapter_portability.py

.PHONY: test-adapted-benchmark
test-adapted-benchmark:
	$(PYTHON) scripts/test_adapted_benchmark.py

.PHONY: test-memory-benchmark
test-memory-benchmark:
	$(PYTHON) scripts/test_memory_benchmark.py

.PHONY: test-vision-lifetimes
test-vision-lifetimes:
	$(SBCL) --noinform --no-sysinit --no-userinit --script scripts/test-vision-lifetimes.lisp

.PHONY: test-upload-benchmark
test-upload-benchmark:
	$(PYTHON) scripts/test_upload_benchmark.py

.PHONY: test-mlx-vlm
test-mlx-vlm:
	$(PYTHON) scripts/test_mlx_vlm.py

.PHONY: test-benchmark
test-benchmark:
	$(PYTHON) scripts/test_benchmark.py

.PHONY: test-selection
test-selection:
	$(PYTHON) scripts/test_selection.py

.PHONY: test-adaptation
test-adaptation:
	$(PYTHON) scripts/test_adaptation.py

.PHONY: example-train-page
example-train-page: build-images
	$(SBCL) --noinform --no-sysinit --no-userinit --script examples/train-page.lisp "$(TRAIN_OUTPUT)"

.PHONY: test-document-training test-document-training-real
test-document-training:
	$(SBCL) --noinform --no-sysinit --no-userinit --script scripts/test-document-training.lisp

test-document-training-real:
	$(SBCL) --noinform --no-sysinit --no-userinit --script scripts/test-document-training-real.lisp

.PHONY: test-training-inputs example-supervision
test-training-inputs: build-images
	$(SBCL) --noinform --no-sysinit --no-userinit --script scripts/test-training-inputs.lisp

example-supervision:
	$(SBCL) --noinform --no-sysinit --no-userinit --script examples/answer-supervision.lisp

.PHONY: test-table-score example-table-recognition
test-table-score:
	$(PYTHON) scripts/test_table_score.py

# Supply a fresh output directory; no model download or automatic retries.
example-table-recognition: build-images
	$(SBCL) --noinform --no-sysinit --no-userinit --script examples/recognize-tables.lisp "$(TABLE_OUTPUT)"

.PHONY: example-table example-merged-table
example-merged-table:
	$(SBCL) --noinform --no-sysinit --no-userinit --script examples/otsl-to-markdown.lisp headed merged

example-table:
	$(SBCL) --noinform --no-sysinit --no-userinit --script examples/otsl-to-markdown.lisp

.PHONY: test example build-images test-images example-images test-images-sanitized example-image-generation example-markdown example-image-markdown
.PHONY: test-pdf example-pdf-markdown
test-pdf:
	$(SBCL) --noinform --no-sysinit --no-userinit --script scripts/test-pdf.lisp

example-pdf-markdown: build-images
	$(SBCL) --noinform --no-sysinit --no-userinit --script examples/pdf-to-markdown.lisp

test:
	$(SBCL) --noinform --no-sysinit --no-userinit --script scripts/test.lisp

example:
	$(SBCL) --noinform --no-sysinit --no-userinit --script examples/vision-layout.lisp

example-markdown:
	$(SBCL) --noinform --no-sysinit --no-userinit --script examples/doctags-to-markdown.lisp

build-images:
	sh scripts/build-images.sh

test-images: build-images
	$(SBCL) --noinform --no-sysinit --no-userinit --script scripts/test-images.lisp

example-images: build-images
	$(SBCL) --noinform --no-sysinit --no-userinit --script examples/native-image-input.lisp

test-images-sanitized:
	sh scripts/test-images-sanitized.sh

example-image-generation: build-images
	$(SBCL) --noinform --no-sysinit --no-userinit --script examples/generate-image.lisp

example-image-markdown: build-images
	$(SBCL) --noinform --no-sysinit --no-userinit --script examples/generate-markdown.lisp
