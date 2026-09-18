# Using cl-docling

cl-docling 0.27.0 runs the pinned SmolDocling vision-language architecture from
Common Lisp on Apple Silicon. It uses cl-transformer-blocks for transformer and
MLX operations, preserves Hugging Face processor/tokenizer semantics, supports
bounded LoRA training, and writes exports that ordinary Python Transformers can
reload.

This guide describes only the material shipped in the public repository. The
historical book pages used for the final local evaluation are not redistributed;
their hash-bound aggregate report is public so the result and its failures remain
inspectable.

## Start without model weights

With SBCL, ASDF, Make, and Python 3:

```sh
make test
make example-markdown
make example-table
make example-merged-table
make example-public-bundle BUNDLE_OUTPUT=.build/my-bundle/
make test-book-final-evaluation
```

These commands parse synthetic or recorded DocTags and replay shipped evidence.
They do not download a model, run OCR, or reproduce private source pages. Use a
new output directory for every bundle example.

## Install the native model

Provide Python 3.12, uv, CMake, a C/C++20 compiler, Rust/rustup, SBCL, and libpng.
On macOS the image build defaults to `/opt/homebrew/opt/libpng`; set
`DOCLING_PNG_PREFIX` if needed. Clone the compatible engine and this project as
siblings:

```sh
git clone https://github.com/gwangjinkim/cl-transformer-blocks.git
git clone https://github.com/gwangjinkim/cl-docling.git
cd cl-transformer-blocks
git checkout --detach 159ea7c67026c2388622ff2212db242de5a224a2
uv sync --frozen
.venv/bin/python scripts/bootstrap.py
cd ../cl-docling
uv sync --frozen
make build-images
.venv/bin/python scripts/model-manifest.py .build/smoldocling \
  --download --check references/smoldocling.lock.json
TB_DEVICE=cpu make check-native-install INSTALL_REPORT=.build/install-cpu.json
TB_DEVICE=gpu make check-native-install INSTALL_REPORT=.build/install-metal.json
```

Setup downloads dependencies and the pinned roughly 518 MB checkpoint. Native
inference then uses local files and does not call a Python inference service.
The model/config/tokenizer/processor files are checked against pinned hashes.
Model terms are separate from this source repository's MIT license.

The qualified native platform is Apple Silicon/macOS with SBCL 2.6.7 and FP32
MLX CPU/Metal. A requested but unavailable GPU is an error, never a silent CPU
fallback. Other operating systems, CUDA, other architectures, and other model
families are not yet qualified.

## Convert a PDF, with explicit resume

Poppler, GNU `timeout`, and `shasum` are required for the PDF command:

```sh
bin/cl-docling --help
bin/cl-docling --input /documents/notes.pdf --output /outputs/new-job \
  --model /models/smoldocling --pages 1-4 --device gpu
bin/cl-docling --input /documents/notes.pdf --output /outputs/new-job \
  --model /models/smoldocling --pages 1-4 --device gpu --resume
```

Defaults are page 1, CPU, 144 DPI, and 512 new tokens. At most 16 increasing
pages may be selected. The command reuses one loaded model, records each attempt,
and publishes the bundle manifest last. Resume verifies the source, checkpoint,
runtime libraries, and options before skipping completed generation.

Exit 0 means parser-clean Markdown export, not correct OCR. Exit 2 means saved
diagnostics block Markdown, 3 means a page execution failed, 1 is setup/job
failure, and 64 is invalid command syntax. Truncated or malformed generations
remain recorded failures and are not retried or repaired automatically.

This is trusted-local recovery, not an authenticated cache. There is no fsync
durability guarantee, disk quota, hard inference wall-time, automatic cleanup,
or safe handling promise for hostile PDFs and directories. An abrupt process
death can leave a stale lock; inspect processes before removing it manually.

## Run and train from Lisp

```sh
TB_DEVICE=gpu make example-image-generation
TB_DEVICE=gpu make example-train-page TRAIN_OUTPUT=.build/my-adapter/
```

The training example performs two updates on a synthetic grid using answer-only
labels, frozen vision features, and decoder LoRA. It is an interchange smoke test,
not evidence of general OCR improvement. Lower loss alone is not acceptance.

To use separate source directories, set `DOCLING_ENGINE`, `TB_DEPENDENCY_ROOT`,
`TB_MLX_LIBRARY`, `TB_TOKENIZER_LIBRARY`, and `DOCLING_IMAGE_LIBRARY` to prepared
local paths. `DOCLING_MODEL` selects an existing checkpoint; it never downloads
one.

## Export and reload in Python

Merge a compatible adapter into an ordinary checkpoint, then use the Python
reference runner:

```sh
export DOCLING_MODEL="$PWD/.build/smoldocling/"
DOCLING_BENCHMARK_ADAPTER="$PWD/.build/my-adapter/" \
  sbcl --dynamic-space-size 4096 --noinform --no-sysinit --no-userinit \
  --script scripts/export-selected-model.lisp .build/my-merged-model/
uv run --no-sync python scripts/benchmark-python.py \
  --checkpoint .build/my-merged-model --output .build/my-python-results.json
```

The exporter checks adapter/base identity. Transformers loads the resulting local
directory normally; it does not use a Lisp loader. This proves local file-format
interchange. Publishing weights to Hugging Face is a separate, deliberate action.

## Parse DocTags and preserve evidence

```lisp
(asdf:load-system "cl-docling/bundles")
(let ((page (docling:parse-doctags
             "<doctag><text>Hello.</text></doctag>"
             :page-number 1)))
  (docling:write-document-bundle
   (list page) #P"/existing/parent/new-bundle/"))
```

For real generations, also supply the actual token IDs and stop reason. Bundles
preserve raw DocTags, IDs, diagnostics, stop reasons, and permitted Markdown.
Strict export fails closed on malformed, unsupported, or truncated structures.
Supported output includes text, headings, flat lists, page footers, footnote
paragraphs, and validated OTSL tables. Arbitrary formulas, code structures,
nested lists, complex tables, and cross-page semantic repair remain limited.

## Read the final experiment honestly

The public compact report is
`tests/fixtures/book-final/report.json`; its policy is
`references/book-final-evaluation.lock.json`. Validate it with:

```sh
make test-book-final-evaluation
```

Six bounded native updates changed the candidate adapters and lowered training
loss, but neither candidate improved the validation set. The frozen protocol
therefore selected the unchanged base model. On the three protected final pages:

- 2/3 outputs passed strict parser/export checks;
- 1/3 was exact;
- one illustrated page repeated malformed text to the 2,048-token limit;
- aggregate Markdown CER was 0.14696 and WER was 0.10;
- fresh Transformers CPU reproduced all 2,481 native token IDs and raw DocTags;
- native Metal took 27.39 seconds with 966,901,760-byte peak RSS, while the
  differently configured Python CPU reference took 76.22 seconds and
  3,528,081,408-byte peak RSS.

Those backend timings are not a language benchmark. Three pages do not establish
population-level PDF accuracy. The retained base fallback and failed page are the
important result: the stack can perform a controlled native-to-Python round trip,
but this small fine-tune did not improve document conversion.

## Contribute

Useful contributions include a small redistributable failing page, an independently
checked processor fixture, a malformed-DocTags regression, a measured native
optimization, or support for a new architecture with its own processor, numerical,
generation, and interchange tests. Include the exact reproduction command and
expected versus observed behavior. Preserve failures and original inputs; do not
submit private documents, caches, or unlicensed model/data artifacts.

See `README.md`, `LICENSE`, and `THIRD-PARTY-NOTICES.txt`. Open an issue or pull
request at https://github.com/gwangjinkim/cl-docling.
