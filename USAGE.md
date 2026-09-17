# Using cl-docling

Current source: 0.25.0. Qualified native platform: Apple Silicon/macOS, SBCL 2.6.7,
FP32 MLX CPU/Metal. Runtime requires compatible engine 0.41.2 (kernel API 0.4.0).
The compatible public engine revision is
`159ea7c67026c2388622ff2212db242de5a224a2` (0.41.2).

## Offline examples

```sh
make test
make test-bundles
make example-markdown
make example-table
make example-merged-table
make example-public-bundle BUNDLE_OUTPUT=.build/my-bundle/
```

Only SBCL/ASDF and Make are needed. These parse authored/saved DocTags, not new
model output. The bundle example preserves all 1,089 recorded NASA output tokens.
Use new output directories; never overwrite earlier results to retry.

## Native installation

Provide Python 3.12, uv, CMake, a C/C++20 compiler, Rust/rustup, SBCL, and libpng.
Keep compatible engine and document source directories as siblings. On macOS,
the image build defaults to `/opt/homebrew/opt/libpng`; set `DOCLING_PNG_PREFIX`
if your libpng installation differs. The engine pins its Rust toolchain.

Clone the public sources into a new parent directory, then build the engine:

```sh
git clone https://github.com/gwangjinkim/cl-transformer-blocks.git
git clone https://github.com/gwangjinkim/cl-docling.git
cd cl-transformer-blocks
git checkout --detach 159ea7c67026c2388622ff2212db242de5a224a2
uv sync --frozen
.venv/bin/python scripts/bootstrap.py
cd ../cl-docling
```

In cl-docling:

```sh
uv sync --frozen
make build-images
.venv/bin/python scripts/model-manifest.py .build/smoldocling \
  --download --check references/smoldocling.lock.json
TB_DEVICE=cpu make check-native-install INSTALL_REPORT=.build/install-cpu.json
TB_DEVICE=gpu make check-native-install INSTALL_REPORT=.build/install-metal.json
```

Setup downloads dependencies and the pinned approximately 518 MB model; native
execution subsequently uses local files. These operations do not publish anything.
`references/smoldocling.lock.json` pins model/config/tokenizer/processor hashes.
Model terms are CDLA-Permissive-2.0, separate from this source's MIT license.
The install check requires exact 152-token reference output and no live tensor
handles afterward. A requested unavailable GPU is an error, never a CPU fallback.

On 18 September 2026, fresh GitHub clones of document revision
`305b166225c7aa7d7f6b9cb4082ed0ab74b21c45` and the engine revision above rebuilt
native libraries and new Python environments. CPU and Metal each matched all
152 reference tokens and Markdown, with zero remaining tensor handles. The run
reused this host's tools, locked package/source caches and a model copy checked
against all 13 pinned file hashes. It did not test a second machine, a cache-empty
OS, or a new Hugging Face download. Build products depend on local library paths;
do not move an old compiled directory in place of rebuilding. Later release-only
documentation/evidence changes do not alter the qualified runtime sources.

## Run a page and train an adapter

```sh
TB_DEVICE=gpu make example-image-generation
TB_DEVICE=gpu make example-train-page TRAIN_OUTPUT=.build/my-adapter/
```

The training example performs two updates on a synthetic grid with answer-only
labels and frozen vision features. It is not the selected step-16 adapter used in
the recorded quality/benchmark reports. Lower loss alone is not better OCR.
Inspect [the complete example](examples/train-page.lisp) before changing it.

For separate source directories, set `DOCLING_ENGINE`, `TB_DEPENDENCY_ROOT`,
`TB_MLX_LIBRARY`, `TB_TOKENIZER_LIBRARY` and `DOCLING_IMAGE_LIBRARY` to the prepared
local paths. `DOCLING_MODEL` selects a prepared checkpoint; it never downloads one.
The default sibling layout avoids these overrides.

Export your adapter as an ordinary merged model (this helper uses Metal):

```sh
export DOCLING_MODEL="$PWD/.build/smoldocling/"
DOCLING_BENCHMARK_ADAPTER="$PWD/.build/my-adapter/" \
  sbcl --dynamic-space-size 4096 --noinform --no-sysinit --no-userinit \
  --script scripts/export-selected-model.lisp .build/my-merged-model/
uv run --no-sync python scripts/benchmark-python.py \
  --checkpoint .build/my-merged-model --output .build/my-python-results.json
```

Despite its historical name, the export helper merges the supplied adapter; it
does not select/train it. Adapter identity must match the original base path.
The Python command runs ordinary Transformers CPU generation on three images,
including repeated timing. It does not use a Lisp loader. This is local file
interchange; no Hugging Face upload or remote model reload is claimed.

This two-update recipe was executed in the fresh public clone. Loss was
0.74863887 then 0.74848735; the exported model loaded in ordinary Transformers.
Native and Python outputs match exactly on all three demo pages (152, 56 and
53 tokens). This verifies interchange, not a quality gain. The recorded smoke-run
timings are not a replacement for the controlled benchmarks below.

## Parse and save results

```lisp
(asdf:load-system "cl-docling/bundles")
(let ((page (docling:parse-doctags "<doctag><text>Hello.</text></doctag>"
                                  :page-number 1)))
  (docling:write-document-bundle (list page) #P"/existing/parent/new-bundle/"))
```

For actual generation, pass the real token-ID vector and `:stop-reason` (`:eos`
or `:length`) to the parser. The example above has unknown generation status.
The writer preserves raw text, supplied IDs/stop reasons, diagnostics, and allowed
Markdown. `manifest.sexp` is published last: absent means incomplete; `:blocked`
means evidence saved without Markdown; `:partial` means explicitly permitted
warnings; `:complete` means parser-clean export, not correct recognition.
Output parents must exist; existing destinations are rejected. I/O failures can
leave incomplete artifacts. No fsync/durability, hostile-directory protection,
automatic resume, artifact loader or cryptographic seal is provided.

`documents-to-markdown` combines an ordered, nonempty list of parsed pages with
labeled boundaries. It rejects duplicate/reversed pages and defaults to 256 pages
and 8,000,000 output characters. Any diagnostic blocks strict export. Explicit
`:allow-partial t` preserves global/page warnings and page-scoped diagnostics.
It does not repair OCR, merge cross-page tables or deduplicate repeated furniture.

Supported DocTags include text, titles/headings, flat lists, page footers and
validated OTSL tables, including supported spans. Unsupported/malformed/truncated
elements retain raw evidence and block strict rendering. Page headers, complex
table semantics, arbitrary formulas/code structures and nested lists remain limited.

## Inspect measurements, not marketing claims

```sh
make test-selection test-adapted-benchmark test-public-pdf
make test-benchmark test-mlx-vlm test-upload-benchmark test-memory-benchmark
make test-install test-public-install test-bundle-reference
make test-coverage
```

These Python-standard-library tests replay committed evidence, not a fresh model
run. All results and failures are kept under `tests/fixtures/` and `docs/evidence/`.
On one M3 Max, a selected-adapter library page takes median 0.99 s in merged
Lisp/Metal, 0.82 s in controlled Python/MLX-VLM Metal and 6.78 s in Transformers CPU.
These do not isolate language overhead or establish sustained throughput. Python
MLX is faster in that comparison. Different backend/device timings stay labeled.

The shared-template synthetic final test improves 2/4 to 3/4 accepted pages; its
remaining heading-span failure is retained. Real NASA outputs have 34/36 and
22/28 exact cells despite exact Lisp/Python token agreement. Per-page results and
scope of the new nine-category coverage evaluation accompany its saved reports.
That frozen synthetic check passes 8/9 pages with the base and 7/9 with the selected
adapter; all 18 outputs match Python. Literal list markers fail both strict text
references, and the adapter adds a malformed table close. Both have micro-CER
1.47% and micro-WER 3.62%; these lexical metrics conceal the export regression.
This is one source per category, not broad real-scan reliability or a semantic
code/formula benchmark. The adapter was not retrained or reselected on these pages.
Do not infer population accuracy from a few synthetic or adjacent public pages.

No automatic folder service, Gemma/audio support, CUDA qualification, vision/full-
model fine-tuning, model-weight distribution or general PDF reliability is promised.
For a contribution, include a permitted source, frozen reference, exact commands,
observed errors and platform; keep private documents and model caches out of Git.
