# cl-docling

Native Common Lisp document models, using
[cl-transformer-blocks](https://github.com/gwangjinkim/cl-transformer-blocks) and MLX.

Load a Hugging Face SmolDocling checkpoint, run its vision encoder and decoder
from Lisp, train frozen-vision decoder LoRA, and export ordinary model/adapter
files that Python Transformers can reload. Python is setup/reference tooling,
not the native inference implementation.

Start with [USAGE.md](USAGE.md) for installation, runnable examples, training,
Python reload, reproducible measurements and current limitations.

Version 0.27.0 adds header/footnote paragraphs to strict DocTags export. Replaying
four frozen dpbench outputs improves export from 0/4 to 1/4, with unchanged tokens
and recognition errors. Exact-quality acceptance remains 0/4; no new model run.

The bounded real-book experiment is now complete. Six native updates changed the
adapters and lowered loss, but validation found no conversion improvement and
selected the original base. On three protected final pages, 2/3 export strictly
and 1/3 is exact; one illustrated page reaches the 2,048-token cap. Fresh ordinary
Transformers reproduces all 2,481 native IDs and raw DocTags exactly. The failure
and base fallback are retained—this is interoperability evidence, not an OCR gain.

Version 0.26.0 adds a bounded resumable PDF command after native setup:

```sh
bin/cl-docling --input notes.pdf --output new-job --pages 1-4 --device gpu
```

It reuses one model, saves per-page evidence and skips completed generation with
an explicit compatible `--resume`. Selection defaults to page 1 only. Truncated
or malformed outputs remain visible failures, not automatically repaired Markdown.

## Try it without a model

```sh
make test
make example-markdown
make example-public-bundle
```

The examples replay recorded outputs; they do not perform OCR. Requirements are
SBCL/ASDF and Make; the optional bundle writer uses SBCL/POSIX. Python 3 is needed
for recorded-score tests, not these examples. Choose fresh output directories
when repeating examples. Native model execution is qualified on Apple Silicon.

## What works, and what does not

- Native FP32 SmolDocling vision and cached generation on CPU/Metal, preserving
  Hugging Face processor/tokenizer semantics and every image tile.
- Frozen-vision decoder LoRA, native optimizer resume, PEFT interchange and merged
  exports loadable by ordinary Python Transformers.
- PNG input, bounded optional Poppler rasterization, conservative DocTags parsing,
  tables/headers/footers/footnote paragraphs, multi-page Markdown and evidence-preserving local bundles.
- Exact numerical interchange is not OCR accuracy. The selected adapter improves
  one narrow synthetic final test from 2/4 to 3/4 accepted pages. Two real NASA
  pages score 34/36 and 22/28 exact cells; both fail exact-table quality.
  A fresh nine-category synthetic check passes 8/9 with the base and 7/9 with
  the selected adapter; the regression and every Python-matching output are retained.
- Only the pinned SmolDocling architecture is qualified. No Gemma, audio, CUDA,
  arbitrary vision models, full vision training or general-purpose PDF reliability
  is implied. Model-weight publication is separate from source publication.

The source tree includes raw evaluation/benchmark evidence and Markdown test
fixtures. Private planning documents and development history are not part of this
public snapshot. This is an independent project, not an official Docling release.

## Build it together

Good contributions include a small permitted failing page, independently checked
processor fixture, malformed-DocTags regression, or a measured native optimization.
Provide the reproduction command and expected versus observed behavior; preserve
failures and original inputs. Do not submit private documents or model caches.
Open an issue or pull request in this repository. New model families need their
own architecture, processor, numerical and generation checks.

MIT licensed. See [LICENSE](LICENSE) and
[third-party notices](THIRD-PARTY-NOTICES.txt). Model/data/dependency licenses remain
separate; the source license does not relicense model weights.
