"""Independent authored geometry oracle, using pinned upstream Docling only."""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
from docling_core.types.doc.tokens import DocumentToken

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
assert importlib.metadata.version('docling-core') == '2.97.0'
cases = []
for size, box in [((1000, 2000), (0, 0, 1000, 2000)),
                  ((2000, 1000), (200, 100, 1800, 900)),
                  ((1000, 1000), (1, 3, 5, 7)),
                  ((1786, 2526), (127, 177, 1656, 1036)),
                  ((2376, 1836), (199, 187, 2179, 288)),
                  ((3, 7), (1, 2, 2, 6)),
                  ((1000, 2000), (0, 0, 0, 0)),
                  ((1000, 2000), (999, 1999, 1000, 2000))]:
    cases.append(dict(size=size, box=box, expected=DocumentToken.get_location(box, *size)))
source = Path(inspect.getfile(DocumentToken))
report = dict(docling_core='2.97.0',
              source='https://github.com/docling-project/docling-core/blob/v2.97.0/docling_core/types/doc/tokens.py',
              source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
              license='MIT', cases=cases)
with args.output.open('x') as stream:
    stream.write(json.dumps(report, indent=2) + '\n')
