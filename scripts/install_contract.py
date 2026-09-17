"""Replay the bounded clean-local-build evidence without model dependencies."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def validate_install(report):
    expected = json.loads((ROOT / 'tests/fixtures/benchmark/python-cpu.json').read_text())['cases']['native-page']
    case = report['case']
    for key in ('tokens', 'raw', 'stop_reason', 'tiles'):
        if case[key] != expected[key]:
            raise ValueError('Installation output mismatch: ' + key)
    if (case['prompt_tokens'] != len(expected['prompt_ids'])
            or case['markdown'] != (ROOT / 'tests/fixtures/doctags/native-page.md').read_text()
            or report['remaining_handles'] != 0 or report['matches_reference'] is not True):
        raise ValueError('Installation processor/Markdown/lifetime gate failed')
    systems = report['source_systems']
    doc = Path(systems['cl-docling']['source'])
    engine = Path(systems['cl-transformer-blocks']['source'])
    if (not doc.is_absolute() or not engine.is_absolute() or doc.parent != engine.parent or engine == doc
            or systems['cl-transformer-blocks']['version'] != '0.41.2'
            or systems['cl-transformer-blocks/kernels']['version'] != '0.4.0'
            or Path(systems['cl-transformer-blocks/kernels']['source']) != engine):
        raise ValueError('Not the qualified sibling engine/kernel setup')
    dependencies = {'cffi', 'yason', 'babel', 'alexandria', 'trivial-features',
                    'trivial-garbage', 'trivial-gray-streams'}
    if set(systems) != dependencies | {'cl-docling', 'cl-transformer-blocks', 'cl-transformer-blocks/kernels'}:
        raise ValueError('Incomplete source-system provenance')
    for name in dependencies:
        if engine / '.build/deps' not in Path(systems[name]['source']).parents:
            raise ValueError('Dependency escaped the fresh engine tree: ' + name)
    expected_libraries = {str(doc / '.build/native/libdocling_images.dylib'),
                          str(engine / '.build/native/libtb_mlx.dylib'),
                          str(engine / '.build/tokenizer/release/libtb_tokenizer.dylib')}
    if set(report['direct_foreign_libraries']) != expected_libraries:
        raise ValueError('Native libraries escaped the fresh build trees')
    if Path(report['checkpoint']) != doc / '.build/smoldocling':
        raise ValueError('Model did not use the local verified copy')
    if report['device'] not in ('cpu', 'gpu'):
        raise ValueError('Unqualified installation device')


def validate_pair(cpu, metal):
    validate_install(cpu)
    validate_install(metal)
    if cpu['device'] != 'cpu' or metal['device'] != 'gpu' or cpu['case'] != metal['case']:
        raise ValueError('Required CPU/Metal pair missing or divergent')
