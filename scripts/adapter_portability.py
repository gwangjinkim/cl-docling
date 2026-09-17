"""Explicit offline file packaging, never a native inference fallback.

Seal while the original immutable local base exists; relocate against an exact
copy later. A manifest checks byte identity, not authenticity or training history.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

# Complete flat-file contract of the qualified document loader, including absence.
# README is deliberately included: even documentation changes require a new seal.
BASE_FILES = (
    'config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json',
    'special_tokens_map.json', 'added_tokens.json', 'vocab.json', 'merges.txt',
    'generation_config.json', 'chat_template.json', 'chat_template.jinja',
    'preprocessor_config.json', 'processor_config.json', 'README.md',
)
ADAPTER_FILES = ('adapter_config.json', 'adapter_model.safetensors')
MANIFEST = 'portable_adapter.json'


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key: ' + key)
        result[key] = value
    return result


def read_json(path):
    with path.open(encoding='utf-8') as stream:
        result = json.load(stream, object_pairs_hook=unique_object)
    if not isinstance(result, dict):
        raise ValueError('Expected JSON object: ' + str(path))
    return result


def file_identity(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError('Expected a regular non-symlink file: ' + str(path))
    digest, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
            size += len(block)
    return dict(sha256=digest.hexdigest(), bytes=size)


def base_inventory(directory):
    entries = {p.name for p in directory.iterdir()}
    if '.cache' in entries:
        cache = directory / '.cache'
        if cache.is_symlink() or not cache.is_dir():
            raise ValueError('Download metadata must be a non-symlink .cache directory')
        entries.remove('.cache')  # Hugging Face bookkeeping, never consumed by the native loader.
    if entries - set(BASE_FILES) or not {'config.json', 'model.safetensors'} <= entries:
        raise ValueError('Base must be a supported flat checkpoint, without extra entries')
    return {name: file_identity(directory / name) if name in entries else None for name in BASE_FILES}


def adapter_inventory(directory):
    return {name: file_identity(directory / name) for name in ADAPTER_FILES}


def local_config(directory):
    config = read_json(directory / 'adapter_config.json')
    source = config.get('base_model_name_or_path')
    if not isinstance(source, str) or not Path(source).is_absolute() or config.get('revision') is not None:
        raise ValueError('Only absolute local bases with null/absent revision can be staged')
    return config


def validate_inventory(value, names, required):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError('Incomplete or unsupported manifest file inventory')
    for name, identity in value.items():
        if identity is None and name not in required:
            continue
        if (not isinstance(identity, dict) or set(identity) != {'sha256', 'bytes'}
                or not isinstance(identity['sha256'], str)
                or not re.fullmatch('[0-9a-f]{64}', identity['sha256'])
                or type(identity['bytes']) is not int or identity['bytes'] < 0):
            raise ValueError('Invalid file identity: ' + name)


def write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def new_output(path):
    # Do not resolve a dangling symlink and accidentally create its target.
    path = Path(path).absolute()
    path.mkdir(parents=True, exist_ok=False)
    return path


def separate_output(output, *inputs):
    candidate = Path(output).resolve()
    if any(candidate == source or source in candidate.parents for source in inputs):
        raise ValueError('Output must be outside every input directory')


def seal(base, adapter, output):
    base, adapter = Path(base).resolve(strict=True), Path(adapter).resolve(strict=True)
    separate_output(output, base, adapter)
    factors = adapter_inventory(adapter)
    config = local_config(adapter)
    if Path(config['base_model_name_or_path']).resolve() != base:
        raise ValueError('Adapter does not identify this source base')
    manifest = dict(schema_version=1, base_files=base_inventory(base), adapter_files=factors)
    output = new_output(output)
    for name in ADAPTER_FILES:
        shutil.copyfile(adapter / name, output / name)
    if adapter_inventory(output) != factors:
        raise ValueError('Adapter changed while being copied; output is incomplete')
    # Completion marker is written last. Source files must remain immutable.
    write_json(output / MANIFEST, manifest)
    return manifest


def relocate(bundle, base, output):
    bundle, base = Path(bundle).resolve(strict=True), Path(base).resolve(strict=True)
    separate_output(output, bundle, base)
    manifest_identity = file_identity(bundle / MANIFEST)
    manifest = read_json(bundle / MANIFEST)
    if (set(manifest) != {'schema_version', 'base_files', 'adapter_files'}
            or type(manifest['schema_version']) is not int or manifest['schema_version'] != 1):
        raise ValueError('Unsupported portable adapter manifest')
    validate_inventory(manifest['base_files'], BASE_FILES, {'config.json', 'model.safetensors'})
    validate_inventory(manifest['adapter_files'], ADAPTER_FILES, set(ADAPTER_FILES))
    if adapter_inventory(bundle) != manifest['adapter_files']:
        raise ValueError('Sealed adapter bytes changed')
    if base_inventory(base) != manifest['base_files']:
        raise ValueError('Target base files do not match the sealed source')
    config = local_config(bundle)
    source = config['base_model_name_or_path']
    # Match the engine's canonical directory namestring (including final slash).
    config['base_model_name_or_path'] = base.as_posix().rstrip('/') + '/'
    output = new_output(output)
    shutil.copyfile(bundle / 'adapter_model.safetensors', output / 'adapter_model.safetensors')
    write_json(output / 'adapter_config.json', config)
    staged = adapter_inventory(output)
    if staged['adapter_model.safetensors'] != manifest['adapter_files']['adapter_model.safetensors']:
        raise ValueError('Factors changed while being copied; output is incomplete')
    receipt = dict(schema_version=1, source_manifest=manifest_identity,
                   source_base=source, target_base=config['base_model_name_or_path'],
                   changed_config_fields=(['base_model_name_or_path']
                                          if source != config['base_model_name_or_path'] else []),
                   source_adapter_files=manifest['adapter_files'], staged_adapter_files=staged,
                   base_files=manifest['base_files'])
    write_json(output / 'relocation.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command, source in (('seal', 'adapter'), ('relocate', 'bundle')):
        sub = commands.add_parser(command)
        for name in ('base', source, 'output'):
            sub.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'seal':
        seal(args.base, args.adapter, args.output)
    else:
        relocate(args.bundle, args.base, args.output)
    print('PASS: ' + args.command + ' completed; no model execution or upload.')


if __name__ == '__main__':
    main()
