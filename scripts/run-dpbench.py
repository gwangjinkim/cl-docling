"""Bounded foreground native smoke baseline on pinned external regression inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from datetime import datetime,timezone
from dataset_audit import ROOT,verify_file
from dpbench_score import validate_manifest,score


def digest(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser()
    for name in ('inputs','model','output'): parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--replay',action='store_true',help='Score existing native outputs, never run inference')
    args=parser.parse_args()
    inputs,model,output=args.inputs.resolve(),args.model.resolve(),args.output.resolve()
    manifest=json.loads((inputs/'manifest.json').read_text()); validate_manifest(manifest)
    protocol_path=ROOT/'references/dpbench.lock.json'
    if manifest['protocol']!=json.loads(protocol_path.read_text()) or manifest['protocol_sha256']!=digest(protocol_path):
        raise ValueError('Protocol changed')
    for name,wanted in manifest['files'].items():
        if Path(name).name!=name or digest(inputs/name)!=wanted: raise ValueError('Input file identity differs')
    lock=json.loads((ROOT/'references/smoldocling.lock.json').read_text())
    for name,spec in lock['files'].items(): verify_file(model/name,spec['sha256'],spec['bytes'])
    provenance=output/'run.json'
    if not args.replay:
        output.mkdir(parents=True,exist_ok=False)
        started=datetime.now(timezone.utc).isoformat()
        engine=ROOT.parent/'cl-transformer-blocks'
        environment={**os.environ,'DOCLING_MODEL':str(model),'TB_DEVICE':'gpu','DOCLING_ENGINE':str(engine)}
        command=['sbcl','--dynamic-space-size','4096','--noinform','--no-sysinit','--no-userinit',
                 '--script','scripts/dpbench-experiment.lisp',str(inputs),str(output/'native')]
        run=dict(started=started,command=command,platform=platform.platform(),
                 source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                 manifest_sha256=digest(inputs/'manifest.json'),model_revision=lock['revision'],
                 engine_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=engine,text=True).strip(),
                 direct_libraries_sha256={str(p):digest(p) for p in
                     (engine/'.build/native/libtb_mlx.dylib',ROOT/'.build/native/libdocling_images.dylib')},
                 implementation_files={str(p.relative_to(ROOT)):digest(p) for p in
                     [Path(__file__),ROOT/'scripts/dpbench_score.py',ROOT/'scripts/dpbench-experiment.lisp',
                      ROOT/'scripts/experiment-common.lisp',*sorted((ROOT/'src').glob('*.lisp'))]})
        with (output/'native.log').open('x') as log:
            try:
                completed=subprocess.run(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,timeout=600)
                run['exit_code']=completed.returncode
            except subprocess.TimeoutExpired:
                run['exit_code']=None; run['timeout']=True
        run['finished']=datetime.now(timezone.utc).isoformat()
        provenance.write_text(json.dumps(run,indent=2)+'\n')
    run=json.loads(provenance.read_text())
    if run['manifest_sha256']!=digest(inputs/'manifest.json'): raise ValueError('Replay input changed')
    reports={}; retained={}
    for case in manifest['cases']:
        name=case['name']; base=output/'native/base'
        path=base/(name+'.json'); error=base/(name+'.error.json'); md=base/(name+'.md')
        result=json.loads(path.read_text()) if path.is_file() else {'error':'missing native case'}
        if error.is_file(): result['error']=json.loads(error.read_text())['error']
        expected=json.loads((inputs/(name+'.expected.json')).read_text())
        reports[name]=score(expected,result,md.read_text() if md.is_file() else None)
        reports[name]['generated_tokens']=len(result.get('tokens',[]))
        for artifact in (path,error,md,base/(name+'.doctags')):
            if artifact.is_file(): retained[str(artifact.relative_to(output))]=digest(artifact)
    completion=output/'native/completion.json'
    completed=json.loads(completion.read_text()) if completion.is_file() else None
    summary=dict(pages=4,strict_exports=sum(r['strict_export'] for r in reports.values()),
                 accepted=sum(r['accepted'] for r in reports.values()),
                 reference_tables=sum(r['expected_tables'] for r in reports.values()),
                 execution_failures=sum(r['execution_error'] for r in reports.values()))
    for prefix,edits,total in [('markdown_cer','character_edits','reference_characters'),
                              ('markdown_wer','word_edits','reference_words'),
                              ('raw_text_cer','raw_text_character_edits','raw_text_reference_characters'),
                              ('raw_text_wer','raw_text_word_edits','raw_text_reference_words')]:
        denominator=sum(r[total] for r in reports.values())
        summary[prefix]=sum(r[edits] for r in reports.values())/denominator if denominator else None
    report=dict(schema_version=1,protocol=manifest['protocol'],manifest_sha256=digest(inputs/'manifest.json'),
                model_revision=lock['revision'],run_sha256=digest(provenance),
                completion=completed,retained_sha256=retained,summary=summary,cases=reports)
    name='replay-report.json' if args.replay else 'report.json'
    with (output/name).open('x') as stream: stream.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    if run['exit_code']!=0 or not completed or completed.get('remaining_handles')!=0 or summary['execution_failures']:
        raise SystemExit('Incomplete execution; every selected case remains in the report')


if __name__=='__main__': main()
