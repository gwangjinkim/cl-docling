"""Prepare fixed external regression pages; no training data or model predictions."""
import argparse
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
from dataset_audit import ROOT,verify_file
from dpbench_score import validate_manifest


def digest(data): return hashlib.sha256(data).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--parquet',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import pyarrow.parquet as pq
    from PIL import Image
    from docling_core.types.doc.document import DoclingDocument,ContentLayer
    lock_path=ROOT/'references/dpbench.lock.json'
    lock=json.loads(lock_path.read_text())
    for name,version in lock['tools'].items():
        if importlib.metadata.version(name)!=version: raise ValueError('Wrong tool version: '+name)
    verify_file(args.parquet,lock['sha256'],lock['bytes'])
    batch=next(pq.ParquetFile(args.parquet).iter_batches(batch_size=4,
               columns=['document_id','GroundTruthDocument','GroundTruthPageImages']))
    rows=batch.to_pylist()
    if len(rows)!=4: raise ValueError('Missing selected rows')
    args.output.mkdir(parents=True,exist_ok=False)
    cases=[]; files={}
    for index,row in enumerate(rows):
        name=f'page-{index:03d}'
        raw=row['GroundTruthDocument']
        document=DoclingDocument.model_validate_json(raw)
        if len(row['GroundTruthPageImages'])!=1: raise ValueError('Expected one supplied page image')
        data=row['GroundTruthPageImages'][0]['bytes']
        image=Image.open(io.BytesIO(data))
        if image.format!='PNG': raise ValueError('Expected unchanged PNG; no implicit transcoding')
        size=list(image.size); image.verify()
        markdown=document.export_to_markdown(included_content_layers=set(ContentLayer)).rstrip('\n')+'\n'
        tables=[]
        for table in document.tables:
            d=table.data
            tables.append(dict(rows=d.num_rows,columns=d.num_cols,cells=[
                [c.start_row_offset,c.start_col_offset,c.row_span,c.col_span,c.text] for c in d.table_cells]))
        expected=dict(markdown=markdown,
                      text=document.export_to_text(included_content_layers=set(ContentLayer)),tables=tables)
        contents={name+'.png':data,name+'.ground-truth.json':raw.encode('utf-8'),
                  name+'.expected.json':(json.dumps(expected,indent=2)+'\n').encode('utf-8')}
        for filename,value in contents.items():
            (args.output/filename).write_bytes(value); files[filename]=digest(value)
        cases.append(dict(name=name,split='regression',row=index,document_id=row['document_id'],
                          image_sha256=digest(data),ground_truth_sha256=digest(raw.encode('utf-8')),
                          image_size=size,reference_characters=len(markdown),reference_tables=len(tables)))
    manifest=dict(schema_version=1,protocol=lock,protocol_sha256=digest(lock_path.read_bytes()),
                  training_approved=False,cases=cases,files=files)
    validate_manifest(manifest)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for c in cases: print(c['name'],c['image_size'],c['reference_characters'],c['reference_tables'])


if __name__=='__main__': main()
