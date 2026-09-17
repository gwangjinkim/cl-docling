"""Authored, source-disjoint M6 coverage cases. Freeze before inference."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = dict(task='Convert this page to docling.', max_new_tokens=512,
                device='gpu', dtype='float32', adapter_step=16)


def cases():
    # x/y positions are part of the authored source, not model-derived labels.
    specs = [
        ('prose', [(30,30,'Field station'), (30,90,'The west gate opens at sunrise.'), (30,130,'Visitors sign the blue register.'), (30,170,'Return each key before leaving.')]),
        ('columns', [(30,30,'Coastal survey'), (30,90,'North route'), (30,130,'Check the beacon.'), (30,170,'Count six buoys.'), (320,90,'South route'), (320,130,'Inspect the jetty.'), (320,170,'Record the tide.')]),
        ('lists', [(30,30,'Packing checklist'), (30,90,'- Compass'), (30,130,'- Water bottle'), (30,170,'- Rain jacket'), (30,230,'1. Check the forecast'), (30,270,'2. Tell a friend')]),
        ('code', [(30,30,'Lisp function'), (30,90,'(defun square (number)'), (30,125,'  (* number number))'), (30,190,'(square 7)'), (30,225,'49')]),
        ('formula', [(30,30,'Motion notebook'), (30,90,'distance = speed * time'), (30,135,'d = 12 * 3'), (30,180,'d = 36 km'), (30,245,'Average speed: 12 km/h')]),
        ('furniture', [(30,20,'Observatory circular'), (30,80,'Evening report'), (30,135,'The eastern dome is closed.'), (30,175,'Use the smaller telescope tonight.'), (30,390,'Observatory circular'), (540,430,'6')]),
        ('table', [(30,30,'Sample log')]),
        ('rotated', [(30,30,'Archive notice'), (30,90,'Room seven is ready.'), (30,130,'Leave the lamps switched off.'), (30,170,'Keep the doorway clear.')]),
        ('low-resolution', [(30,30,'Delivery note'), (30,90,'Three parcels arrived on Tuesday.'), (30,130,'Store the fragile box upstairs.'), (30,170,'The receipt is inside the envelope.')]),
    ]
    result=[]
    for name, blocks in specs:
        texts=[text for x,y,text in blocks]
        if name=='lists': texts=['Packing checklist','Compass','Water bottle','Rain jacket','Check the forecast','Tell a friend']
        item=dict(name=name, source='m6-coverage-'+name, split='final', blocks=blocks, expected=' '.join(texts))
        if name=='table':
            item['rows']=[['Site','Count'],['Dune','14'],['Marsh','9']]
            item['expected'] += ' ' + ' '.join(t for row in item['rows'] for t in row)
        result.append(item)
    return result


def main():
    from PIL import Image, ImageDraw, ImageFont, __version__
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    data=cases()
    font=ImageFont.load_default(size=22)
    for case in data:
        image=Image.new('RGB',(600,480),'white'); draw=ImageDraw.Draw(image)
        for x,y,text in case['blocks']:
            bbox=draw.textbbox((x,y),text,font=font)
            assert 0<=bbox[0]<bbox[2]<600 and 0<=bbox[1]<bbox[3]<480, (case['name'],text,bbox)
            draw.text((x,y),text,font=font,fill='black')
        if 'rows' in case:
            for r,row in enumerate(case['rows']):
                for c,text in enumerate(row):
                    x,y=30+c*230,90+r*65
                    draw.rectangle((x,y,x+230,y+65),outline='black',width=2)
                    draw.text((x+12,y+20),text,font=font,fill='black')
        if case['name']=='rotated': image=image.transpose(Image.Transpose.ROTATE_90)
        if case['name']=='low-resolution': image=image.resize((240,192),Image.Resampling.LANCZOS)
        image.save(args.output/(case['name']+'.png'))
    (args.output/'expected.json').write_text(json.dumps(data,indent=2)+'\n')
    (args.output/'final.json').write_text(json.dumps(dict(protocol=PROTOCOL,cases=[dict(name=c['name'],split='final') for c in data]),indent=2)+'\n')
    manifest=dict(protocol=PROTOCOL,pillow=__version__,font='Pillow bundled Aileron, 22px',
                  policy='Nine authored sources; no new training or selection; one case per category; synthetic coverage, not population quality.',
                  engine='da73fdd3d09c4abf870002ae43eb440ac1a1e641',
                  adapter_sha256=json.loads((ROOT/'tests/fixtures/selection-results/selection.json').read_text())['adapter_sha256'],
                  files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.iterdir())})
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__': main()
