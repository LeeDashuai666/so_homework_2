#!/usr/bin/env python3
"""Export a new source-document-disjoint evaluation set, without inspecting model outputs."""
from collections import defaultdict, Counter
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys

import pyarrow.parquet as pq
from PIL import Image
from huggingface_hub import constants

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from vlm_runtime import load_processor,prepare_inputs


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out=HERE/'data';out.mkdir(exist_ok=True)
    if (out/'manifest.json').exists():
        print('Reusing frozen framework dataset manifest');return
    (out/'images').mkdir(exist_ok=True)
    spec=json.loads((ROOT/'configs/experiment-assets.lock.json').read_text())['dataset']
    source=Path(constants.HF_HUB_CACHE)/('datasets--'+spec['repo_id'].replace('/','--'))/'snapshots'/spec['revision']
    files=[source/p for p in sorted(spec['files'])]
    old=[]
    for name in ['development','evaluation']:
        old.extend(json.loads(s) for s in (ROOT/f'data/processed/docvqa/{name}.jsonl').read_text().splitlines())
    old_sources={r['source_document_id'] for r in old};old_pixels={r['pixel_sha256'] for r in old}
    columns=['questionId','question','answers','docId','ucsf_document_id','ucsf_document_page_no']
    grouped=defaultdict(list)
    for file in files:
        for row in pq.read_table(file,columns=columns).to_pylist():grouped[str(row['docId'])].append(row)
    sources=defaultdict(list)
    for page,rows in grouped.items():
        doc=str(rows[0]['ucsf_document_id'])
        if doc not in old_sources:sources[doc].append(page)
    rank=lambda value:hashlib.sha256(('docvqa-framework-v1\0'+str(value)).encode()).hexdigest()
    pages=[]
    for doc in sorted(sources,key=rank):
        pages.extend(sorted(sources[doc],key=rank)[:64-len(pages)])
        if len(pages)==64:break
    assert len(pages)==64
    images={}
    for file in files:
        for batch in pq.ParquetFile(file).iter_batches(batch_size=24,columns=['docId','image']):
            for row in batch.to_pylist():
                page=str(row['docId'])
                if page not in pages or page in images:continue
                raw=row['image']['bytes']
                with Image.open(BytesIO(raw)) as image:
                    rgb=image.convert('RGB');pixel=hashlib.sha256(rgb.tobytes()).hexdigest()
                    assert pixel not in old_pixels,'Duplicate RGB page across old and new splits'
                    ext='.png' if image.format=='PNG' else '.jpg';relative=f'images/page-{page}{ext}'
                    (out/relative).write_bytes(raw)
                    images[page]={'image_path':relative,'pixel_sha256':pixel,'file_sha256':hashlib.sha256(raw).hexdigest(),
                                  'width':rgb.width,'height':rgb.height}
    assert set(images)==set(pages)
    cfg=json.loads((ROOT/'configs/protocol.lock.json').read_text())['runtime']
    cfg['dataset_path']=str(out.relative_to(ROOT));processor=load_processor(cfg)
    records=[]
    for page in pages:
        for raw in sorted(grouped[page],key=lambda r:str(r['questionId'])):
            row={'question_id':str(raw['questionId']),'page_id':page,'question':raw['question'],'answers':raw['answers'],
                 'source_document_id':str(raw['ucsf_document_id']),'source_page_number':str(raw['ucsf_document_page_no']),
                 'image_path':images[page]['image_path'],'pixel_sha256':images[page]['pixel_sha256'],
                 'split':'framework_evaluation'}
            cpu,meta=prepare_inputs(processor,[row],cfg)
            assert meta['input_lengths'][0]+64<=1024,'Common static capacity exceeded before protocol freeze'
            row.update(input_tokens=meta['input_lengths'][0],visual_tokens=meta['visual_tokens'][0],grid_thw=meta['grid_thw'][0])
            records.append(row)
    target=out/'evaluation.jsonl';target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    manifest={'created_at':datetime.now(timezone.utc).isoformat(),'selection_namespace':'docvqa-framework-v1',
              'selection_rule':'Source-document SHA256 order, then page SHA256; first 64 pages; every original question retained',
              'source':spec,'pages':len(pages),'questions':len(records),'source_documents':len({r['source_document_id'] for r in records}),
              'old_source_document_overlap':0,'old_rgb_overlap':0,'quality_or_performance_filter':False,
              'question_counts_per_page':dict(Counter(Counter(r['page_id'] for r in records).values())),
              'evaluation_sha256':sha(target),'images':images,
              'old_split_sha256':{n:sha(ROOT/f'data/processed/docvqa/{n}.jsonl') for n in ['development','evaluation']}}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['source','images']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
