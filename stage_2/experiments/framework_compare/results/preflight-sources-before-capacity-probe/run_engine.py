#!/usr/bin/env python3
"""One independent inference process: compatibility, quality, or three formal workloads."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics as st
import time

import torch

from common import Inputs,HERE,ROOT,read,write,sha,quality
from adapters import HFAdapter,VLLMAdapter


def summary(rows):
    from vlmbench.metrics import quantiles
    steps=sum(r['decode_steps'] for r in rows)
    return {'questions':len(rows),'anls':st.mean(r['anls'] for r in rows),'em':st.mean(r['em'] for r in rows),
            'mean_ttft_s':st.mean(r['ttft_s'] for r in rows),'mean_e2e_s':st.mean(r['e2e_s'] for r in rows),
            'ttft_s':quantiles(r['ttft_s'] for r in rows),'e2e_s':quantiles(r['e2e_s'] for r in rows),
            'total_e2e_s':sum(r['e2e_s'] for r in rows),'decode_steps':steps,
            'decode_ms_per_token':1000*sum(r['decode_s'] for r in rows)/steps if steps else None,
            'encoder_calls':sum(r['encoder_calls'] for r in rows),'encoded_images':sum(r['encoded_images'] for r in rows),
            'graph_replays':sum(r['graph_replays'] for r in rows),
            'prefix_hit_requests':sum(r['num_cached_tokens']>0 for r in rows),
            'cached_tokens':sum(r['num_cached_tokens'] for r in rows),'input_tokens':sum(r['input_tokens'] for r in rows),
            'output_tokens':sum(len(r['token_ids']) for r in rows),
            'peak_allocated_bytes':max(r['peak_allocated_bytes'] for r in rows),
            'peak_reserved_bytes':max(r['peak_reserved_bytes'] for r in rows)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',required=True,choices=['H0','H1','P','V00','V10','V01','V11'])
    parser.add_argument('--phase',choices=['preflight','development','formal'],default='preflight')
    parser.add_argument('--seed',type=int,default=4101)
    parser.add_argument('--options',type=Path,default=HERE/'options-candidate.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):parser.error('Use a new result directory')
    args.output.mkdir(parents=True,exist_ok=True)
    options=read(args.options);options['seed']=args.seed
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    dataset='evaluation' if args.phase=='formal' else 'development'
    inputs=Inputs(dataset)
    started=time.perf_counter()
    adapter=HFAdapter(inputs,args.variant,options) if args.variant in ['H0','H1','P'] else VLLMAdapter(inputs,args.variant,options,capture_pixels=args.phase=='preflight')
    ready=time.perf_counter()
    manifest={'recorded_at':datetime.now(timezone.utc).isoformat(),'variant':args.variant,'phase':args.phase,'seed':args.seed,
              'options':options,'dataset':dataset,'input_sha256':sha(inputs.file),'initialization_s':ready-started,
              'environment':adapter.info(),'source_sha256':{p.name:sha(p) for p in HERE.glob('*.py')}}
    write(args.output/'manifest.json',manifest)
    print('ENGINE READY',args.variant,round(ready-started,2),flush=True)
    if args.phase=='preflight':
        grouped=defaultdict(list)
        for row in inputs.rows:grouped[row['page_id']].append(row)
        selected=[r for group in [g for g in grouped.values() if len(g)>=2][:3] for r in group[:2]]
        scenarios=[('preflight',[],selected)]
        write(args.output/'expected-inputs.json',[inputs.expected(r) for r in selected])
    elif args.phase=='development':scenarios=[('quality',[],inputs.rows)]
    else:scenarios=inputs.scenarios(args.seed)
    results={}
    for scenario,warmup,requests in scenarios:
        adapter.clear()
        with (args.output/f'{scenario}-warmup.jsonl').open('w') as stream:
            for row in warmup:
                result=adapter.predict(row)
                stream.write(json.dumps({'question_id':row['question_id'],**result},ensure_ascii=False)+'\n')
        diagnostics_before=adapter.diagnostics()
        selected=[];wall_started=time.perf_counter()
        with (args.output/f'{scenario}-requests.jsonl').open('w') as stream:
            for i,row in enumerate(requests):
                result=adapter.predict(row)
                result.update(variant=args.variant,scenario=scenario,seed=args.seed,page_id=row['page_id'],
                              question_id=row['question_id'],**quality(result['answer'],row['answers']))
                assert result['input_tokens']==row.get('input_tokens',result['input_tokens'])
                selected.append(result);stream.write(json.dumps(result,ensure_ascii=False)+'\n');stream.flush()
                if (i+1)%32==0:print('PROGRESS',args.variant,scenario,i+1,'/',len(requests),flush=True)
        result=summary(selected);result['wall_elapsed_s']=time.perf_counter()-wall_started
        result['diagnostics_before']=diagnostics_before;result['diagnostics_after']=adapter.diagnostics()
        results[scenario]=result;write(args.output/'summary.json',results)
        print('SCENARIO DONE',args.variant,scenario,json.dumps({k:v for k,v in result.items() if k not in ['diagnostics_before','diagnostics_after']}),flush=True)
    write(args.output/'complete.json',{'complete':True,'variant':args.variant,'scenarios':list(results)})


if __name__=='__main__':main()
