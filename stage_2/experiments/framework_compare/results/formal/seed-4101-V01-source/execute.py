#!/usr/bin/env python3
"""Sequential development gate, frozen protocol, and 21 main + 3 capacity-control starts."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess

from launch import run,HERE,ROOT,IMAGE

VARIANTS=['H0','H1','P','V00','V10','V01','V11','V11E']
PERFORMANCE_SOURCE=['common.py','adapters.py','run_engine.py','launch.py','execute.py','prepare_data.py']


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def develop():
    order=list(VARIANTS);random.Random(4099).shuffle(order)
    for variant in order:run(variant,'development',4101,HERE/'results/development'/variant,HERE/'options-candidate.json')
    rows={v:[json.loads(x) for x in (HERE/'results/development'/v/'quality-requests.jsonl').read_text().splitlines()] for v in VARIANTS}
    baseline={r['question_id']:r for r in rows['H0']}
    scores={v:sum(r['anls'] for r in records)/len(records) for v,records in rows.items()}
    checks={}
    for variant in VARIANTS:
        data=rows[variant]
        assert {r['question_id'] for r in data}==set(baseline)
        checks[variant]={'questions':len(data),'anls':scores[variant],'delta_vs_H0':scores[variant]-scores['H0'],
                         'input_ids_match':all(r['prompt_sha256']==baseline[r['question_id']]['prompt_sha256'] for r in data),
                         'identical_output_tokens':sum(r['token_ids']==baseline[r['question_id']]['token_ids'] for r in data),
                         'encoder_calls':sum(r['encoder_calls'] for r in data),
                         'graph_probe_replays':sum(r['graph_replays'] for r in data),
                         'cached_tokens':sum(r['num_cached_tokens'] for r in data)}
    passed=all(c['delta_vs_H0']>=-0.005 and c['input_ids_match'] for c in checks.values())
    gate={'passed':passed,'max_anls_drop':0.005,'baseline':'fresh H0 in this development run',
          'read_split':'original development only','variants':checks}
    write(HERE/'results/development/quality-gate.json',gate)
    print('DEVELOPMENT GATE',json.dumps(gate,ensure_ascii=False),flush=True)
    if not passed:raise RuntimeError('Quality/input gate failed; preserve data and resolve before formal evaluation')
    for variant in ['V00','V01']:
        run(variant,'profile',4101,HERE/'results/profile'/variant,HERE/'options-candidate.json')


def freeze():
    lock=HERE/'protocol.lock.json'
    if lock.exists():return read(lock)
    gate=read(HERE/'results/development/quality-gate.json');assert gate['passed']
    options=read(HERE/'options-candidate.json');options['note']='Frozen after development quality and compatibility gates'
    write(HERE/'options.lock.json',options)
    source={name:sha(HERE/name) for name in PERFORMANCE_SOURCE}
    reference=read(ROOT/'configs/protocol.lock.json')
    for name,expected in reference['source_sha256'].items():assert sha(ROOT/name)==expected
    image=read_image()
    jobs=[]
    for seed in [4101,4102,4103]:
        order=list(VARIANTS);random.Random(seed).shuffle(order)
        jobs.extend({'seed':seed,'variant':variant} for variant in order)
    cfg={'frozen_at':datetime.now(timezone.utc).isoformat(),'source_sha256':source,
         'options_sha256':sha(HERE/'options.lock.json'),'dataset_manifest_sha256':sha(HERE/'data/manifest.json'),
         'evaluation_sha256':sha(HERE/'data/evaluation.jsonl'),'image':image,
         'model_revision':read(ROOT/'configs/vlm-assets.lock.json')['model']['revision'],
         'quality_gate':gate,'variants':VARIANTS,'main_starts':21,'capacity_control_starts':3,'jobs':jobs,
         'scenarios':['hot','session','unique'],'input_boundary':'Resident verified RGB images and prompt text -> CPU-visible first token / final answer',
         'timing_exclusions':['image file I/O','loading resident image set','model initialization','between-request diagnostics/logging'],
         'resource_policy':'Same 8188 MiB GPU, serial runs, fixed pool byte quotas; actual reserved/device usage recorded. Not equal persistent-cache representations.',
         'vllm_offline_core':'InprocClient, VLLM_ENABLE_V1_MULTIPROCESSING=0; no HTTP',
         'encoder_capacity_control':'V11E sets scheduler_config.encoder_cache_size=16384 after config creation; compute budget stays 1024; no framework source edits',
         'reference_source_sha256':reference['source_sha256'],
         'prototype_runtime_sha256':sha(ROOT/'experiments/cross_layer_v2/runtime.py')}
    write(lock,cfg);print('FROZEN',lock,flush=True);return cfg


def read_image():
    image=json.loads(subprocess.check_output(['docker','image','inspect',IMAGE]))[0]
    return {'id':image['Id'],'repo_digests':image.get('RepoDigests',[]),
            'build_commit':image['Config'].get('Labels',{}).get('org.opencontainers.image.revision'),
            'reused_from':'infer-lab experiment 5.3; no pull/install'}


def formal():
    cfg=freeze()
    for job in cfg['jobs']:
        assert all(sha(HERE/name)==expected for name,expected in cfg['source_sha256'].items())
        assert sha(HERE/'options.lock.json')==cfg['options_sha256']
        assert sha(HERE/'data/evaluation.jsonl')==cfg['evaluation_sha256']
        result=HERE/'results/formal'/f"seed-{job['seed']}-{job['variant']}"
        run(job['variant'],'formal',job['seed'],result,HERE/'options.lock.json')
    write(HERE/'results/formal/complete.json',{'complete':True,'engine_starts':len(cfg['jobs']),'scene_blocks':len(cfg['jobs'])*3,
                                              'protocol_sha256':sha(HERE/'protocol.lock.json')})
    print('FORMAL EXPERIMENT COMPLETE',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['development','freeze','formal','all'])
    args=parser.parse_args()
    if args.stage in ['development','all']:develop()
    if args.stage=='freeze':freeze()
    if args.stage in ['formal','all']:formal()


if __name__=='__main__':main()
