#!/usr/bin/env python3
"""Launch one host or existing-Docker inference job with persistent logs and GPU telemetry."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
import hashlib

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
IMAGE='sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34'


def command(variant,phase,seed,output,options):
    args=[str(HERE/'run_engine.py'),'--variant',variant,'--phase',phase,'--seed',str(seed),
          '--options',str(options),'--output',str(output)]
    if variant in ['H0','H1','P']:return [str(ROOT/'.venv/bin/python'),'-u',*args]
    name='so-stage2-vqa-'+hashlib.sha256(str(output).encode()).hexdigest()[:10]
    result=['docker','run','--rm','--name',name,'--gpus','all','--ipc=host','--network=none',
            '--user',f'{os.getuid()}:{os.getgid()}','--entrypoint','python3','--workdir',str(HERE)]
    mounts=[(ROOT.parent,ROOT.parent,'ro'),(HERE,HERE,'rw'),
            (Path('/home/lxy18/.cache/huggingface'),Path('/home/lxy18/.cache/huggingface'),'ro')]
    for a,b,mode in mounts:result+=['-v',f'{a}:{b}:{mode}']
    env={'HF_HOME':'/home/lxy18/.cache/huggingface','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
         'VLLM_NO_USAGE_STATS':'1','DO_NOT_TRACK':'1','TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'2',
         'MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','TORCHINDUCTOR_COMPILE_THREADS':'1','MAX_JOBS':'1',
         'VLLM_CACHE_ROOT':str(HERE/'cache/vllm'),'TORCHINDUCTOR_CACHE_DIR':str(HERE/'cache/inductor'),
         'TRITON_CACHE_DIR':str(HERE/'cache/triton'),'CUDA_CACHE_PATH':str(HERE/'cache/cuda'),
         'XDG_CACHE_HOME':str(HERE/'cache/xdg'),'PYTHONPATH':str(HERE)}
    for k,v in env.items():result+=['-e',k+'='+v]
    return [*result,IMAGE,'-u',*args]


def run(variant,phase,seed,output,options):
    output=Path(output).resolve();options=Path(options).resolve()
    if (output/'complete.json').exists():return {'reused':True,'output':str(output)}
    if output.exists() and any(output.iterdir()):raise RuntimeError('Nonempty incomplete directory preserved: '+str(output))
    output.parent.mkdir(parents=True,exist_ok=True)
    cmd=command(variant,phase,seed,output,options)
    launch=output.parent/(output.name+'-launch.json')
    launch.write_text(json.dumps({'argv':cmd,'started_at':datetime.now(timezone.utc).isoformat()},indent=2)+'\n')
    print('START',variant,phase,seed,output,flush=True)
    log=output.parent/(output.name+'.log');telemetry=output.parent/(output.name+'-gpu.jsonl')
    started=time.perf_counter()
    with log.open('w') as stream,telemetry.open('w') as monitor:
        proc=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,env=os.environ.copy())
        while proc.poll() is None:
            q=subprocess.run(['nvidia-smi','--query-gpu=memory.used,temperature.gpu,utilization.gpu,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True)
            monitor.write(json.dumps({'at':datetime.now(timezone.utc).isoformat(),'values':q.stdout.strip()})+'\n');monitor.flush()
            time.sleep(2)
    result={'variant':variant,'phase':phase,'seed':seed,'returncode':proc.returncode,
            'elapsed_s':time.perf_counter()-started,'output':str(output),'log':str(log)}
    launch.write_text(json.dumps({'argv':cmd,**result},indent=2)+'\n')
    print('FINISH',json.dumps(result),flush=True)
    if proc.returncode:raise RuntimeError('Inference job failed; inspect '+str(log))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',required=True)
    parser.add_argument('--phase',default='preflight')
    parser.add_argument('--seed',type=int,default=4101)
    parser.add_argument('--options',type=Path,default=HERE/'options-candidate.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.variant,args.phase,args.seed,args.output,args.options)


if __name__=='__main__':main()
