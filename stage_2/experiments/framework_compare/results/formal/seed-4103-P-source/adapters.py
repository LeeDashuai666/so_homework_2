"""Inference adapters. The historical runtime and its measured source are unchanged."""
import hashlib
import json
from pathlib import Path
import sys
import time

from common import HERE,ROOT


class PhaseClock:
    """Observe the old runtime's start timestamp without changing its implementation."""
    def __init__(self):self.active=False;self.first=None
    def perf_counter(self):
        now=time.perf_counter()
        if self.active and self.first is None:self.first=now
        return now


class HFAdapter:
    def __init__(self,inputs,variant,options):
        import torch
        sys.path.insert(0,str(ROOT/'experiments/cross_layer_v2'))
        import runtime as prototype
        self.inputs=inputs;self.variant=variant;self.torch=torch
        self.clock=PhaseClock();prototype.time=self.clock
        # All images were decoded and verified before timing, in every adapter.
        prototype.prepare_inputs=inputs.prepare_cpu
        self.engine=prototype.Runtime(gpu_cache_mib=options['persistent_cache_mib'])
        self.engine.cfg=inputs.cfg
        self.encoder_calls=0;self.encoded_images=0
        original=self.engine.native_visual
        def encode(*args,**kwargs):
            self.encoder_calls+=1
            grid=kwargs.get('image_grid_thw',args[1] if len(args)>1 else None)
            self.encoded_images+=len(grid) if grid is not None else 1
            return original(*args,**kwargs)
        self.engine.native_visual=encode;self.engine.core.get_image_features=encode
        self.graph_build_s=0
        if variant=='P':
            self.engine.build_graph(self.engine.prepare(inputs.rows[0]));self.graph_build_s=self.engine.graph_build_s
        self.clear()
        self.environment={'torch':torch.__version__,'transformers':__import__('transformers').__version__,
                          'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'threads':torch.get_num_threads(),
                          'graph_build_s':self.graph_build_s,'runtime_sha256':hashlib.sha256((ROOT/'experiments/cross_layer_v2/runtime.py').read_bytes()).hexdigest()}

    def clear(self):
        self.engine.store.clear();self.engine.work.reset();self.torch.cuda.empty_cache()
        self.torch.cuda.reset_peak_memory_stats();self.encoder_calls=0;self.encoded_images=0

    def predict(self,row):
        self.torch.cuda.synchronize()
        calls,images=self.encoder_calls,self.encoded_images
        started=time.perf_counter();request=self.engine.prepare(row)
        self.clock.first=None;self.clock.active=True
        try:
            result=self.engine.generate(request,prefix=True,graph=True) if self.variant=='P' else self.engine.native(request,visual_cache=self.variant=='H1')
        finally:self.clock.active=False
        ended=time.perf_counter()
        result.update(ttft_s=self.clock.first+result['ttft_s']-started,e2e_s=ended-started,
                      preparation_s=self.clock.first-started,encoder_calls=self.encoder_calls-calls,
                      encoded_images=self.encoded_images-images,input_tokens=request.inputs['input_ids'].shape[1],
                      prompt_sha256=hashlib.sha256(request.inputs['input_ids'].cpu().numpy().tobytes()).hexdigest(),
                      num_cached_tokens=request.boundary if result.get('prefix_hit') else 0,
                      graph_replays=result['decode_steps'] if self.variant=='P' else 0,
                      peak_reserved_bytes=self.torch.cuda.max_memory_reserved())
        return result

    def info(self):return self.environment
    def diagnostics(self):return {'encoder_calls':self.encoder_calls,'encoded_images':self.encoded_images,
                                  'peak_allocated_bytes':self.torch.cuda.max_memory_allocated(),
                                  'peak_reserved_bytes':self.torch.cuda.max_memory_reserved()}


def install_worker_probe(worker,capture_pixels=False,normalization=None):
    import torch
    runner=worker.model_runner
    model=runner.get_model() if hasattr(runner,'get_model') else runner.model
    probe={'encoder_calls':0,'encoded_images':0,'graph_replays':0,'pixel_inputs':[],'graph_callers':{}}
    original=model.visual.forward
    def encode(*args,**kwargs):
        grid=kwargs.get('grid_thw',args[1] if len(args)>1 else None)
        probe['encoder_calls']+=1;probe['encoded_images']+=len(grid) if grid is not None else 1
        if capture_pixels:
            raw=args[0].detach().cpu().contiguous();pixels=raw.to(torch.bfloat16)
            item={'grid_thw':grid.cpu().tolist(),'dtype':str(raw.dtype),'shape':list(raw.shape),
                  'min':float(raw.min()),'max':float(raw.max()),
                  'pixel_bf16_sha256':hashlib.sha256(pixels.view(torch.uint8).numpy().tobytes()).hexdigest()}
            if raw.dtype==torch.uint8 and normalization:
                channels=raw.float().reshape(-1,3,normalization['temporal_patch_size'],normalization['patch_size'],normalization['patch_size'])
                mean=torch.tensor(normalization['mean']).view(1,3,1,1,1)
                std=torch.tensor(normalization['std']).view(1,3,1,1,1)
                norm=((channels*normalization['rescale_factor']-mean)/std).reshape(raw.shape).to(torch.bfloat16)
                item['normalized_pixel_bf16_sha256']=hashlib.sha256(norm.view(torch.uint8).numpy().tobytes()).hexdigest()
            probe['pixel_inputs'].append(item)
        return original(*args,**kwargs)
    model.visual.forward=encode
    real_replay=torch.cuda.CUDAGraph.replay
    def replay(graph,*args,**kwargs):
        probe['graph_replays']+=1
        caller=sys._getframe(1).f_code
        key=Path(caller.co_filename).name+':'+caller.co_name
        probe['graph_callers'][key]=probe['graph_callers'].get(key,0)+1
        return real_replay(graph,*args,**kwargs)
    torch.cuda.CUDAGraph.replay=replay
    worker._stage2_probe=probe
    return {'model':type(model).__name__,'runner':type(runner).__name__,
            'encoder_cache_tokens':worker.vllm_config.scheduler_config.encoder_cache_size}


def get_worker_probe(worker,reset=False):
    import torch
    result=dict(worker._stage2_probe)
    result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    if reset:
        for key in ['encoder_calls','encoded_images','graph_replays']:worker._stage2_probe[key]=0
        worker._stage2_probe['pixel_inputs']=[]
        worker._stage2_probe['graph_callers']={}
        torch.cuda.reset_peak_memory_stats()
    return result


class WorkerProbe:
    """Named worker-extension RPCs; no serialization of executable callables."""
    def stage2_install(self,capture_pixels=False,normalization=None):return install_worker_probe(self,capture_pixels,normalization)
    def stage2_diagnostics(self,reset=False):return get_worker_probe(self,reset)
    def stage2_profile_start(self):
        import torch
        self._stage2_profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA])
        self._stage2_profiler.__enter__()
        return True
    def stage2_profile_stop(self,path):
        self._stage2_profiler.__exit__(None,None,None)
        self._stage2_profiler.export_chrome_trace(path)
        calls={}
        for event in self._stage2_profiler.events():
            if 'cudaGraphLaunch' in event.name or 'cudaLaunchKernel' in event.name:
                calls[event.name]=calls.get(event.name,0)+1
        return calls


class VLLMAdapter:
    def __init__(self,inputs,variant,options,capture_pixels=False):
        import torch,transformers,vllm
        from vllm import LLMEngine,EngineArgs,SamplingParams
        self.inputs=inputs;self.variant=variant;self.index=0;self.torch=torch
        graph=variant in ('V01','V11','V11E');prefix=variant in ('V10','V11','V11E')
        params=dict(model=str(ROOT/'models/qwen3-vl-2b-instruct'),dtype='bfloat16',
                    max_model_len=1024,max_num_seqs=1,tensor_parallel_size=1,max_num_batched_tokens=1024,
                    enable_prefix_caching=prefix,enable_chunked_prefill=False,
                    kv_cache_memory_bytes=options['vllm_kv_cache_bytes'],gpu_memory_utilization=options['gpu_memory_utilization'],
                    mm_processor_cache_gb=options['processor_cache_gib'],
                    mm_device_do_normalize=True,
                    mm_processor_kwargs={'size':{'shortest_edge':131072,'longest_edge':786432}},
                    limit_mm_per_prompt={'image':1,'video':0},
                    compilation_config={'mode':options['compilation_mode'],'cudagraph_mode':options['graph_mode'] if graph else 'NONE',
                                        'cudagraph_capture_sizes':[1],'max_cudagraph_capture_size':1},
                    seed=options.get('seed',4101),disable_log_stats=False,generation_config='vllm',
                    worker_extension_cls='adapters.WorkerProbe')
        started=time.perf_counter()
        if variant=='V11E':
            class CapacityArgs(EngineArgs):
                def create_engine_config(self,*args,**kwargs):
                    config=super().create_engine_config(*args,**kwargs)
                    config.scheduler_config.encoder_cache_size=options['expanded_encoder_cache_tokens']
                    return config
            engine_args=CapacityArgs(**params)
        else:engine_args=EngineArgs(**params)
        self.engine=LLMEngine.from_engine_args(engine_args,enable_multiprocessing=False)
        self.startup_s=time.perf_counter()-started
        torch.set_num_threads(2)
        self.sampling=SamplingParams(temperature=0,top_p=1,top_k=-1,repetition_penalty=1,
                                   max_tokens=64,stop_token_ids=inputs.eos,ignore_eos=False,skip_special_tokens=True)
        proc=inputs.processor.image_processor;vision=json.loads((ROOT/'models/qwen3-vl-2b-instruct/config.json').read_text())['vision_config']
        normalization={'mean':proc.image_mean,'std':proc.image_std,'rescale_factor':proc.rescale_factor,
                       'patch_size':vision['patch_size'],'temporal_patch_size':vision['temporal_patch_size']}
        self.worker_info=self.engine.collective_rpc('stage2_install',args=(capture_pixels,normalization))
        self.options=params
        self.environment={'vllm':vllm.__version__,'torch':torch.__version__,'transformers':transformers.__version__,
                          'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'threads':torch.get_num_threads(),
                          'startup_s':self.startup_s,'engine_args':params,'worker':self.worker_info,
                          'resolved_compilation':str(self.engine.vllm_config.compilation_config),
                          'resolved_encoder_cache_tokens':self.engine.vllm_config.scheduler_config.encoder_cache_size,
                          'normalization':normalization,
                          'engine_core_client':type(self.engine.engine_core).__name__,
                          'python_graph_probe_scope':'Dynamic replay dispatch only; CUDA profiler provides authoritative runtime launch counts',
                          'resolved_cache':str(self.engine.vllm_config.cache_config)}

    def clear(self):
        assert self.engine.reset_prefix_cache()
        self.engine.reset_mm_cache();self.engine.reset_encoder_cache()
        self.engine.collective_rpc('stage2_diagnostics',args=(True,))

    def predict(self,row,profile_path=None,fixed_new_tokens=None):
        self.index+=1;rid=f'q-{self.index}-{row["question_id"]}'
        before=self.engine.collective_rpc('stage2_diagnostics')[0]
        started=time.perf_counter()
        prompt={'prompt':self.inputs.prompt(row),'multi_modal_data':{'image':self.inputs.images[row['pixel_sha256']]},
                'multi_modal_uuids':{'image':row['pixel_sha256']}}
        sampling=self.sampling
        if fixed_new_tokens:
            from vllm import SamplingParams
            sampling=SamplingParams(temperature=0,max_tokens=fixed_new_tokens,ignore_eos=True,skip_special_tokens=True)
        self.engine.add_request(rid,prompt,sampling)
        first=None;last=None
        while self.engine.has_unfinished_requests():
            for output in self.engine.step():
                if output.request_id!=rid:continue
                if output.outputs and output.outputs[0].token_ids and first is None:
                    first=time.perf_counter()
                    if profile_path:self.engine.collective_rpc('stage2_profile_start')
                if output.finished:last=output
        ended=time.perf_counter()
        assert last is not None and first is not None
        profile_counts=self.engine.collective_rpc('stage2_profile_stop',args=(str(profile_path),))[0] if profile_path else None
        after=self.engine.collective_rpc('stage2_diagnostics')[0]
        answer=last.outputs[0];tokens=list(answer.token_ids)
        ids=self.torch.tensor(last.prompt_token_ids,dtype=self.torch.int64)
        result={'question_id':row['question_id'],'answer':answer.text,'token_ids':tokens,
                'ttft_s':first-started,'e2e_s':ended-started,'decode_s':ended-first,'decode_steps':len(tokens)-1,
                'input_tokens':len(last.prompt_token_ids),'prompt_sha256':hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
                'num_cached_tokens':last.num_cached_tokens or 0,'finish_reason':answer.finish_reason,
                **{key:after[key]-before[key] for key in ['encoder_calls','encoded_images','graph_replays']},
                'peak_allocated_bytes':after['peak_allocated_bytes'],'peak_reserved_bytes':after['peak_reserved_bytes']}
        if profile_counts is not None:result['profile_runtime_calls']=profile_counts
        return result

    def info(self):return self.environment
    def diagnostics(self):return self.engine.collective_rpc('stage2_diagnostics')[0]
