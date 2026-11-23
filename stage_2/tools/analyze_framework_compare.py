#!/usr/bin/env python3
"""Independently audit frozen framework trials and rebuild tables/figures/report."""
from collections import defaultdict,Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics as st
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'experiments/framework_compare'
sys.path.insert(0,str(ROOT/'src'))
from vlmbench.metrics import quality,quantiles

VARIANTS=['H0','H1','P','V00','V10','V01','V11','V11E']


def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines()]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def table(headers,values):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join('---' for _ in headers)+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in values)
def csv_write(path,data):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)


def expected_scenes(data,seed):
    grouped=defaultdict(list)
    for r in data:grouped[r['page_id']].append(r)
    pages=[p for p in grouped.values() if len(p)>=2][:16]
    hot=[r for page in pages for r in page[1:]];random.Random(seed).shuffle(hot)
    order=list(grouped);random.Random(seed).shuffle(order)
    return {'hot':hot,'session':[r for p in order for r in grouped[p]],'unique':[grouped[p][0] for p in order]}


def summarize(data):
    steps=sum(r['decode_steps'] for r in data)
    return {'questions':len(data),'anls':st.mean(r['anls'] for r in data),'em':st.mean(r['em'] for r in data),
            'mean_ttft_s':st.mean(r['ttft_s'] for r in data),'mean_e2e_s':st.mean(r['e2e_s'] for r in data),
            'total_e2e_s':sum(r['e2e_s'] for r in data),'decode_steps':steps,
            'decode_ms_per_token':1000*sum(r['decode_s'] for r in data)/steps,
            'encoder_calls':sum(r['encoder_calls'] for r in data),'encoded_images':sum(r['encoded_images'] for r in data),
            'graph_replays':sum(r['graph_replays'] for r in data),'prefix_hit_requests':sum(r['num_cached_tokens']>0 for r in data),
            'cached_tokens':sum(r['num_cached_tokens'] for r in data),'input_tokens':sum(r['input_tokens'] for r in data),
            'output_tokens':sum(len(r['token_ids']) for r in data),
            'peak_allocated_bytes':max(r['peak_allocated_bytes'] for r in data),'peak_reserved_bytes':max(r['peak_reserved_bytes'] for r in data)}


def main():
    protocol=read(HERE/'protocol.lock.json');completion=read(HERE/'results/formal/complete.json')
    assert completion['complete'] and completion['engine_starts']==24 and completion['scene_blocks']==72
    assert completion['protocol_sha256']==sha(HERE/'protocol.lock.json')
    assert all(sha(HERE/f)==h for f,h in protocol['source_sha256'].items())
    assert sha(HERE/'options.lock.json')==protocol['options_sha256']
    assert sha(HERE/'data/manifest.json')==protocol['dataset_manifest_sha256']
    assert all(sha(ROOT/f)==h for f,h in protocol['reference_source_sha256'].items())
    assert sha(ROOT/'experiments/cross_layer_v2/runtime.py')==protocol['prototype_runtime_sha256']
    assert sha(HERE/'data/evaluation.jsonl')==protocol['evaluation_sha256']
    data=rows(HERE/'data/evaluation.jsonl');references={r['question_id']:r for r in data}
    assert len(data)==269 and len(references)==269 and len({r['page_id'] for r in data})==64
    assert len({r['source_document_id'] for r in data})==47
    from PIL import Image
    dataset_manifest=read(HERE/'data/manifest.json')
    for page,item in dataset_manifest['images'].items():
        path=HERE/'data'/item['image_path'];assert sha(path)==item['file_sha256']
        with Image.open(path) as image:
            assert hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()==item['pixel_sha256']
        assert all(r['pixel_sha256']==item['pixel_sha256'] and r['image_path']==item['image_path'] for r in data if r['page_id']==page)
    old=[]
    for name in ['development','evaluation']:old.extend(rows(ROOT/f'data/processed/docvqa/{name}.jsonl'))
    assert not ({r['source_document_id'] for r in data}&{r['source_document_id'] for r in old})
    assert not ({r['pixel_sha256'] for r in data}&{r['pixel_sha256'] for r in old})
    all_rows={};blocks=[];manifest_by_job={};startup=[];backends=set()
    frozen_options=read(HERE/'options.lock.json')
    for job in protocol['jobs']:
        variant,seed=job['variant'],job['seed'];folder=HERE/'results/formal'/f'seed-{seed}-{variant}'
        assert read(folder/'complete.json')['complete']
        manifest=read(folder/'manifest.json');manifest_by_job[(variant,seed)]=manifest
        assert manifest['variant']==variant and manifest['seed']==seed and manifest['phase']=='formal'
        assert manifest['options']==dict(frozen_options,seed=seed)
        env=manifest['environment']
        assert env['threads']==2 and env['cuda']=='13.0'
        assert env['torch']==('2.13.0+cu130' if variant.startswith('V') else '2.14.0+cu130')
        assert env['transformers']==('5.19.0' if variant.startswith('V') else '5.17.0')
        if variant.startswith('V'):
            args=env['engine_args']
            assert env['engine_core_client']=='InprocClient' and env['vllm']=='0.32.0'
            assert args['enable_prefix_caching']==(variant in ['V10','V11','V11E'])
            assert args['compilation_config']['mode']==3
            assert args['compilation_config']['cudagraph_mode']==('NONE' if variant in ['V00','V10'] else 'FULL_DECODE_ONLY')
            assert args['kv_cache_memory_bytes']==frozen_options['vllm_kv_cache_bytes']
            assert env['worker'][0]['encoder_cache_tokens']==(16384 if variant=='V11E' else 1024)
            log=(folder.parent/(folder.name+'.log')).read_text()
            found=re.findall(r'Using (\w+) attention backend out of',log)
            assert found;backends.update(found)
        assert manifest['input_sha256']==protocol['evaluation_sha256']
        for f,h in protocol['source_sha256'].items():
            assert manifest['source_sha256'][f]==h
            assert sha(folder.parent/(folder.name+'-source')/f)==h
        recorded=read(folder/'summary.json')
        warmup=rows(folder/'hot-warmup.jsonl')
        grouped=defaultdict(list)
        for row in data:grouped[row['page_id']].append(row)
        expected_warmup=[page[0] for page in grouped.values() if len(page)>=2][:16]
        assert [r['question_id'] for r in warmup]==[r['question_id'] for r in expected_warmup]
        telemetry=[]
        for sample in rows(folder.parent/(folder.name+'-gpu.jsonl')):
            try:telemetry.append([float(x) for x in sample['values'].split(',')])
            except ValueError:pass
        startup.append({'variant':variant,'seed':seed,'initialization_s':manifest['initialization_s'],
                        'hot_page_warmup_s':sum(r['e2e_s'] for r in warmup),
                        'sampled_device_peak_mib':max(r[0] for r in telemetry),
                        'sampled_temperature_max_c':max(r[1] for r in telemetry),
                        'sampled_power_max_w':max(r[3] for r in telemetry)})
        for scene,members in expected_scenes(data,seed).items():
            selected=rows(folder/f'{scene}-requests.jsonl')
            assert [r['question_id'] for r in selected]==[r['question_id'] for r in members]
            for row in selected:
                reference=references[row['question_id']];score=quality(row['answer'],reference['answers'])
                assert row['input_tokens']==reference['input_tokens']
                assert row['decode_steps']==len(row['token_ids'])-1 and 0<len(row['token_ids'])<=64
                assert math.isclose(row['anls'],score['anls'],abs_tol=1e-12) and row['em']==score['em']
                assert 0<=row['ttft_s']<=row['e2e_s'] and row['decode_s']>=0
            actual=summarize(selected)
            for key,value in actual.items():assert math.isclose(value,recorded[scene][key],rel_tol=1e-10,abs_tol=1e-9),(variant,seed,scene,key)
            for metric in ['ttft_s','e2e_s']:
                for key,value in quantiles(r[metric] for r in selected).items():
                    assert math.isclose(value,recorded[scene][metric][key],abs_tol=1e-10)
            actual.update(variant=variant,seed=seed,scene=scene)
            actual['startup_s']=manifest['initialization_s']
            blocks.append(actual);all_rows[(variant,seed,scene)]=selected
    for (variant,seed,scene),selected in all_rows.items():
        baseline={r['question_id']:r for r in all_rows[('H0',seed,scene)]}
        assert all(r['prompt_sha256']==baseline[r['question_id']]['prompt_sha256'] for r in selected)
    means={}
    metrics=['mean_ttft_s','mean_e2e_s','decode_ms_per_token','encoder_calls','encoded_images','graph_replays',
             'prefix_hit_requests','cached_tokens','input_tokens','output_tokens','anls','em','peak_allocated_bytes','peak_reserved_bytes','startup_s']
    flat=[]
    for scene in ['hot','session','unique']:
        means[scene]={}
        for variant in VARIANTS:
            selected=[b for b in blocks if b['variant']==variant and b['scene']==scene]
            m={'questions':selected[0]['questions']}
            for key in metrics:m[key]=st.mean(b[key] for b in selected);m[key+'_sd']=st.stdev(b[key] for b in selected)
            means[scene][variant]=m;flat.append({'scene':scene,'variant':variant,**m})
    eval_gate={}
    for variant in VARIANTS:
        checks=[]
        for seed in [4101,4102,4103]:
            a=all_rows[('H0',seed,'session')];b=all_rows[(variant,seed,'session')]
            amap={r['question_id']:r for r in a}
            delta=st.mean(r['anls'] for r in b)-st.mean(r['anls'] for r in a)
            checks.append({'seed':seed,'anls_delta_vs_H0':delta,'passed':delta>=-0.005,
                           'identical_tokens':sum(r['token_ids']==amap[r['question_id']]['token_ids'] for r in b)})
        eval_gate[variant]={'passed':all(r['passed'] for r in checks),'seeds':checks}
    paired=[];token_pairs=[]
    for a,b in [('H0','H1'),('H1','P'),('V00','V10'),('V00','V01'),('V01','V11'),('V10','V11'),('V11','V11E'),('P','V11'),('P','V11E')]:
        for scene in means:
            diffs=[(next(x for x in blocks if x['variant']==a and x['scene']==scene and x['seed']==seed)['mean_e2e_s']-
                    next(x for x in blocks if x['variant']==b and x['scene']==scene and x['seed']==seed)['mean_e2e_s'])*1000
                   for seed in [4101,4102,4103]]
            average=st.mean(diffs);half=4.302652729911275*st.stdev(diffs)/math.sqrt(3)
            paired.append({'reference':a,'candidate':b,'scene':scene,'mean_saved_ms':average,
                           't95_low_ms':average-half,'t95_high_ms':average+half,
                           'reference_over_candidate':means[scene][a]['mean_e2e_s']/means[scene][b]['mean_e2e_s']})
            for seed in [4101,4102,4103]:
                first=all_rows[(a,seed,scene)];second=all_rows[(b,seed,scene)]
                token_pairs.append({'reference':a,'candidate':b,'scene':scene,'seed':seed,'questions':len(first),
                                    'identical_tokens':sum(x['token_ids']==y['token_ids'] for x,y in zip(first,second)),
                                    'reference_output_tokens':sum(len(x['token_ids']) for x in first),
                                    'candidate_output_tokens':sum(len(x['token_ids']) for x in second)})
    profile={}
    for variant in ['V00','V01']:
        p=HERE/'results/profile'/variant
        result=read(p/'profile.json');trace=read(p/'decode-trace.json')
        events=[e for e in trace['traceEvents'] if e.get('cat')=='cuda_runtime']
        counts=Counter(e['name'] for e in events if 'cudaGraphLaunch' in e['name'] or 'cudaLaunchKernel' in e['name'])
        assert dict(counts)==result['profile_runtime_calls'] and result['decode_steps']==8
        profile[variant]={'runtime_calls':dict(counts),'token_ids':result['token_ids'],
                          'scope':'Eight CPU-visible output intervals after the first token; framework scheduling can run ahead. Not exactly eight GPU launches.'}
    profile_tokens_identical=profile['V00']['token_ids']==profile['V01']['token_ids']
    preparation={scene:{v:st.mean(st.mean(r['preparation_s'] for r in all_rows[(v,seed,scene)]) for seed in [4101,4102,4103])
                        for v in ['H0','H1','P']} for scene in means}
    preparation_blocks=[{'variant':v,'seed':seed,'scene':scene,
                         'mean_preparation_s':st.mean(r['preparation_s'] for r in all_rows[(v,seed,scene)])}
                        for scene in means for v in ['H0','H1','P'] for seed in [4101,4102,4103]]
    output=HERE/'results/analysis';output.mkdir(exist_ok=True)
    csv_write(output/'blocks.csv',blocks);csv_write(output/'means.csv',flat);csv_write(output/'paired.csv',paired)
    csv_write(output/'startup-resources.csv',startup)
    csv_write(output/'output-pairs.csv',token_pairs)
    csv_write(output/'input-preparation.csv',preparation_blocks)
    quality_passed=all(c['passed'] for c in eval_gate.values())
    audit={'passed':quality_passed,'integrity_passed':True,'formal_quality_passed':quality_passed,
           'trials':24,'main_trials':21,'encoder_capacity_trials':3,'scene_blocks':72,
           'requests':sum(len(v) for v in all_rows.values()),'pages':len({r['page_id'] for r in data}),
           'questions':len(data),'old_source_overlap':0,'old_rgb_overlap':0,'expanded_input_ids_identical':True,
           'formal_quality_gate':eval_gate,'means':means,'paired':paired,'decode_profiles':profile,
           'profile_tokens_identical':profile_tokens_identical,'protocol_sha256':sha(HERE/'protocol.lock.json'),
           'source_sha256':protocol['source_sha256'],'startup_resources':startup,'paired_token_identity':token_pairs,
           'vllm_attention_backends':sorted(backends),'hf_mean_input_preparation_s':preparation,
           'hf_input_preparation_blocks':preparation_blocks,'analysis_script_sha256':sha(Path(__file__))}
    assert backends=={'FLASH_ATTN'},backends
    write(output/'analysis.json',audit)
    # Scientific figures use the same palette and Chinese typography as the design document.
    font=Path('/mnt/c/Windows/Fonts/msyh.ttc')
    if font.exists():font_manager.fontManager.addfont(str(font));family=font_manager.FontProperties(fname=str(font)).get_name()
    else:family='Noto Sans CJK SC'
    plt.rcParams.update({'font.family':family,'font.size':10,'axes.unicode_minus':False,'svg.fonttype':'path',
                         'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    figures=HERE/'figures';figures.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(7.4,2.9));width=.24
    for offset,(scene,label,color) in zip([-width,0,width],[('hot','热页面多问','#346b9a'),('session','含首次miss的会话','#28857d'),('unique','每页一次','#b18b59')]):
        values=[means[scene][v]['mean_e2e_s']*1000 for v in VARIANTS]
        errors=[means[scene][v]['mean_e2e_s_sd']*1000 for v in VARIANTS]
        ax.bar([i+offset for i in range(len(VARIANTS))],values,width,yerr=errors,capsize=2,color=color,label=label)
    ax.set_xticks(range(len(VARIANTS)),VARIANTS);ax.set_ylabel('完整请求耗时 / ms');ax.set_ylim(bottom=0)
    ax.legend(frameon=False,ncol=3,fontsize=8,loc='upper left',bbox_to_anchor=(0,1.16));fig.tight_layout()
    for ext in ['png','svg','pdf']:fig.savefig(figures/f'framework-latency.{ext}',dpi=240,bbox_inches='tight')
    plt.close(fig)
    report=table(['实现','热TTFT/ms','热完整/ms','会话完整/ms','每页一次/ms','评估ANLS'],[
        [v,f"{means['hot'][v]['mean_ttft_s']*1000:.1f} ± {means['hot'][v]['mean_ttft_s_sd']*1000:.1f}",
         f"{means['hot'][v]['mean_e2e_s']*1000:.1f} ± {means['hot'][v]['mean_e2e_s_sd']*1000:.1f}",
         f"{means['session'][v]['mean_e2e_s']*1000:.1f} ± {means['session'][v]['mean_e2e_s_sd']*1000:.1f}",
         f"{means['unique'][v]['mean_e2e_s']*1000:.1f} ± {means['unique'][v]['mean_e2e_s_sd']*1000:.1f}",
         f"{means['session'][v]['anls']:.6f}"] for v in VARIANTS])
    mechanisms=table(['实现','热encoder执行','热前缀命中请求','热命中token','热decode/ms每token','峰值allocated/GiB'],[
        [v,f"{means['hot'][v]['encoder_calls']:.0f}",f"{means['hot'][v]['prefix_hit_requests']:.0f}/{means['hot'][v]['questions']}",
         f"{means['hot'][v]['cached_tokens']:.0f}",f"{means['hot'][v]['decode_ms_per_token']:.2f}",
         f"{means['session'][v]['peak_allocated_bytes']/1024**3:.3f}"] for v in VARIANTS])
    hot=means['hot'];session=means['session'];unique=means['unique']
    def milliseconds(group,v,key='mean_e2e_s'):return group[v][key]*1000
    def saved(group,a,b,key='mean_e2e_s'):return (1-group[b][key]/group[a][key])*100
    faster='P' if hot['P']['mean_e2e_s']<hot['V11E']['mean_e2e_s'] else 'V11E'
    capacity_tokens=[r for r in token_pairs if r['reference']=='V11' and r['candidate']=='V11E' and r['scene']=='hot']
    same_capacity_tokens=sum(r['identical_tokens'] for r in capacity_tokens)
    capacity_total=sum(r['questions'] for r in capacity_tokens)
    prep_p=[r['mean_preparation_s']*1000 for r in preparation_blocks if r['variant']=='P' and r['scene']=='hot']
    decode_p=[b['decode_ms_per_token'] for b in blocks if b['variant']=='P' and b['scene']=='hot']
    conclusions=f'''## 结果分析

在统一CPU输入就绪边界下，完整组合P相对已有视觉缓存H1的热请求耗时比为{hot['H1']['mean_e2e_s']/hot['P']['mean_e2e_s']:.2f}，从{milliseconds(hot,'H1'):.1f}降至{milliseconds(hot,'P'):.1f} ms，减少{saved(hot,'H1','P'):.1f}%。包含首次miss的会话中，两者为{milliseconds(session,'H1'):.1f}与{milliseconds(session,'P'):.1f} ms。它们与旧开发集GPU阶段计时分别统计。

框架内部，V00到V10的热TTFT从{milliseconds(hot,'V00','mean_ttft_s'):.1f}变为{milliseconds(hot,'V10','mean_ttft_s'):.1f} ms；V00到V01的热decode从{hot['V00']['decode_ms_per_token']:.2f}变为{hot['V01']['decode_ms_per_token']:.2f} ms/token。APC与图模式分别关联输入状态复用和后续执行成本，组合收益以V11的实际完整请求为准，不把两项百分比相加。

V11与扩容辅助V11E的热请求分别为{milliseconds(hot,'V11'):.1f}和{milliseconds(hot,'V11E'):.1f} ms，encoder执行从{hot['V11']['encoder_calls']:.1f}变为{hot['V11E']['encoder_calls']:.1f}次；这两个配置在热集合的输出token有{same_capacity_tokens}/{capacity_total}条逐序列一致。容量对照将视觉结果驻留与APC/图模式区分开，避免用一个encoder容量较小的框架配置代表全部框架能力。

访问局部性也能从调用数观察：V00在随机交错的热集合上平均编码{hot['V00']['encoder_calls']:.1f}次/{hot['V00']['questions']}问，在按页连续的完整会话中编码{session['V00']['encoder_calls']:.1f}次/{session['V00']['questions']}问。热集合已预热不等于所有页面状态仍驻留；问题输出长度也不同，所以场景名字不能代替实际命中与计算量记录。

热请求中P与V11E分别为{milliseconds(hot,'P'):.1f}和{milliseconds(hot,'V11E'):.1f} ms，均值较低者为{faster}；含miss会话分别为{milliseconds(session,'P'):.1f}和{milliseconds(session,'V11E'):.1f} ms，每页一次分别为{milliseconds(unique,'P'):.1f}和{milliseconds(unique,'V11E'):.1f} ms。这里比较的是本机指定负载和各自软件栈，解释时同时使用缓存计数、实际显存与输出长度。

P的热请求输入准备平均为{preparation['hot']['P']*1000:.1f} ms，已经计入完整延迟。本原型没有跨请求CPU processor缓存，框架配置了0.125GiB缓存并使用GPU图像归一化，因此P与vLLM的差异还包含输入处理路径；不能全部归于语言层或CUDA图。框架V00的解码成本也保留在表中，展示其在APC/图开关之外已有的执行能力。

三轮P热请求的输入准备为{min(prep_p):.1f}—{max(prep_p):.1f} ms，decode为{min(decode_p):.2f}—{max(decode_p):.2f} ms/token。较慢区组的额外时间主要出现在输入准备和首token之前，全部原始区组均保留，未删除或替换。GPU温度/功耗采样用于描述运行背景，现有记录不足以把波动单独归因于热降频或某个主机进程。

'''
    text=f'''# 推理框架对照：独立评估结果

完成24次独立启动（21次主对照、3次encoder容量辅助），三个种子、三个访问场景，共72个区组、{audit['requests']}条计时请求。评估集为新选出的64页{len(data)}问，源文档和RGB图片均与旧开发/评估集合互斥，保留所选页面的全部真实问题。

## 实验范围与配置

复用infer-lab已有vLLM 0.32.0 Docker镜像，固定image ID为`{protocol['image']['id']}`，不重新下载镜像或模型。容器Torch 2.13.0+cu130、Transformers 5.19.0；HF/本项目使用既有Torch 2.14.0+cu130、Transformers 5.17.0。比较的是各自软件栈的完整请求路径，框架内部消融使用同一容器环境。

H0为HF原生，H1为HF加视觉缓存，P为本项目完整组合；V00/V10/V01/V11分别为APC/图的关关、开关、关开、开开。vLLM固定编译mode=3、FLASH_ATTN后端、图模式FULL_DECODE_ONLY或NONE、max_num_seqs=1、max_model_len=1024、生成上限64、greedy和真实EOS。CPU processor缓存0.125GiB，KV池1648MiB；本实验max_num_batched_tokens=1024时派生的encoder缓存为1024 token，非所有vLLM配置的通用默认值；V11E仅将encoder缓存扩为16384 token，单步encoder计算预算仍1024。它是额外容量控制，框架源码未改。

全GPU BF16、BF16 KV、相同固定revision和原始RGB图片，不使用量化或推测解码；展开后的全部prompt token IDs已逐请求核对一致。框架将图像归一化移至GPU，原生路径在CPU处理，视觉入口中间表示因此不同；答案质量单独检查，不能声称所有中间张量逐位相同。

时间从驻留CPU图片和提示就绪开始，到首个CPU可见token/最终答案结束，包含各自预处理、传输和执行。原图文件读取、图片驻留准备、模型加载和建图单独处理，主计时不含HTTP。不同于前一轮从GPU输入就绪开始的开发计时，两轮数字不直接混作同一加速比。

## 真实请求结果

下表为三个独立进程种子的均值±样本标准差。热集合有{means['hot']['H0']['questions']}问，含miss会话覆盖全部{len(data)}问，每页一次为64问；每个场景内所有实现的问题成员与顺序一致。

场景之间的问题成员和输出长度不同，复用效果由同一场景内的配对配置和encoder调用数判断；不将热、冷两个均值的差直接当作缓存收益。实际输出token总数、按总decode步数加权的ms/token和配置间逐条token身份均保存在CSV中。

{report}

![框架请求耗时](figures/framework-latency.png)

{conclusions}

## 机制和内存证据

{mechanisms}

持久视觉负载、前缀KV和框架KV池不是同一种内存表示。本项目1536MiB为持久负载，另有112MiB工作KV；框架1648MiB KV池同时供活跃状态和前缀使用，另有encoder缓存与图池。V11E增加了encoder驻留容量，各实现运行在同一8188MiB设备上，并记录实际allocated、reserved和每2秒驱动采样；不称为所有缓存表示逐字节等容量。

初始化时间、热场景16页的预热请求费用和驱动采样峰值另见`results/analysis/startup-resources.csv`。初始化不包括随后首次请求可能触发的懒编译；驱动数据为2秒采样，不能当作无遗漏的瞬时峰值。HF适配器沿用已有运行时，H0/H1也保留一个未用于生成的112MiB静态工作区，内存结果包含这部分测量框架成本。

Python重放探针只覆盖动态方法分发，可能漏掉初始化时已保存的绑定方法，也可能计入框架提前执行后丢弃的步骤，因此不用于声称全部CUDA发射次数。诊断在首个CPU可见token后启动CUDA profiler，固定检查其后八个输出间隔。框架调度可能提前执行下一步，这个CPU观察窗口不保证恰好覆盖八次GPU模型调用，不能将观测到的7次图发射直接解释为八步中有一步回退。原始trace及独立重数结果如下；这些诊断时间不纳入主表。

```json
{json.dumps(profile,ensure_ascii=False,indent=2)}
```

## 验证与统计

评估质量门槛沿用相对共同H0的ANLS下降不超过0.005，全部配置通过状态为`{quality_passed}`；逐种子结果见`results/analysis/analysis.json`，原始答案和token全部保留。开发选择只使用原282问，配置、实现哈希、问题集合与重复顺序在新评估输出产生前写入`protocol.lock.json`。

审计逐条重算ANLS/EM、时间汇总、请求集合、输入ID身份、源码快照和CUDA trace计数。配对t95区间基于三个独立种子、df=2，分别保存在`paired.csv`，不能把单次请求当作独立设备重复来缩小误差。

三个场景之间显式清空prefix、processor和encoder缓存并保留预热记录；冷复用对照表示页面缓存未命中，不是每条请求都重新加载权重。图模式对整体执行的影响由实际trace观察，不能仅凭枚举名称把全部变化归因于某个kernel。

## 原始数据与复现

代码、冻结协议、开发门槛、每次独立启动的源码快照、所有warmup和逐请求JSONL、GPU遥测、启动日志及两份CUDA trace均位于本目录。逐区组/均值/配对CSV位于`results/analysis/`。一键与逐步命令见`README.md`，已有结果不会覆盖。

AI用于适配器、分析脚本和文档辅助，答案与性能记录均由固定本地模型在真实GPU上产生。状态隔离的运行时不变量检查沿用第二轮验证；本次框架扩展不新增形式化证明。
'''
    (HERE/'框架对比实验报告.md').write_text(text)
    from markdown_it import MarkdownIt
    import base64
    body=MarkdownIt('commonmark',{'html':True}).enable('table').render(text)
    encoded=base64.b64encode((figures/'framework-latency.png').read_bytes()).decode()
    body=body.replace('src="figures/framework-latency.png"','src="data:image/png;base64,'+encoded+'"')
    style="body{max-width:1050px;margin:36px auto;padding:0 24px;font:16px/1.8 'Microsoft YaHei',sans-serif;color:#26374b}h1{font-size:27px}h2{font-size:21px;margin-top:28px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:7px;border-bottom:1px solid #d7dde3;text-align:left}th{background:#edf3f9}img{max-width:100%}pre{white-space:pre-wrap;background:#f2f5f7;padding:16px}a{color:#346b9a}@media print{body{font-size:10pt;margin:0}table{font-size:8pt}img,table{break-inside:avoid}}"
    (HERE/'框架对比实验报告.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>框架对比实验报告</title><style>'+style+'</style><body>'+body+'</body></html>')
    print(json.dumps({k:audit[k] for k in ['passed','trials','scene_blocks','requests','pages','questions']},ensure_ascii=False))


if __name__=='__main__':main()
