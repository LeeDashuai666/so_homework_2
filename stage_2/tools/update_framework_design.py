#!/usr/bin/env python3
"""Insert the audited independent evaluation into the design-document source."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics as st

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'experiments/framework_compare'
VARIANTS=['H0','H1','P','V00','V10','V01','V11','V11E']


def main():
    source=HERE/'results/analysis/analysis.json'
    audit=json.loads(source.read_text());assert audit['integrity_passed']
    means=audit['means'];hot=means['hot'];session=means['session'];unique=means['unique']
    def ms(group,v,key='mean_e2e_s'):return f"{group[v][key]*1000:.1f}"
    def cell(group,v,key='mean_e2e_s'):
        return f"{group[v][key]*1000:.1f} ± {group[v][key+'_sd']*1000:.1f}"
    records=[]
    for v in VARIANTS:
        records.append([v,cell(hot,v,'mean_ttft_s'),cell(hot,v),cell(session,v),cell(unique,v),f"{session[v]['anls']:.6f}"])
    table='| 实现 | 热TTFT/ms | 热请求/ms | 会话请求/ms | 每页一次/ms | ANLS |\n'
    table+='| --- | --- | --- | --- | --- | --- |\n'
    table+='\n'.join('| '+' | '.join(row)+' |' for row in records)
    profiles=audit['decode_profiles']
    def calls(v):return '、'.join(f"{n}次{name}" for name,n in profiles[v]['runtime_calls'].items()) or '0次CUDA发射'
    quality_text='所有配置在三个种子中均通过ANLS下降不超过0.005的门槛' if audit['formal_quality_passed'] else '部分配置未通过ANLS下降不超过0.005的门槛，具体逐种子结果见实验报告，不将这些配置作为合格加速结果'
    resources=audit['startup_resources']
    def resource(v,key):return st.mean(r[key] for r in resources if r['variant']==v)
    delta=min(r['anls_delta_vs_H0'] for v in audit['formal_quality_gate'].values() for r in v['seeds'])
    prep_p=[r['mean_preparation_s']*1000 for r in audit['hf_input_preparation_blocks'] if r['variant']=='P' and r['scene']=='hot']
    new=f'''### 4.7 与vLLM的独立评估对照

框架对照复用infer-lab已有的vLLM 0.32.0 Docker镜像，使用固定image ID，不重新安装推理依赖或下载模型。容器采用Torch 2.13.0+cu130、Transformers 5.19.0，HF与本项目沿用4.1节环境。模型revision、BF16、原始RGB图片、聊天模板、视觉预算、greedy、64-token上限与真实EOS保持一致。全部展开后的prompt token IDs逐请求核对一致；vLLM使用GPU图像归一化，与HF的CPU中间表示不同，质量由真实答案单独检验。

新评估集为64页269问、47个源文档，与旧开发集和旧评估集的源文档、RGB均无交集，保留选中页面的全部真实问题。先在原282问上完成开发验收，再冻结配置、源码哈希和新评估顺序。每种实现使用4101—4103三个独立进程种子，并在每个种子内随机化实现顺序；共21次主启动、3次容量辅助启动、72个场景区组、{audit['requests']}条计时记录。

H0为HF原生，H1为HF加完整视觉缓存，P为本项目三项组合。vLLM的V00、V10、V01、V11依次对应APC与解码图模式的关关、开关、关开、开开，编译mode=3保持一致，图模式为NONE或FULL_DECODE_ONLY，使用InprocClient、B=1和1024-token上下文。本实验max_num_batched_tokens=1024时，框架派生的encoder缓存容量为1024 token；额外V11E仅将驻留容量扩为16384 token，单步encoder计算预算仍为1024，用来检查跨页面驻留容量的影响。

热场景用固定16页的首问预热，测量其余全部{hot['H0']['questions']}问；会话场景清空页面缓存后测量全部269问，保留首次miss；每页一次场景清空缓存后测量64个首问。场景之间重置encoder、processor和prefix缓存。计时从驻留CPU图片和提示就绪开始，覆盖预处理、传输、首token和答案完成；文件读取、模型初始化和建图单列，不包含HTTP。它比表3的GPU输入就绪范围更完整，表6内的加速比均重新配对计算。

表6 独立评估的完整请求结果，三进程均值±标准差

<!-- widths: 1.5, 3.0, 3.0, 3.1, 3.1, 2.3 -->
{table}

![图7 三种访问场景下的跨框架完整请求耗时](assets/design/frameworks.png)

热页面上，H1到P的完整请求由{ms(hot,'H1')} ms变为{ms(hot,'P')} ms，耗时比为{hot['H1']['mean_e2e_s']/hot['P']['mean_e2e_s']:.2f}；V00到V01由{ms(hot,'V00')} ms变为{ms(hot,'V01')} ms，反映在相同框架软件栈内改变图执行模式的效果。P、V11和V11E在热场景分别为{ms(hot,'P')}、{ms(hot,'V11')}、{ms(hot,'V11E')} ms，在含miss会话中分别为{ms(session,'P')}、{ms(session,'V11')}、{ms(session,'V11E')} ms。本轮vLLM组合的平均请求耗时低于原型。P的热请求输入准备约为{audit['hf_mean_input_preparation_s']['hot']['P']*1000:.1f} ms；框架另有CPU processor缓存，因此跨框架差异也包含输入处理路径，不能全部归为单一kernel。

三轮P热请求的输入准备范围为{min(prep_p):.1f}—{max(prep_p):.1f} ms，较慢区组的额外时间主要出现在输入准备与首token之前。所有区组均保留在均值和标准差中；GPU采样用于描述设备状态，不据此将波动单独归因于某一种降频原因。

encoder计数说明容量也是比较的一部分：热场景{hot['H0']['questions']}问中，V11平均执行{hot['V11']['encoder_calls']:.1f}次视觉编码，V11E执行{hot['V11E']['encoder_calls']:.1f}次，P执行{hot['P']['encoder_calls']:.1f}次；V11与V11E的APC命中请求分别为{hot['V11']['prefix_hit_requests']:.0f}与{hot['V11E']['prefix_hit_requests']:.0f}。图片预处理缓存、语言前缀缓存和完整视觉结果驻留分别影响不同工作，是否省去视觉编码需要看真实调用数。不同场景的问题成员和输出长度有差别，缓存收益以各场景内的同问题配置对照判断。

资源方面，vLLM固定KV池为1648 MiB，CPU processor缓存为0.125 GiB；P的1536 MiB为持久负载，另有112 MiB工作KV。框架KV池还供活跃请求使用，encoder与图池另计，所以不将两者称为等总缓存容量。P、V11、V11E在会话中的平均峰值allocated分别为{session['P']['peak_allocated_bytes']/1024**3:.3f}、{session['V11']['peak_allocated_bytes']/1024**3:.3f}、{session['V11E']['peak_allocated_bytes']/1024**3:.3f} GiB，均在同一8188 MiB设备上串行运行。三者初始化均值为{resource('P','initialization_s'):.2f}、{resource('V11','initialization_s'):.2f}、{resource('V11E','initialization_s'):.2f} s，首次请求可能触发的懒编译另外保留在热页面预热记录中。

{quality_text}，最小逐种子变化为{delta:+.6f}。首token后八个CPU可见输出间隔的trace记录V00为{calls('V00')}，V01为{calls('V01')}，用于确认执行方式；框架可能提前执行下一步，该窗口不保证对应恰好八次GPU调用，也不能据此判断某一步回退。这些profiler时间不进入主表。原始答案、输出长度、EM、逐种子质量、配对t95区间、缓存调用与驱动采样均保存于框架实验目录。

'''
    md=ROOT/'docs/设计文档.md';text=md.read_text()
    text=text.replace('<!-- pagebreak -->\n\n## 3 实现细节','## 3 实现细节')
    text=text.replace('<!-- pagebreak -->\n\n## 4 实验验证与结果分析','## 4 实验验证与结果分析')
    text,n=re.subn(r'### 4\.7 .*?(?=### 4\.8 |## 5 )',lambda _:new,text,flags=re.S);assert n==1
    text=text.replace('当前相对Transformers原生路径的加速，不能直接作为超过vLLM的证据；拟补做的框架实验在4.7节给出。','与Transformers原生路径的开发对照说明阶段收益，4.7节进一步在独立评估集上比较完整请求路径和vLLM的实际缓存、图执行行为。')
    text=text.replace('本次结果定位为开发验证，新的独立评估与开放环矩阵尚未执行。','表3定位为开发验证，独立进程的框架评估见4.7节；开放环服务测量尚未执行。')
    text=text.replace('因此，本轮结果支持的是本机目标负载下的开发结论，而不是任意多模态请求都具有同样的加速比。','本轮开发消融与独立框架评估共同支持本机目标负载下的机制分析；具体收益随访问方式、缓存容量和执行栈变化。')
    text=text.replace('进一步验证将先固定新的源文档互斥评估集和测量协议，再比较连续同页、页面交错、全局随机与每页一次等访问方式；增加不同缓存预算、冷启动及原生/视觉缓存B=2对照。重复测量改为独立进程，沿用相同质量门槛、完整原始记录和预先固定的重复次数。','本次已完成源文档互斥的独立评估、三个独立进程种子、页面会话与每页一次对照。后续可进一步比较更大的页面工作集、全局随机访问、不同缓存预算、冷启动总成本和B=2对照，并补充到达率受控的开放环服务实验。')
    marker='模型固定revision为89644892e4d85e24eaac8bacfd4f463576704203'
    repro='''框架对照代码与完整原始记录位于stage_2/experiments/framework_compare/。运行以下命令复核9720条计时记录、72个区组和两份解码trace，并重建框架图表：

```bash
./stage_2/experiments/framework_compare/run.sh report
python stage_2/tools/update_framework_design.py
```

从数据准备到真实执行的入口为同目录run.sh all，复用完成单元且拒绝覆盖未完成目录。全新重跑需使用独立工作副本，前置条件、固定镜像、配置哈希和逐步命令见该目录README.md；报告复算不需要启动GPU。

'''
    if '复核9720条计时记录' not in text:text=text.replace(marker,repro+marker)
    text=text.replace('/v0.26.0/','/v0.32.0/').replace('/configuration/engine_args/','/cli/serve/')
    text=text.replace('调研参考v0.26.0文档，正式实验所用版本在兼容性验收后冻结。','正式对照固定v0.32.0既有镜像，完整image ID与配置在实验协议中保存。')
    md.write_text(text)
    for ext in ['png','svg','pdf']:shutil.copy2(HERE/f'figures/framework-latency.{ext}',ROOT/f'docs/assets/design/frameworks.{ext}')
    check={'analysis_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'table6_rows':records,
           'formal_quality_passed':audit['formal_quality_passed'],'timed_requests':audit['requests'],
           'protocol_sha256':audit['protocol_sha256'],'figure2':'Three simple boxes; two optimization levels',
           'document_source_sha256':hashlib.sha256(md.read_bytes()).hexdigest()}
    (ROOT/'docs/框架对比摘要核对.json').write_text(json.dumps(check,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(check,ensure_ascii=False))


if __name__=='__main__':main()
