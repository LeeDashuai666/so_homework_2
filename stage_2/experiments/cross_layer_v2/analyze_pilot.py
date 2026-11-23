#!/usr/bin/env python3
"""Audit raw development records and rebuild the second-round report and figures."""

import argparse
import base64
from collections import Counter, defaultdict
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from markdown_it import MarkdownIt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vlmbench.metrics import quality, quantiles

VARIANTS = ["NATIVE", "VISUAL_HF", "STATIC_EAGER", "PREFIX", "GRAPH", "BOTH"]


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def table(headers, values):
    return "| " + " | ".join(headers) + " |\n| " + " | ".join("---" for _ in headers) + " |\n" + "\n".join(
        "| " + " | ".join(map(str, row)) + " |" for row in values)


def render(path):
    body = MarkdownIt("commonmark", {"html": True}).enable("table").render(path.read_text())
    def embed(match):
        local = path.parent / html.unescape(match.group(1))
        if local.is_file() and local.suffix == ".png":
            return '<img src="data:image/png;base64,' + base64.b64encode(local.read_bytes()).decode() + '"'
        return match.group(0)
    body = re.sub(r'<img src="([^"]+)"', embed, body)
    style = "body{max-width:1080px;margin:40px auto;padding:0 25px;color:#253044;font:16px/1.85 system-ui,'Noto Sans CJK SC',sans-serif}h2{margin-top:34px;border-bottom:1px solid #ddd}table{border-collapse:collapse;display:block;overflow:auto;font-size:14px}td,th{padding:7px 10px;border:1px solid #ddd}th,pre{background:#f1f4f8}img{max-width:100%}pre{padding:14px;overflow:auto}a{color:#286a9e}@media print{body{margin:0;font-size:11pt}img,table{break-inside:avoid}}"
    path.with_suffix(".html").write_text(f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(path.stem)}</title><style>{style}</style><body>{body}</body></html>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "results/pilot")
    parser.add_argument("--report", type=Path, default=HERE / "开发验证报告.md")
    parser.add_argument("--runtime-check", type=Path, default=HERE / "results/runtime-check.json")
    args = parser.parse_args()
    source = args.input.resolve()
    manifest, complete = read(source / "manifest.json"), read(source / "complete.json")
    assert complete["complete"] and complete["quality_gate_passed"]
    assert manifest["read_split"] == "development only"
    for name, expected in manifest["source_sha256"].items():
        assert sha(HERE / name) == expected, f"Measured source changed: {name}"
    protocol = read(ROOT / "configs/protocol.lock.json")
    for name, expected in protocol["source_sha256"].items():
        assert sha(ROOT / name) == expected, f"First-round frozen source changed: {name}"
    data = rows(ROOT / "data/processed/docvqa/development.jsonl")
    assert manifest["quality_count"] == len(data), "Report requires the complete development quality gate"
    assert manifest["repeats"] == 3, "This development report uses the registered three-block t interval"
    references = {r["question_id"]:r for r in data}
    pages = defaultdict(list)
    for r in data:
        pages[r["page_id"]].append(r)
    chosen_pages = [page for page in pages.values() if len(page) >= 2][:manifest["timing_pages"]]
    expected_timing_ids = {r["question_id"] for page in chosen_pages for r in page[1:]}
    quality_rows, timing_rows = rows(source / "quality-requests.jsonl"), rows(source / "timing-requests.jsonl")
    assert len(quality_rows) == len(VARIANTS)*manifest["quality_count"]
    assert len(timing_rows) == len(VARIANTS)*manifest["repeats"]*manifest["timing_questions"]
    quality_summary, recorded_blocks = read(source / "quality-summary.json"), read(source / "timing-blocks.json")
    quality_maps, timing_maps, blocks, means = {}, {}, [], {}
    quota = manifest["cache_quota_mib"] * 1024**2
    for variant in VARIANTS:
        selected = [r for r in quality_rows if r["variant"] == variant]
        assert len(selected) == manifest["quality_count"]
        quality_maps[variant] = {r["question_id"]:r for r in selected}
        assert len(quality_maps[variant]) == len(selected)
        for r in selected:
            score = quality(r["answer"], references[r["question_id"]]["answers"])
            assert abs(score["anls"] - r["anls"]) < 1e-12 and score["em"] == r["em"]
            assert len(r["token_ids"]) <= 64 and r["decode_steps"] == len(r["token_ids"]) - 1
            assert r["persistent_payload_bytes"] <= quota
        assert abs(st.mean(r["anls"] for r in selected) - quality_summary[variant]["anls"]) < 1e-12
        assert abs(st.mean(r["em"] for r in selected) - quality_summary[variant]["em"]) < 1e-12
        for name, value in quantiles(r["ttft_s"] for r in selected).items():
            assert math.isclose(value,quality_summary[variant]["ttft_s"][name],abs_tol=1e-12)
        assert abs(sum(r["generation_s"] for r in selected) - quality_summary[variant]["sum_generation_s"]) < 1e-8
        timing_maps[variant] = {(r["seed"],r["question_id"]):r for r in timing_rows if r["variant"] == variant}
        assert len(timing_maps[variant]) == manifest["repeats"] * manifest["timing_questions"]
        for seed in range(2901,2901+manifest["repeats"]):
            selected = [r for r in timing_maps[variant].values() if r["seed"] == seed]
            assert {r["question_id"] for r in selected} == expected_timing_ids
            actual = next(b for b in recorded_blocks if b["variant"] == variant and b["seed"] == seed)
            b = {"seed":seed,"variant":variant,"questions":len(selected),
                 "mean_generation_s":st.mean(r["generation_s"] for r in selected),
                 "mean_ttft_s":st.mean(r["ttft_s"] for r in selected),
                 "total_decode_s":sum(r["decode_s"] for r in selected),
                 "total_decode_steps":sum(r["decode_steps"] for r in selected),
                 "prefix_hit_rate":st.mean(r.get("prefix_hit",False) for r in selected),
                 "peak_allocated_bytes":max(r["peak_allocated_bytes"] for r in selected)}
            for k in b:
                assert math.isclose(b[k],actual[k],abs_tol=1e-8) if isinstance(b[k],(float,int)) else b[k] == actual[k]
            b["decode_ms_per_token"] = 1000*b["total_decode_s"]/b["total_decode_steps"]
            blocks.append(b)
        chosen = [b for b in blocks if b["variant"] == variant]
        means[variant] = {"variant":variant}
        for k in ["mean_generation_s","mean_ttft_s","decode_ms_per_token"]:
            means[variant][k] = st.mean(b[k] for b in chosen)
            means[variant][k+"_sd"] = st.stdev(b[k] for b in chosen)
        means[variant]["peak_allocated_gib"] = max(b["peak_allocated_bytes"] for b in chosen)/1024**3
    for v in VARIANTS:
        assert set(quality_maps[v]) == set(quality_maps["NATIVE"])
        assert set(timing_maps[v]) == set(timing_maps["NATIVE"])
        assert all(timing_maps[v][k]["token_ids"] == timing_maps["NATIVE"][k]["token_ids"] for k in timing_maps[v])
    pairs = {}
    for a,b in [("NATIVE","VISUAL_HF"),("STATIC_EAGER","GRAPH"),("PREFIX","BOTH")]:
        count = sum(quality_maps[a][k]["token_ids"] == quality_maps[b][k]["token_ids"] for k in quality_maps[a])
        assert count == manifest["quality_count"]
        pairs[a+"__"+b] = count
    gate = read(source / "quality-gate.json")
    native_anls = quality_summary["NATIVE"]["anls"]
    for v in VARIANTS:
        delta = quality_summary[v]["anls"] - native_anls
        same = sum(quality_maps[v][k]["token_ids"] == quality_maps["NATIVE"][k]["token_ids"] for k in quality_maps[v])
        assert delta >= gate["threshold"] == -0.005
        assert math.isclose(delta,gate["variants"][v]["anls_delta_vs_native"],abs_tol=1e-12)
        assert same == gate["variants"][v]["identical_tokens_vs_native"]
    launches = read(source / "launch-counts.json")
    trace_metrics = {}
    for kind in ["eager","graph"]:
        trace = read(source / f"decode-{kind}-trace.json")
        counts = Counter(e["name"] for e in trace["traceEvents"] if e.get("cat") == "cuda_runtime"
                         and ("cudaLaunchKernel" in e.get("name","") or "cudaGraphLaunch" in e.get("name","")))
        assert dict(counts) == launches[kind]["runtime_launch_calls"]
        kernels = [e for e in trace["traceEvents"] if e.get("cat") == "kernel"]
        intervals = sorted((e["ts"],e["ts"]+e["dur"]) for e in kernels)
        merged = []
        for start,end in intervals:
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1],end)
            else:
                merged.append([start,end])
        covered = sum(end-start for start,end in merged)
        span = max(end for _,end in intervals)-intervals[0][0]
        trace_metrics[kind] = {"kernel_events":len(kernels),"kernel_union_ms":covered/1000,
                               "first_to_last_kernel_ms":span/1000,"kernel_gap_ms":(span-covered)/1000,
                               "kernel_coverage_fraction":covered/span,
                               "scope":"Profiled decode window; includes profiler perturbation, not GPU occupancy or unprofiled latency"}
    runtime = read(args.runtime_check)
    assert runtime["passed"] and runtime["source_sha256"]["runtime.py"] == sha(HERE / "runtime.py")
    paired = []
    for a,b,k,scale,label in [("STATIC_EAGER","PREFIX","mean_ttft_s",1000,"A: TTFT saved, ms"),
                             ("STATIC_EAGER","GRAPH","decode_ms_per_token",1,"B: decode saved, ms/token"),
                             ("VISUAL_HF","BOTH","mean_generation_s",1000,"Combined generation saved, ms")]:
        differences = [(next(x for x in blocks if x["seed"] == seed and x["variant"] == a)[k] -
                        next(x for x in blocks if x["seed"] == seed and x["variant"] == b)[k])*scale
                       for seed in range(2901,2901+manifest["repeats"])]
        mean = st.mean(differences)
        # Exact t critical is only registered for this three-block development pilot.
        half = 4.302652729911275*st.stdev(differences)/math.sqrt(3) if len(differences) == 3 else None
        paired.append({"comparison":label,"mean_saved":mean,"block_differences":differences,
                       "t95_low":mean-half if half is not None else None,"t95_high":mean+half if half is not None else None})
    for filename, values in [("timing-means.csv",list(means.values())),("timing-blocks-recomputed.csv",blocks)]:
        with (source / filename).open("w",newline="") as stream:
            writer = csv.DictWriter(stream,fieldnames=list(values[0]))
            writer.writeheader(); writer.writerows(values)
    audit = {"passed":True,"scope":"DEVELOPMENT only; formal first-round results unchanged",
             "quality_rows":len(quality_rows),"timing_rows":len(timing_rows),"timing_blocks":len(blocks),
             "exact_token_pairs":pairs,"all_timing_tokens_identical":True,
             "cache_quota_bytes":quota,"trace_launch_counts_recomputed":True,
             "trace_diagnostics":trace_metrics,
             "runtime_invariants_passed":True,"means":means,"paired_diagnostics":paired,
             "first_round_frozen_sources_unchanged":True,
             "audit_source_sha256":{p.name:sha(p) for p in [Path(__file__),HERE / "run.sh",HERE / "verify_runtime.py"]},
             "raw_sha256":{p.name:sha(p) for p in source.iterdir()
                           if p.suffix in (".json", ".jsonl") and p.name != "audit-and-analysis.json"},
             "input_sha256":{str(p.relative_to(ROOT)):sha(p) for p in [ROOT / "data/processed/docvqa/development.jsonl",
                              ROOT / "configs/protocol.lock.json",ROOT / "configs/vlm-assets.lock.json"]},
             "measured_source_sha256":manifest["source_sha256"]}
    write(source / "audit-and-analysis.json",audit)
    figures = args.report.parent / "figures"
    figures.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
    fig,axes = plt.subplots(1,3,figsize=(13,4))
    labels = ["Visual HF","Static eager","Prefix","Graph","Both"]
    chosen = VARIANTS[1:]
    for ax,metric,scale,label in zip(axes,["mean_ttft_s","decode_ms_per_token","mean_generation_s"],
                                   [1000,1,1000],["TTFT (ms)","Decode (ms/token)","Answer generation (ms)"]):
        values = [means[v][metric]*scale for v in chosen]
        ax.bar(range(5),values,yerr=[means[v][metric+"_sd"]*scale for v in chosen],capsize=3,
               color=["#8d99ae","#c3cbd5","#e2a24c","#619bcb","#3a9382"])
        for i,v in enumerate(chosen):
            ax.scatter([i]*3,[b[metric]*scale for b in blocks if b["variant"] == v],color="#304050",s=16,zorder=3)
        ax.set_xticks(range(5),labels,rotation=32,ha="right"); ax.set_ylabel(label); ax.set_ylim(bottom=0)
    fig.suptitle(f"Development pilot: {manifest['timing_questions']} real hot-page questions; three paired blocks (mean ± SD)")
    fig.tight_layout()
    for ext in ["png","svg","pdf"]:
        fig.savefig(figures / f"cross-layer-pilot.{ext}",dpi=180,bbox_inches="tight")
    plt.close(fig)
    capture = read(source / "graph-build.json")["seconds"]
    amortize = math.ceil(capture/(means["STATIC_EAGER"]["mean_generation_s"]-means["GRAPH"]["mean_generation_s"]))
    performance = table(["路径","平均 TTFT/ms","平均完整生成/ms","解码/ms 每 token","热页面峰值 allocated/GiB"],
                        [[v,f"{means[v]['mean_ttft_s']*1000:.1f} ± {means[v]['mean_ttft_s_sd']*1000:.1f}",
                          f"{means[v]['mean_generation_s']*1000:.1f} ± {means[v]['mean_generation_s_sd']*1000:.1f}",
                          f"{means[v]['decode_ms_per_token']:.2f} ± {means[v]['decode_ms_per_token_sd']:.2f}",
                          f"{means[v]['peak_allocated_gib']:.3f}"] for v in VARIANTS])
    quality_table = table(["路径","ANLS","相对原生变化","与原生 token 完全相同","全开发集峰值 allocated/GiB"],
                          [[v,f"{quality_summary[v]['anls']:.6f}",f"{quality_summary[v]['anls']-native_anls:+.6f}",
                            f"{gate['variants'][v]['identical_tokens_vs_native']}/{manifest['quality_count']}",
                            f"{quality_summary[v]['peak_allocated_bytes']/1024**3:.3f}"] for v in VARIANTS])
    paired_table = table(["配对比较","平均节省","三组区组差值","开发区组 t95 区间"],
                         [[p["comparison"],f"{p['mean_saved']:.2f}",", ".join(f"{v:.2f}" for v in p["block_differences"]),
                           f"[{p['t95_low']:.2f}, {p['t95_high']:.2f}]" if p["t95_low"] is not None else "—"] for p in paired])
    trace_table = table(["八步 decode trace","GPU kernel 事件数","kernel 覆盖时长/ms","首末 kernel 跨度/ms","窗口 kernel 覆盖比例"],
                        [[v,str(trace_metrics[v]["kernel_events"]),f"{trace_metrics[v]['kernel_union_ms']:.2f}",
                          f"{trace_metrics[v]['first_to_last_kernel_ms']:.2f}",
                          f"{trace_metrics[v]['kernel_coverage_fraction']*100:.1f}%"] for v in ["eager","graph"]])
    rel = lambda path: os.path.relpath(path.resolve(),args.report.parent.resolve())
    result_link = rel(source)
    prefix_rows = list(quality_maps["PREFIX"].values())
    misses = [r for r in prefix_rows if not r["prefix_hit"]]
    hits = [r for r in prefix_rows if r["prefix_hit"]]
    average_miss_prefill = st.mean(r["language_prefill_tokens"] for r in misses)
    average_hit_prefill = st.mean(r["language_prefill_tokens"] for r in hits)
    persistent_max_mib = max(r["persistent_payload_bytes"] for r in quality_rows)/1024**2
    block_steps = sorted(set(b["total_decode_steps"] for b in blocks))
    assert len(block_steps) == 1
    eager_launches = sum(launches["eager"]["runtime_launch_calls"].values())
    graph_launches = sum(launches["graph"]["runtime_launch_calls"].values())
    text = f'''# 第二轮开发验证：多模态前缀复用与 CUDA Graph 解码

状态：**开发验证已完成，两项新方向进入正式实验候选。尚未重跑第二轮独立评估与开放环实验。** 第一轮 45 轮主实验、18 轮辅助实验、558 问评估、报告与提交包完整保留。

## 1. 经验与本轮选择

第一轮视觉特征缓存有直接收益；图像亲和调度的独立增益区间跨零，且存在尾延迟取舍；价值替换的预算实验吞吐差绝对值最大约 0.59%，不足以支持其作为核心贡献。继续更换调度权重或替换评分函数，难以增加有力的跨层证据。

第二轮保留同一文档多问的真实多模态任务和已有视觉缓存，采用两个方向：

| 方向 | 对应层次 | 实际减少的工作 | 独立证据 |
| --- | --- | --- | --- |
| A：多模态公共前缀 KV 复用 | 软件算法／模型计算图 | 重复图像前缀的语言 prefill | 命中后只计算问题后缀，TTFT 下降 |
| B：静态 KV + CUDA Graph 解码 | CPU–GPU 协同／GPU 执行 | 逐 token 的主机 CUDA 提交 | 真实 CUDA trace 中 runtime 发射调用减少，decode 时间下降 |

视觉特征缓存和前缀缓存属于同一个计算复用层，不能将二者计为“两层”。作业中的两个不同层次由 **A 的模型计算复用与 B 的 GPU 执行优化**共同满足；不需要更换物理硬件。B 并非单纯切换到 GPU，而是改变已有 GPU 的命令提交和静态显存执行方式。

建议题目：**面向文档视觉问答的跨层推理优化——多模态前缀状态复用与 CUDA Graph 解码。** 详细方案见[经验总结与第二轮优化](../../plan/经验总结与第二轮优化.md)。

## 2. 测量对象与控制条件

- 设备：WSL2 上的 RTX 4060 Laptop 8GB，复用现有 PyTorch 2.14.0+cu130 / Transformers 5.17.0；本轮没有安装或升级基础环境。
- 模型：固定 revision 的 Qwen3-VL-2B-Instruct；BF16/SDPA、单请求 B=1、每图最多 768 视觉 token、greedy、相同提示、EOS 与 64 新 token 上限。
- 只读取第一轮的 **64 页 282 问开发集**。第二轮没有读取原 558 问评估输出，也没有用新独立评估集选参数。
- 缓存方案共用 **{manifest['cache_quota_mib']} MiB 持久负载上限**，同页视觉特征与前缀 KV 使用互斥表示。模型、静态工作 KV、图池和临时张量另计，并测量总 allocated；这个预算不同于第一轮，不能直接与第一轮的吞吐横比。
- 六路径：NATIVE 原生推理；VISUAL_HF 原生生成器加完整视觉缓存；STATIC_EAGER 视觉缓存加静态 KV/普通逐步执行；PREFIX 加 A；GRAPH 加 B；BOTH 加 A+B。后三者相对于 STATIC_EAGER 构成 2×2 消融。
- 热页面计时取开发划分中前 {manifest['timing_pages']} 个至少两问的页面，用每页第一条**不同的真实问题**预热，测其余全部 {manifest['timing_questions']} 问。三个种子 2901–2903 随机化六路径顺序，并在相同问题上配对；总计 {len(blocks)} 区组、{len(timing_rows)} 条计时记录。
- 指标从已准备好 GPU 输入开始，TTFT 是首个 CPU 可见 token，完整生成包含真实 EOS 与答案解码；decode 是首 token 后的实际逐步循环时间，含公共同步与 EOS 检查。**这些不是含排队、图像预处理和网络的服务端到端延迟，也不是纯 GPU kernel 时长。**
- 原生路径保留 DynamicCache；所有路径在同一个原型进程中执行，静态工作 KV 与捕获图也常驻。显存数值表示这个共同测量进程的 PyTorch allocated，不能当作独立纯 HF 程序的最低显存需求。

## 3. 热页面配对计时结果

下表为三个区组的均值 ± 样本标准差；每区组均为相同 {manifest['timing_questions']} 问、**{block_steps[0]} 次首 token 后的 decode 步骤**。六路径在全部 {len(timing_maps['NATIVE'])} 个“问题×种子”配对上输出 token 完全相同，性能收益来自执行变化。

{performance}

![开发集跨层分解](figures/cross-layer-pilot.png)

**A 的独立作用：** 对比 STATIC_EAGER 与 PREFIX，TTFT 从 {means['STATIC_EAGER']['mean_ttft_s']*1000:.1f} 降至 {means['PREFIX']['mean_ttft_s']*1000:.1f} ms，下降 {(1-means['PREFIX']['mean_ttft_s']/means['STATIC_EAGER']['mean_ttft_s'])*100:.1f}%；decode 基本不变，符合“只复用前缀 prefill”的机制。

**B 的独立作用：** 对比 STATIC_EAGER 与 GRAPH，decode 从 {means['STATIC_EAGER']['decode_ms_per_token']:.2f} 降至 {means['GRAPH']['decode_ms_per_token']:.2f} ms/token，下降 {(1-means['GRAPH']['decode_ms_per_token']/means['STATIC_EAGER']['decode_ms_per_token'])*100:.1f}%；TTFT 基本不变，符合“仅捕获单 token 解码”的机制。STATIC_EAGER 比 VISUAL_HF 的普通 decode 更慢，因此保留原生强基线很有必要；不能将静态 KV 本身写成已证实的加速贡献。

**组合效果：** BOTH 完整生成 {means['BOTH']['mean_generation_s']*1000:.1f} ms，对比已有视觉缓存的 VISUAL_HF {means['VISUAL_HF']['mean_generation_s']*1000:.1f} ms，生成阶段速度比为 **{means['VISUAL_HF']['mean_generation_s']/means['BOTH']['mean_generation_s']:.2f}×**，耗时下降 {(1-means['BOTH']['mean_generation_s']/means['VISUAL_HF']['mean_generation_s'])*100:.1f}%。对比完全原生为 {means['NATIVE']['mean_generation_s']/means['BOTH']['mean_generation_s']:.2f}×。两项作用于不同阶段，采用强基线后仍有直接收益。

{paired_table}

这些 t 区间使用三个配对区组、df=2；区组在同一进程中，反映开发计时的一致性，不能替代独立进程和新评估集的正式推断。第三个区组普通解码较慢，原始区组全部保留，不删掉不利或波动结果。

## 4. 质量、状态隔离与 GPU 证据

全部 282 道开发问题按相同 session 顺序执行，第一问真实 miss，不预先填满缓存。每路径均正常生成，原始答案和 token 全部保留。原定质量门槛为相对原生 ANLS 下降不超过 0.005，本轮全部通过。

{quality_table}

STATIC_EAGER 相对原生有 {manifest['quality_count']-gate['variants']['STATIC_EAGER']['identical_tokens_vs_native']} 问 token 不同，PREFIX/BOTH 有 {manifest['quality_count']-gate['variants']['PREFIX']['identical_tokens_vs_native']} 问不同；前缀切分与静态执行形状会改变 BF16 数值路径。ANLS 的小幅正变化属于开发数值差异，不作为精度提升贡献。GRAPH 与 STATIC_EAGER **282/282 完全一致**，BOTH 与 PREFIX **282/282 完全一致**，图重放未额外改变答案。

前缀边界为首个 `<vision_end>`（含该 token），只保存固定角色和图像的各层语言 K/V；问题、assistant 后缀与回答仍独立计算。Key 包含固定模型 revision、处理配置、后端、图片 RGB 身份、grid、精确前缀 token 和 mRoPE 位置。本轮 {len(misses)} 次 miss、{len(hits)} 次命中：miss 的平均语言 prefill 为 {average_miss_prefill:.2f} token，hit 为 {average_hit_prefill:.2f} token。

GPU 状态验证已通过：同页不同问题共享正确前缀、异页 miss、持久前缀与工作 KV 存储地址分离、后续问题与回答不改写缓存、不同长度请求重放后地址稳定、容量边界和真实页面驱逐后的重新 miss。合成的长问题只用于容量检查：输入 {runtime['capacity_edge']['prompt_tokens']} token，固定检查 64 个输出 token，最终 KV 长度 {runtime['capacity_edge']['final_cache_length']}，容量 1024；没有混入质量或性能数据。细节见 [runtime-check.json](results/runtime-check.json)。这属于运行时不变量检查，第一轮 Z3 检查不被冒用为新前缀实现的形式化证明。

真实 VLM 的相同八步 decode profiler 记录：eager 为 **{eager_launches:,} 次 cudaLaunchKernel**，graph 为 **{graph_launches} 次 cudaGraphLaunch**，即平均每步 {eager_launches//8} 次 kernel host 发射变成一次 graph 发射。这里减少的是 **CPU runtime 提交次数**，图内部仍执行模型 kernels，不能表述为 GPU 计算 kernel 数从 {eager_launches:,} 减至 {graph_launches}。[原始 trace 与计数]({result_link}/launch-counts.json)均保留，审计脚本从 trace 重新计数。

{trace_table}

两种方式的 GPU kernel 事件数近似相同，捕获窗口中 kernel 覆盖更紧密，支持“减少主机提交和 kernel 间空隙”的执行层解释。此处覆盖比例为 kernel 时间区间并集除以首末 kernel 跨度；空隙还可能包含复制与同步，**不等于 SM occupancy、全进程 GPU 利用率或纯硬件空闲率**。Profiler 会放大 eager 的主机开销，因此该表只作机制诊断，性能收益采用第 3 节未开启 profiler 的实际配对计时。

## 5. 显存、miss 与启动成本

本模型完整视觉特征每视觉 token 16 KiB，语言 KV 每 token 112 KiB，后者是前者的 7 倍。PREFIX/BOTH 热页面计时峰值约 {means['BOTH']['peak_allocated_gib']:.2f} GiB，完整开发集质量循环峰值约 {quality_summary['BOTH']['peak_allocated_bytes']/1024**3:.2f} GiB allocated；持久缓存实际最大 {persistent_max_mib:.2f} MiB，未超 {manifest['cache_quota_mib']} MiB。allocated 不包含全部驱动/桌面开销，reserved 和驱动剩余信息另见运行时检查。前缀复制成本已包含在 TTFT，不能写成零成本命中。

一次图构建总计 **{capture:.3f} s**，包括首次完整 prefill、预热、快照/恢复和捕获；报告完整保留这笔启动费用。按这组热页面 STATIC_EAGER→GRAPH 的平均每请求节省估算，约 **{amortize} 次相似请求**可摊销这笔构建总费用；这是按开发均值计算的估算，未包含部署冷启动的其他开销。

在包含首次 miss 的 282 问单次开发扫描中，PREFIX 的生成时间和为 {quality_summary['PREFIX']['sum_generation_s']:.2f} s，STATIC_EAGER 为 {quality_summary['STATIC_EAGER']['sum_generation_s']:.2f} s，减少约 {(1-quality_summary['PREFIX']['sum_generation_s']/quality_summary['STATIC_EAGER']['sum_generation_s'])*100:.1f}%；PREFIX 的 TTFT p95 为 {quality_summary['PREFIX']['ttft_s']['p95']*1000:.1f} ms，VISUAL_HF 为 {quality_summary['VISUAL_HF']['ttft_s']['p95']*1000:.1f} ms。说明前缀方案的热命中收益不能直接推广到首次访问和低复用页面。BOTH 的生成时间和为 {quality_summary['BOTH']['sum_generation_s']:.2f} s，VISUAL_HF 为 {quality_summary['VISUAL_HF']['sum_generation_s']:.2f} s，但这组单次质量循环没有随机化重复，只作包含 miss 的辅助诊断，不代替上面的配对计时结论。

当前证据支持：**前缀复用适合有局部性的活跃文档多问，CUDA Graph 在本机 B=1 解码中收益明显。** 下一阶段必须加入全局随机访问、缓存不足、每页仅一次、冷启动和 B=2 强对照，检查适用范围。

## 6. 复现与交付

从仓库根目录运行：

```bash
./stage_2/experiments/cross_layer_v2/run.sh report   # 从已有原始数据复核并生成本文、CSV、PNG/SVG/PDF
./stage_2/experiments/cross_layer_v2/run.sh check --output /tmp/vlm-runtime-check-new.json
./stage_2/experiments/cross_layer_v2/run.sh pilot --output stage_2/experiments/cross_layer_v2/results/new-pilot
```

复用 `stage_2/env.sh`、已有模型锁、处理配置与开发数据；不重复安装依赖。pilot 拒绝覆盖非空结果目录，计时参数和测量代码 SHA256 写入 manifest。若要从新结果生成报告，显式传入 `report --input ... --report ...`。完整重跑与模型/数据前置条件见 [README](README.md)。

- 实现：[runtime.py](runtime.py)、[pilot.py](pilot.py)、[状态验证](verify_runtime.py)、[原始数据复核与报告生成](analyze_pilot.py)。
- 原始数据：[质量逐请求]({result_link}/quality-requests.jsonl)、[计时逐请求]({result_link}/timing-requests.jsonl)、[参数及源码哈希]({result_link}/manifest.json)、两份 CUDA profiler trace。
- 中间与结果：[六路径均值 CSV]({result_link}/timing-means.csv)、[区组 CSV]({result_link}/timing-blocks-recomputed.csv)、[从原始数据重算的审计结果]({result_link}/audit-and-analysis.json)。
- 图同时提供 [SVG](figures/cross-layer-pilot.svg) 和 [PDF](figures/cross-layer-pilot.pdf)，HTML 嵌入 PNG 可离线阅读和打印。

正式报告更新前，按[第二轮计划](../../plan/经验总结与第二轮优化.md)固定新的源文档互斥评估集、资源预算与重复次数；届时才将新实验结果合入最终作业交付。本文件的所有新性能数字均为**开发集验证**。
'''
    for label, target in {
        "runtime.py":HERE / "runtime.py", "pilot.py":HERE / "pilot.py",
        "verify_runtime.py":HERE / "verify_runtime.py", "analyze_pilot.py":Path(__file__),
        "README.md":HERE / "README.md", "results/runtime-check.json":args.runtime_check,
        "../../plan/经验总结与第二轮优化.md":ROOT / "plan/经验总结与第二轮优化.md",
    }.items():
        text = text.replace(f"]({label})",f"]({rel(target)})")
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(text)
    render(args.report)
    print(json.dumps({"audit_passed":True,"quality_rows":len(quality_rows),"timing_rows":len(timing_rows),
                      "report":str(args.report)},ensure_ascii=False))


if __name__ == "__main__":
    main()
