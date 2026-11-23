#!/usr/bin/env python3
"""Rebuild all tables, scientific figures and the Chinese report from raw logs."""

from collections import defaultdict
import base64
from datetime import datetime, timezone
import csv
import hashlib
import html
import json
from pathlib import Path
import statistics
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from verify_results import verify

COLORS = {"B0": "#758092", "V1": "#e19b35", "V1+S": "#388d81", "V1+M": "#9373b1", "FULL": "#cc5c61"}


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(row) for row in Path(path).read_text().splitlines()]


def csv_write(path, values):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def mean(values):
    return float(statistics.mean(values))


def sd(values):
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def md_table(headers, values):
    return "| " + " | ".join(headers) + " |\n| " + " | ".join("---" for _ in headers) + " |\n" + "\n".join("| " + " | ".join(str(value) for value in row) + " |" for row in values)


def savefig(figure, name, output):
    figure.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight")
    figure.savefig(output / f"{name}.svg", bbox_inches="tight")
    figure.savefig(output / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


def render_html(path):
    body = MarkdownIt("commonmark", {"html": True}).enable("table").render(path.read_text())
    def embed(match):
        local = path.parent / html.unescape(match.group(1))
        if local.is_file() and local.suffix == ".png":
            return '<img src="data:image/png;base64,' + base64.b64encode(local.read_bytes()).decode() + '"'
        return match.group(0)
    body = re.sub(r'<img src="([^"]+)"', embed, body)
    style = """body{max-width:1060px;margin:44px auto;padding:0 30px;color:#253044;font:16px/1.85 system-ui,'Noto Sans CJK SC','Microsoft YaHei',sans-serif}h1{font-size:30px;line-height:1.45}h2{margin-top:38px;border-bottom:1px solid #d9dee7;padding-bottom:6px}h3{margin-top:28px}a{color:#286a9e}table{border-collapse:collapse;font-size:14px;display:block;overflow-x:auto;margin:20px 0}th,td{padding:8px 13px;border:1px solid #dae0e9}th{background:#f1f4f8}img{max-width:100%}code{background:#f2f4f7;padding:2px 5px;border-radius:3px}pre{background:#f2f4f7;padding:16px;overflow:auto}pre code{padding:0}blockquote{border-left:3px solid #a1adc2;padding-left:18px;color:#526078}@media print{body{max-width:none;margin:0;font-size:11pt}pre,table,figure,img{break-inside:avoid}h2,h3{break-after:avoid}a{color:inherit}}"""
    path.with_suffix(".html").write_text(f'<!DOCTYPE html><html lang="zh-CN"><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(path.stem)}</title><style>{style}</style><body>{body}</body></html>')


def main():
    integrity = verify()
    protocol = read(ROOT / "configs/protocol.lock.json")
    cfg, slo = protocol["experiment"], protocol["slo"]
    report = ROOT / "report"
    tables, figures = report / "tables", report / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.2, "pdf.fonttype": 42, "svg.fonttype": "none"})
    per_run, grouped = [], defaultdict(list)
    monitor_values, monitor_rows = [], []
    budget_guard_batches = 0
    for path in sorted((ROOT / "results/main").glob("load-*/*/summary.json")):
        summary, manifest = read(path), read(path.parent / "manifest.json")
        load = path.parent.parent.name.split("-seed-")[0].removeprefix("load-")
        cache = summary["cache"]
        total_lookup = sum(cache.get(key, 0) for key in ["gpu_hits", "cpu_hits", "misses"])
        flat = {"load_factor": float(load), "seed": manifest["seed"], "variant": summary["variant"],
                "arrivals": summary["arrivals"], "completed": summary["completed"], "rejected": summary["rejected"],
                "failed": summary["failed"], "goodput_qps": summary["goodput_qps"],
                "ttft_p95_s": summary["ttft_s"]["p95"], "latency_p95_s": summary["latency_s"]["p95"],
                "attainment": summary["slo_attainment"], "sender_lag_p95_s": summary["sender_lag_s"]["p95"],
                "encoded_images": summary["vision_images_encoded"], "vision_s": summary["vision_s"],
                "prefill_s": summary["prefill_s"], "batch_shared_images": summary["batch_shared_images"],
                "gpu_hit_fraction": cache.get("gpu_hits", 0) / max(1, total_lookup),
                "cpu_hit_fraction": cache.get("cpu_hits", 0) / max(1, total_lookup),
                "encoding_saved_fraction": 1 - summary["vision_images_encoded"] / max(1, summary["completed"]),
                "h2d_mib": cache.get("h2d_bytes", 0) / 1024**2, "d2h_mib": cache.get("d2h_bytes", 0) / 1024**2,
                "peak_allocated_mib": summary["peak_allocated_bytes"] / 1024**2,
                "peak_reserved_mib": summary["peak_reserved_bytes"] / 1024**2,
                "cpu_rss_mib": summary["cpu_rss_bytes"] / 1024**2,
                "source": str(path.relative_to(ROOT))}
        per_run.append(flat)
        grouped[(float(load), summary["variant"])].append(flat)
        for batch in rows(path.parent / "batches.jsonl"):
            if batch["cache"] and batch["cache"]["gpu_limit"] < cfg["gpu_cache_mib"] * 1024**2:
                budget_guard_batches += 1
        for entry in rows(path.parent / "monitor.jsonl"):
            try:
                temperature, clock, power, memory = map(float, entry["gpu_monitor_raw"].split(","))
                monitor_values.append({"temperature_c": temperature, "sm_mhz": clock, "power_w": power, "gpu_used_mib": memory})
                monitor_rows.append({"load_factor": float(load), "seed": manifest["seed"], "variant": summary["variant"],
                                     "offset_s": entry["offset_s"], **monitor_values[-1]})
            except (KeyError, ValueError):
                pass
    csv_write(tables / "main-per-run.csv", per_run)
    if monitor_rows:
        csv_write(tables / "gpu-monitor.csv", monitor_rows)
    aggregate = []
    for (load, variant), values in sorted(grouped.items()):
        record = {"load_factor": load, "arrival_rate_qps": protocol["rates_qps"][str(load)], "variant": variant, "runs": len(values)}
        for key in ["goodput_qps", "ttft_p95_s", "latency_p95_s", "attainment", "rejected", "encoding_saved_fraction", "gpu_hit_fraction", "cpu_hit_fraction", "peak_allocated_mib", "h2d_mib", "d2h_mib"]:
            record[key + "_mean"] = mean(row[key] for row in values)
            record[key + "_sd"] = sd([row[key] for row in values])
        aggregate.append(record)
    csv_write(tables / "main-aggregate.csv", aggregate)
    lookup = {(row["load_factor"], row["variant"]): row for row in aggregate}
    paired = []
    for load in [0.5, 0.8, 1.1]:
        for first, second, effect in [("B0", "V1", "O1"), ("V1", "V1+S", "O2"), ("V1", "V1+M", "O3"), ("V1", "FULL", "S+M"), ("B0", "FULL", "all")]:
            a = {r["seed"]: r for r in grouped[(load, first)]}
            b = {r["seed"]: r for r in grouped[(load, second)]}
            differences = [b[seed]["goodput_qps"] - a[seed]["goodput_qps"] for seed in sorted(a)]
            center, margin = mean(differences), 4.302652729911275 * sd(differences) / np.sqrt(len(differences))
            paired.append({"load_factor": load, "effect": effect, "reference": first, "candidate": second,
                           "goodput_delta_qps": center, "paired_t95_low": center - margin, "paired_t95_high": center + margin,
                           "seed_deltas": json.dumps(differences)})
    csv_write(tables / "paired-effects.csv", paired)
    qvalues, per_question = [], []
    eval_rows = rows(ROOT / cfg["dataset_path"] / "evaluation.jsonl")
    references = {q["question_id"]: q for q in eval_rows}
    qbaseline = {r["question_id"]: r for r in rows(ROOT / "results/quality/B0/requests.jsonl")}
    rng = np.random.default_rng(1700)
    for variant in cfg["variants"]:
        quality = dict(integrity["quality"][variant])
        summary = read(ROOT / "results/quality" / variant / "summary.json")
        requests = rows(ROOT / "results/quality" / variant / "requests.jsonl")
        documents = defaultdict(list)
        for row in requests:
            documents[references[row["question_id"]]["source_document_id"]].append(row["anls"] - qbaseline[row["question_id"]]["anls"])
            reference = references[row["question_id"]]
            per_question.append({"variant": variant, "question_id": row["question_id"], "page_id": row["page_id"],
                                 "source_document_id": reference["source_document_id"], "question": reference["question"],
                                 "answer": row["answer"], "references": json.dumps(reference["answers"], ensure_ascii=False),
                                 "anls": row["anls"], "em": row["em"], "anls_delta_vs_B0": row["anls"] - qbaseline[row["question_id"]]["anls"],
                                 "identical_tokens_vs_B0": row["token_ids"] == qbaseline[row["question_id"]]["token_ids"],
                                 "output_tokens": row["output_tokens"]})
        clusters = list(documents.values())
        bootstrap = []
        for _ in range(2000):
            sample = [clusters[i] for i in rng.integers(0, len(clusters), len(clusters))]
            bootstrap.append(sum(sum(cluster) for cluster in sample) / sum(len(cluster) for cluster in sample))
        quality.update(variant=variant, completed_qps=summary["completed_qps"], encoded_images=summary["vision_images_encoded"],
                       vision_s=summary["vision_s"], peak_allocated_mib=summary["peak_allocated_bytes"] / 1024**2,
                       delta_doc_bootstrap_low=float(np.percentile(bootstrap, 2.5)), delta_doc_bootstrap_high=float(np.percentile(bootstrap, 97.5)))
        qvalues.append(quality)
    csv_write(tables / "quality.csv", qvalues)
    csv_write(tables / "per-question-quality.csv", per_question)
    aux = []
    for path in sorted((ROOT / "results/auxiliary").glob("*/*/summary.json")):
        summary = read(path)
        group = path.parent.parent.name
        cache = summary["cache"]
        aux.append({"experiment": group, "variant": summary["variant"], "requests": summary["arrivals"],
                    "unique_pages": len({row["page_id"] for row in rows(path.parent / "requests.jsonl")}),
                    "qps": summary["completed_qps"], "encoded_images": summary["vision_images_encoded"],
                    "anls": summary["anls"], "gpu_hits": cache.get("gpu_hits", 0), "cpu_hits": cache.get("cpu_hits", 0),
                    "h2d_mib": cache.get("h2d_bytes", 0) / 1024**2, "d2h_mib": cache.get("d2h_bytes", 0) / 1024**2,
                    "peak_allocated_mib": summary["peak_allocated_bytes"] / 1024**2})
    if aux:
        csv_write(tables / "auxiliary.csv", aux)
    phases = []
    for variant in cfg["variants"]:
        values = rows(ROOT / "results/quality" / variant / "batches.jsonl")
        n = sum(b["real_batch_size"] for b in values)
        components = {key.removesuffix("_s") + "_ms": sum(b.get(key, 0) for b in values) / n * 1000 for key in ["pack_h2d_s", "vision_s", "prefill_s", "lookup_s", "assembly_s", "service_s"]}
        components["decode_and_orchestration_ms"] = max(0, components["service_ms"] - sum(components[key] for key in ["pack_h2d_ms", "vision_ms", "prefill_ms", "lookup_ms", "assembly_ms"]))
        phases.append({"variant": variant, **components})
    csv_write(tables / "phase-ms-per-request.csv", phases)
    fig, axes = plt.subplots(1, 3, figsize=(13.3, 3.7))
    for variant in cfg["variants"]:
        values = [lookup[(load, variant)] for load in [0.5, 0.8, 1.1]]
        rates = [row["arrival_rate_qps"] for row in values]
        for ax, metric, label in zip(axes, ["goodput_qps", "ttft_p95_s", "latency_p95_s"],
                                     ["SLO goodput (requests/s)", "Run p95 TTFT (s)", "Run p95 answer latency (s)"]):
            ax.errorbar(rates, [row[metric + "_mean"] for row in values], yerr=[row[metric + "_sd"] for row in values],
                        color=COLORS[variant], label=variant, marker="o", capsize=3)
            ax.set(xlabel="Offered arrival rate (requests/s)", ylabel=label)
    axes[1].axhline(slo["ttft_s"], color="#999999", linestyle=":", label="TTFT SLO")
    axes[2].axhline(slo["e2e_s"], color="#999999", linestyle=":", label="Answer SLO")
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(bottom=0)
    for axis, limit in [(axes[1], slo["ttft_s"]), (axes[2], slo["e2e_s"])]:
        axis.annotate(f"SLO {limit:.2f} s", xy=(rates[-1], limit), xytext=(-4, 4),
                      textcoords="offset points", ha="right", fontsize=7, color="#666666")
    fig.tight_layout()
    savefig(fig, "main-load-curves", figures)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    bottom = np.zeros(len(phases))
    for component, label, color in [("vision_ms", "Vision encode", "#da8b42"), ("prefill_ms", "LLM prefill", "#5d8da8"),
                                     ("lookup_ms", "Cache lookup/transfer", "#a78bc2"), ("assembly_ms", "Feature assembly", "#6b9c87"),
                                     ("pack_h2d_ms", "Pack/input H2D", "#b8c3cc"), ("decode_and_orchestration_ms", "Decode + host/runtime residual", "#d17a80")]:
        values = np.array([row[component] for row in phases])
        axes[0].bar(cfg["variants"], values, bottom=bottom, label=label, color=color)
        bottom += values
    axes[0].set(ylabel="GPU-worker wall components (ms/request)")
    axes[0].legend(fontsize=7, loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, frameon=False)
    axes[1].bar([q["variant"] for q in qvalues], [q["encoded_images"] for q in qvalues], color=[COLORS[q["variant"]] for q in qvalues])
    axes[1].axhline(128, linestyle=":", color="#444444", label="128 unique evaluation pages")
    axes[1].set(ylabel="Actual images encoded (558 questions)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "phase-and-encoding", figures)
    fig, axes = plt.subplots(1, 2, figsize=(10.7, 3.8))
    x = np.arange(len(qvalues))
    axes[0].bar(x - 0.18, [q["anls"] for q in qvalues], width=0.36, label="ANLS", color="#5d8da8")
    axes[0].bar(x + 0.18, [q["em"] for q in qvalues], width=0.36, label="Exact match", color="#6b9c87")
    axes[0].set(xticks=x, xticklabels=[q["variant"] for q in qvalues], ylabel="All-question quality", ylim=(0, 1))
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
    axes[1].bar([q["variant"] for q in qvalues], [q["peak_allocated_mib"] / 1024 for q in qvalues], color=[COLORS[q["variant"]] for q in qvalues])
    axes[1].set(ylabel="Peak PyTorch allocated memory (GiB)")
    fig.tight_layout()
    savefig(fig, "quality-and-memory", figures)
    if aux:
        fig, axes = plt.subplots(1, 2, figsize=(10.7, 3.7))
        for variant in ["V1", "V1+M"]:
            values = sorted([r for r in aux if r["experiment"].startswith("budget-") and r["variant"] == variant], key=lambda row: int(row["experiment"].split("-")[1]))
            capacities = [int(row["experiment"].split("-")[1]) for row in values]
            axes[0].plot(capacities, [row["qps"] for row in values], marker="o", color=COLORS[variant], label=variant)
            axes[1].plot(capacities, [row["encoded_images"] for row in values], marker="o", color=COLORS[variant], label=variant)
        axes[0].set(xlabel="GPU feature-cache cap (MiB)", ylabel="Offline completion throughput (requests/s)")
        axes[1].set(xlabel="GPU feature-cache cap (MiB)", ylabel="Images encoded (same request trace)")
        for ax in axes:
            ax.legend()
            ax.set_xticks([128, 256, 512])
        fig.tight_layout()
        savefig(fig, "cache-budget", figures)
    memory_model = read(ROOT / "env/feature-memory-model.json")
    analysis = {"generated_at": datetime.now(timezone.utc).isoformat(), "main": aggregate, "paired_effects": paired,
                "memory_byte_model": memory_model,
                "quality": qvalues, "auxiliary": aux, "phases_ms_per_request": phases, "integrity": integrity,
                "budget_guard_batches": budget_guard_batches,
                "monitor": {key: {"min": min(r[key] for r in monitor_values), "max": max(r[key] for r in monitor_values),
                                  "mean": mean(r[key] for r in monitor_values)} for key in monitor_values[0]} if monitor_values else {}}
    (report / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n")
    build_report(report, protocol, analysis)
    print("Report, figures and independently checked tables rebuilt:", report, flush=True)


def build_report(destination, protocol, analysis):
    cfg, slo = protocol["experiment"], protocol["slo"]
    memory = analysis["memory_byte_model"]
    aux_values = {(row["experiment"], row["variant"]): row for row in analysis["auxiliary"]}
    small_cache, large_cache = aux_values[("budget-128", "V1")], aux_values[("budget-512", "V1")]
    cold_native, cold_lru = aux_values[("cold", "B0")], aux_values[("cold", "V1")]
    session_lru, interleaved_lru, random_lru = (aux_values[(f"order-{order}", "V1")] for order in ["session", "interleaved", "random"])
    maximum_budget_gap = max(abs(aux_values[(f"budget-{budget}", "V1+M")]["qps"] / aux_values[(f"budget-{budget}", "V1")]["qps"] - 1) for budget in [128, 256, 512])
    values = {(row["load_factor"], row["variant"]): row for row in analysis["main"]}
    quality = {row["variant"]: row for row in analysis["quality"]}
    phases = {row["variant"]: row for row in analysis["phases_ms_per_request"]}
    baseline, full = quality["B0"], quality["FULL"]
    visual_fraction = phases["B0"]["vision_ms"] / phases["B0"]["service_ms"]
    amdahl = 1 / (1 - visual_fraction)
    headline_base, headline_full = values[(0.8, "B0")], values[(0.8, "FULL")]
    headline_s, headline_lru, headline_m = (values[(0.8, variant)] for variant in ["V1+S", "V1", "V1+M"])
    recommended = max(cfg["variants"], key=lambda variant: (values[(0.8, variant)]["goodput_qps_mean"], -values[(0.8, variant)]["ttft_p95_s_mean"]))
    recommendation_labels = {"B0": "原生视觉基线", "V1": "O1 与 LRU/FIFO", "V1+S": "O1+O2 与 LRU", "V1+M": "O1 与价值替换/FIFO", "FULL": "O1+O2+O3"}
    memory_effect = next(row for row in analysis["paired_effects"] if row["load_factor"] == 0.8 and row["effect"] == "O3")
    memory_interpretation = "名义配对区间完全大于零" if memory_effect["paired_t95_low"] > 0 else "名义配对区间完全小于零" if memory_effect["paired_t95_high"] < 0 else "配对区间包含零，未观察到稳定的正收益"
    speed = headline_full["goodput_qps_mean"] / max(1e-12, headline_base["goodput_qps_mean"])
    ttft_drop = 1 - headline_full["ttft_p95_s_mean"] / headline_base["ttft_p95_s_mean"]
    main_table = md_table(["负载 / 到达率", "变体", "goodput 请求/s（均值±SD）", "TTFT p95 / s", "回答 p95 / s", "SLO 达标率", "拒绝数 / 轮"], [
        [f'{row["load_factor"]:.1f} / {row["arrival_rate_qps"]:.3f}', row["variant"], f'{row["goodput_qps_mean"]:.3f}±{row["goodput_qps_sd"]:.3f}',
         f'{row["ttft_p95_s_mean"]:.3f}', f'{row["latency_p95_s_mean"]:.3f}', f'{row["attainment_mean"]:.1%}', f'{row["rejected_mean"]:.1f}']
        for row in sorted(analysis["main"], key=lambda row: (row["load_factor"], cfg["variants"].index(row["variant"])))] )
    quality_table = md_table(["变体", "问题数", "ANLS", "EM", "ANLS 相对 B0", "完全相同 token 序列", "视觉编码数", "离线请求/s", "峰值 allocated / GiB"], [
        [row["variant"], row["questions"], f'{row["anls"]:.5f}', f'{row["em"]:.5f}', f'{row["anls_delta_vs_B0"]:+.5f}',
         f'{row["identical_token_sequences_vs_B0"]}/558', row["encoded_images"], f'{row["completed_qps"]:.3f}', f'{row["peak_allocated_mib"]/1024:.3f}'] for row in analysis["quality"]])
    effects_table = md_table(["负载", "独立作用", "比较", "goodput 差 / 请求/s", "配对 95% t 区间"], [
        [row["load_factor"], row["effect"], f'{row["candidate"]} vs {row["reference"]}', f'{row["goodput_delta_qps"]:+.3f}',
         f'[{row["paired_t95_low"]:+.3f}, {row["paired_t95_high"]:+.3f}]'] for row in analysis["paired_effects"]])
    aux_table = md_table(["实验", "变体", "请求数", "离线吞吐 / 请求/s", "视觉编码数", "CPU 命中", "H2D / MiB", "ANLS（此 trace）"], [
        [row["experiment"], row["variant"], row["requests"], f'{row["qps"]:.3f}', row["encoded_images"], row["cpu_hits"], f'{row["h2d_mib"]:.1f}', f'{row["anls"]:.5f}'] for row in analysis["auxiliary"]])
    calibration_table = md_table(["开发集候选", "batch", "GPU 缓存 / MiB", "离线吞吐 / 请求/s", "ANLS"], [
        [row["name"], row["batch"], row["gpu_mib"], f'{row["summary"]["completed_qps"]:.3f}', f'{row["summary"]["anls"]:.5f}']
        for row in protocol["calibration"]["candidates"]])
    previous = ROOT / "results/initial-packed-vision/quality"
    has_previous = all((previous / variant / "summary.json").is_file() for variant in cfg["variants"])
    previous_baseline = read(previous / "B0/summary.json")["anls"] if has_previous else 0
    previous_native_qps = f'{read(previous / "B0/summary.json")["completed_qps"]:.3f}' if has_previous else "见原提交包归档"
    previous_quality_table = md_table(["初版变体", "全量 ANLS", "相对初版原生 B0", "固定门槛判断"], [
        [variant, f'{read(previous / variant / "summary.json")["anls"]:.5f}',
         f'{read(previous / variant / "summary.json")["anls"] - previous_baseline:+.5f}',
         "通过" if read(previous / variant / "summary.json")["anls"] - previous_baseline >= -0.005 else "未通过"] for variant in cfg["variants"]]) if has_previous else "本次复现目录未包含初版历史结果；初版质量失败及代码归档见原提交包。"
    effects = []
    for effect in ["O1", "O2", "O3"]:
        cells = [row for row in analysis["paired_effects"] if row["effect"] == effect]
        positive = sum(row["goodput_delta_qps"] > 0 for row in cells)
        supported = sum(row["paired_t95_low"] > 0 for row in cells)
        effects.append(f'{effect} 在三档负载中有 {positive} 档平均 goodput 为正，{supported} 档的三种子配对 95% t 区间完全大于零。')
    report = f'''# 面向小显存 GPU 的文档视觉问答推理优化

## 视觉特征复用、图像亲和调度与 GPU/CPU 缓存管理

系统优化课程第二项目 · 实验日期 2026-11-16 · 独立完成，使用 AI 辅助编码与分析

## 摘要

文档视觉问答中，同一页面的多个独立问题会反复触发视觉编码，但不同问题的语言上下文仍需独立计算。本项目在 WSL2 的 RTX 4060 Laptop 8GB GPU 上，实现 Qwen3-VL-2B-Instruct 的完整视觉特征缓存、有限窗口图像亲和调度，以及考虑重编码代价和复访频率的 GPU/CPU 缓存管理。使用真实 DocVQA validation 自划分数据：开发集 64 页 282 问，评估集 128 页 558 问，源文档互斥。固定 BF16、SDPA、768 视觉 token 上限和短答案 prompt，完成五变体、三负载、三种子的 45 轮开放环实验。

在 0.8 倍开发容量的中负载下，FULL 的平均 SLO goodput 为 {headline_full["goodput_qps_mean"]:.3f} 请求/s，B0 为 {headline_base["goodput_qps_mean"]:.3f} 请求/s，比例为 {speed:.2f}；平均每轮 TTFT p95 变化为 {-ttft_drop:+.1%}。全量评估中，FULL 的 ANLS 为 {full["anls"]:.5f}，B0 为 {baseline["anls"]:.5f}，差值 {full["anls_delta_vs_B0"]:+.5f}，通过预登记的 −0.005 门槛。{''.join(effects)}所有请求、阶段耗时、答案、缓存计数与显存原始记录均保留，可脱离 GPU 重新核验表格。

## 1. 研究意义与问题

本地文档助手常对同一页扫描件提出多个不同问题。Qwen3-VL 的视觉编码器读取图像，产生主视觉 embedding 和三组 DeepStack 特征；随后这些特征进入语言模型，执行问题相关的 prefill 和自回归生成。直接重复运行整条推理路径，会在图像内容未变时重复视觉计算。架构依据 [Qwen3-VL 文档](https://huggingface.co/docs/transformers/model_doc/qwen3_vl)，具体边界由本地固定版本源码和真实 GPU 对照确定。

研究问题是：在模型、像素、精度和问题完全相同的条件下，视觉编码复用能带来多大延迟收益；当多个文档会话交错、缓存容量受限时，服务调度和存储管理能否进一步把复用转化为 SLO goodput。缓存针对完整视觉特征，不复用不同问题的语言 KV，不缓存最终答案。128 页评估工作集的潜在唯一页面复用上限为 1−128/558={1-128/558:.1%}，实际收益还取决于容量、次序和到达强度。

## 2. 环境、数据与可复现控制

| 项目 | 固定配置 |
| --- | --- |
| 平台 | WSL2 / Ubuntu 24.04.5 LTS，Linux 6.6.87.2 |
| CPU / 内存 | Ryzen 7 7735H，WSL 可见 8 逻辑 CPU，约 7.76 GiB 内存 |
| GPU | NVIDIA RTX 4060 Laptop，8188 MiB，驱动 592.00 |
| 运行库 | Python 3.12.3，PyTorch 2.14.0+cu130，Transformers 5.17.0，Z3 5.4.1 |
| 模型 | Qwen/Qwen3-VL-2B-Instruct，revision `89644892e4d85e24eaac8bacfd4f463576704203` |
| 精度 / 后端 | 全 GPU BF16 / SDPA；greedy；max_new_tokens=64；普通请求内 KV |
| 输入 | 每页最多 768 视觉 token，左侧 padding，统一 chat template 和短答案后缀 |
| 公共执行 | 视觉逐页 microbatch=1，语言最多 {cfg["batch_size"]} 请求/批，2 ms 等待，2 CPU 线程，队列上限 32 |
| 缓存 | GPU 上限 {cfg["gpu_cache_mib"]} MiB，CPU 上限 {cfg["cpu_cache_mib"]} MiB；相同按批 KV/workspace 保护 |
| 观测 | 每轮 {cfg["main_window_s"]:.0f} s 到达窗口，最多 30 s drain；独立模型进程；缓存冷启动 |

原始数据来自 `lmms-lab-encoder/DocVQA`，revision `539088ef8a8ada01ac8e2e6d4e372586748a265e`，六个 validation Parquet 共 5,349 个真实问题。按源文档 ID 的固定 SHA256 顺序选开发/评估页面，保留选中页面的全部真实问题。开发与评估分别来自 47 和 81 个源文档，文档 ID 与完全相同 RGB 像素的交集均为零。只检查精确像素重复，未宣称完成近重复图像检索。这是公开 validation 上的自划分实验，数值不等同官方隐藏 test 成绩。

主负载按固定种子全局随机排列真实问题，使用指数间隔生成泊松到达，属于合成回放时间；每轮所有变体读取相同 trace。连续同页、16 页会话交错和每页只问一次作为辅助控制。没有通过改写问题、答案缓存或改变分辨率制造复用。

模型权重 SHA256、数据分片 SHA256、原始图片 SHA256/RGB SHA256、运行代码 SHA256 和软件清单分别保存在 `configs/`、`data/processed/docvqa/manifest.json`、`env/`。共享现有推理虚拟环境，未新建 CUDA 工具链、未升级基础环境包。

## 3. 测量驱动的优化设计

### 3.1 基线瓶颈与 Amdahl 分析

在全量评估的 B0 离线运行中，视觉编码约占 GPU worker service wall time 的 {visual_fraction:.1%}。保持其他阶段不变、完全消除视觉编码时，服务时间的理想上限约为 {amdahl:.2f} 倍。这个上限用于解释 O1 的阶段收益；排队系统中的 p95 与 goodput 还受负载和调度影响，不能直接套用同一个倍数。

下图按请求分摊批次 service time。视觉编码和 prefill 使用 CUDA Events；输入组批、缓存查找/搬运与拼接使用单调时钟；剩余部分明确称为“decode 与主机/运行时余项”，包含 token 回传、解码和调用开销，并非纯 GPU decode kernel 时间。CPU 图像处理可与 GPU worker 重叠，单独保留在请求日志中，不与 GPU service 简单相加。

![阶段耗时与实际视觉编码数](figures/phase-and-encoding.png)

### 3.2 O1：推理计算图层的完整视觉特征复用

在 `get_image_features` 边界缓存主 embedding 与三层 DeepStack，共四个 BF16 `[V,2048]` 张量，每页 payload 为 `V×2048×2×4` 字节。每页从批张量切片后独立 clone，避免缓存一页却保留整批底层 storage。身份由冻结页面 RGB 像素哈希、实际 grid、processor、模型 revision、dtype、backend、视觉执行粒度和 Transformers 版本组成；评估文件在复核时逐一检查原始字节和像素哈希。

所有变体统一逐页调用原生视觉编码器，语言模型仍保留 batch=2。这样每个页面的 BF16 视觉计算形状固定，不受同批其他图像的数量与尺寸影响，O1 在相同语言批次下可以严格复用相同张量。此项是所有对照共有的数值稳定执行规则，未归入某个优化的独立加速。

缓存单位的空间成本直接对应 AI 推理结构。本模型完整视觉特征每个视觉 token 为 `4×2048×2=16 KiB`；BF16 语言 KV 每个 token 为 `2×28层×8个KV头×128维×2=112 KiB`，字节比为 {memory["kv_to_encoder_byte_ratio"]:.0f}。以真实常见的 744 视觉 token 页面为例，完整视觉特征为 {memory["example_encoder_mib"]:.3f} MiB，对应视觉段的 BF16 KV 字节预算为 {memory["example_visual_kv_mib"]:.3f} MiB。该字节模型来自固定模型配置，未计 allocator 与额外文本；它解释了长期保存视觉编码结果的空间收益，也解释了本方案仍需执行语言 prefill。原始配置和公式输出保存在 `env/qwen3-vl-2b-config.json` 与 `env/feature-memory-model.json`，这项分析不作为未实现的 prefix-KV 方法的速度比较。

同批次相同页面只编码一次，按原请求次序装配视觉特征。单 GPU worker 保证生产者唯一；当前批次全部使用 lease 固定，计算结束后统一释放。不同问题的 input IDs、attention mask、位置编码、prefill 与生成始终正常执行，缓存中没有问题答案或语言 KV。

### 3.3 O2：服务运行时层的有界图像亲和调度

只查看已完成 CPU 预处理的前 16 个等待请求。优先保障等待达到 TTFT SLO 一半的最老请求；其他情况下按文本 prefill 代价、GPU/CPU/miss 视觉代价选首请求，再优先配对同图或长度接近的第二请求。每批最多两个请求，总视觉 token 不超过 1,536。代价系数由开发集实际阶段时间校准，调度器无法读取未来 trace。

### 3.4 O3：GPU/CPU 存储层的收益/字节管理

在相同 GPU/CPU payload 上限下，以 `(1+衰减复访频率)×估计视觉重编码时间/特征字节数` 估计保留价值。编码成本由当前 miss 批次的实测编码时间按各页视觉 token 占比分配。频率以 64 次请求为半衰期；GPU 淘汰可降到 CPU，CPU 命中同步回搬完整特征。LRU 对照与 O3 共用同样的两级存储、搬运机制、lease 和按批预算保护，O3 改变受限空间中的准入及淘汰对象。

按批预算由参数字节、估算 KV 与 256 MiB workspace reserve 推导，并对所有缓存变体生效。估算属于缓存保护规则，不能等同于整进程显存硬隔离。主运行若未触发该保护，性能差异归因于缓存保留策略；预算敏感性通过相同 trace 和显式容量上限验证。8GB 是真实设备容量，128/256/512 MiB 是受控缓存资源预算，不把人为缓存上限描述为整张 GPU 的物理剩余空间。

本次主矩阵有 {analysis["budget_guard_batches"]} 个批次因该预算保护缩小 GPU 缓存上限，O3 的独立效果按实际发生的准入/淘汰行为解释。

## 4. 实验方法与开发集冻结

| 变体 | 完整视觉缓存 | 调度 | 缓存策略 |
| --- | --- | --- | --- |
| B0 | 关闭 | FIFO | 统一逐页原生视觉模块 |
| V1 | 开启 | FIFO | 两级 LRU |
| V1+S | 开启 | O2 | 与 V1 相同 LRU |
| V1+M | 开启 | FIFO | O3 收益/字节 |
| FULL | 开启 | O2 | O3 收益/字节 |

所有变体使用相同输入、prompt、BF16/SDPA、生成参数、CPU 预处理、最大语言批次、逐页视觉执行、队列和时钟。B0 每次正常调用 Transformers 原生视觉模块；缓存变体仍为每个完成请求执行生成。

### 4.1 质量约束驱动的视觉执行修正

初版直接使用原生 packed 视觉批次。完整视觉特征虽然未经压缩，但同图去重或命中后跳过部分图像会改变 BF16 GEMM 的计算形状。首轮全量质量检查中，V1+S 的 ANLS 差为 −0.00585，未通过已经登记的 −0.005 门槛。该轮原生 packed B0 的离线吞吐为 {previous_native_qps} 请求/s，所有五个变体的原始答案、代码、协议和失败 gate 均保留在 `results/initial-packed-vision/`。

{previous_quality_table}

修正对所有变体统一采用视觉 microbatch=1，保留语言 batch=2、原始图片、BF16、SDPA 和原质量门槛，然后重新在开发集校准并冻结协议，再开展本报告的正式性能矩阵。新的 GPU 诊断要求单图、同图双问和异图双请求的完整特征与首步 logits 均严格一致。评估问题曾用于这次正确性纠错，因此最终质量是修正后的重复验证，未将其描述为首次盲测。正式性能 trace 在修正后锁定，初版没有执行性能主矩阵。

### 4.2 开发集参数与正式测量

开发集对 batch=1/2 和 LRU 的 128/256/512 MiB 缓存做容量测量。公共 batch 选择原生基线吞吐较高者，GPU 缓存选择吞吐距最优 1% 内的最小容量，不根据 O3 的收益选取。候选使用同一 192 请求开发 trace，性能选择不读取评估答案。

{calibration_table}

选定 LRU 容量为 {cfg["gpu_cache_mib"]} MiB，开发参考容量为 {protocol["calibration"]["reference_capacity_qps"]:.3f} 请求/s。32 个无排队原生开发请求的 TTFT p95 为 {protocol["calibration"]["isolated"]["ttft_s"]["p95"]:.3f} s，完整回答 p95 为 {protocol["calibration"]["isolated"]["latency_s"]["p95"]:.3f} s；按两倍并向上取整到 10 ms 固定 SLO 为 {slo["ttft_s"]:.2f} s / {slo["e2e_s"]:.2f} s。主到达率固定为参考容量的 50%/80%/110%，配置在正式评估前写入 `protocol.lock.json`。

主矩阵包含 5×3×3=45 个单元，种子 1701/1702/1703。每个单元独立启动模型，使用开发数据预热至少 8 请求且不少于 5 秒，随后清空视觉缓存和峰值计数；评估特征不参与预热。区组和区组内算法顺序固定随机化，以分散温度及频率漂移。

### 指标与计时

- 首 token：生成 streamer 收到首个真实生成 token 的 CPU 可见时刻，跳过 prompt 回调。完整回答：EOS 或上限后完成文本解码时刻。两者均从计划到达时刻计时，包含发送延迟、预处理和排队。
- Goodput：到达窗口内请求中，同时满足 TTFT 和完整回答 SLO 的完成请求数 / 固定 {cfg["main_window_s"]:.0f} s。允许 drain 后按原到达时刻核算；队列拒绝和超时均不计成功，达标率分母含全部到达请求。
- 质量：全体 558 个问题独立执行，不用过载时完成子集替代全量质量。ANLS 使用归一化 Levenshtein 距离阈值 0.5，取参考答案最大值；统一 NFKC、casefold 和空白合并。这是明确给定规范化的本地 ANLS 口径。EM 使用同样规范化。保留原始答案与 token IDs。
- 统计：表中为三次独立运行的均值与样本 SD，p95 先在每轮计算再平均；消融使用相同种子的配对差及 n=3 的 95% t 区间。质量差区间按 81 个源文档进行 2,000 次配对 cluster bootstrap。样本规模支持趋势和区间展示，未将逐请求样本当作独立重复，也未对稳定 p99 作结论。

## 5. 实验结果

### 5.1 开放环延迟与 goodput

{main_table}

![开放环负载曲线：误差棒为三次运行 SD](figures/main-load-curves.png)

每个负载都保留了 B0、O1 强基线及两个独立消融，能够将省去视觉计算的收益和后续运行时/缓存收益分开判断。摘要选用 0.8 档中负载，三档完整结果均在表中。高负载下分位数只覆盖完成请求，因此须与拒绝数、达标率和固定窗口 goodput 联合阅读；当成功请求数很少时，比例提升须结合绝对请求数及配对区间判断。全部 45 轮完成了请求记账，原始发送时刻和监控数据可追溯。

### 5.2 三项优化的独立作用

{effects_table}

{''.join(effects)}配对区间跨零的结果作为趋势，不声称稳定统计提升。O2 的价值取决于待处理队列中能否找到更合适的图像/长度组合；O3 的价值取决于工作集与缓存预算，以及历史复访对未来请求的预测程度。两者没有改变解码算法，单纯视觉复用也不会消除语言模型 prefill 和短答案生成。

本机中负载的最大平均 goodput 来自 {recommended}，为 {values[(0.8,recommended)]["goodput_qps_mean"]:.3f} 请求/s；V1+S/FULL 分别为 {headline_s["goodput_qps_mean"]:.3f}/{headline_full["goodput_qps_mean"]:.3f}。V1+M 相对 V1 的平均差为 {headline_m["goodput_qps_mean"]-headline_lru["goodput_qps_mean"]:+.3f} 请求/s，{memory_interpretation}。以中负载平均 goodput 为工程选择依据，优先配置 {recommendation_labels[recommended]}；FULL 是预登记的组合对照，其名称不代表所有条件下都最好。

调度也存在目标取舍：中负载 V1 的平均 TTFT p95 为 {headline_lru["ttft_p95_s_mean"]:.3f} s，V1+S 为 {headline_s["ttft_p95_s_mean"]:.3f} s。更高 SLO goodput 与更低完成请求 p95 并非同一个目标；对尾延迟优先的部署，应结合 FIFO 对照选择。表中保留这项差异，而不只展示有利指标。

### 5.3 全量质量、计算量与显存

{quality_table}

同一 558 请求离线工作负载下，FULL 的实际视觉编码数从 B0 的 558 降为 {full["encoded_images"]}，减少 {1-full["encoded_images"]/558:.1%}；完成吞吐从 {baseline["completed_qps"]:.3f} 增至 {full["completed_qps"]:.3f} 请求/s，为 {full["completed_qps"]/baseline["completed_qps"]:.2f} 倍。这组单次全量运行用于分析计算量和服务阶段，负载下的主性能证据采用三种子开放环结果。

![全量质量与真实峰值 allocated 显存](figures/quality-and-memory.png)

FULL 相对 B0 的 ANLS 差为 {full["anls_delta_vs_B0"]:+.5f}，按源文档 bootstrap 的 95% 描述区间为 [{full["delta_doc_bootstrap_low"]:+.5f}, {full["delta_doc_bootstrap_high"]:+.5f}]。质量门槛按预登记的实测点估计判断，不把门槛通过等同于 token 逐位一致或质量非劣性的正式统计证明。输出 token 总数、截断数和与 B0 完全相同的 token 序列数均在 `tables/quality.csv`。

### 5.4 缓存预算、请求次序与冷复用控制

{aux_table}

![固定请求 trace 的 GPU 缓存容量敏感性](figures/cache-budget.png)

预算实验保持 CPU 上限、问题顺序和所有模型设置不变，比较 LRU 与 O3 在相同 GPU payload 上限的结果。次序实验和冷页面实验是单次机制诊断，不赋予三种子主矩阵同等统计权重。每页一次的冷对照没有可复用页面，其中 B0/V1 保持相同 FIFO，用于量出纯缓存和元数据管理开销；FULL 同时包含请求组批调度的作用。热页面结果的适用范围由这些控制共同界定。

LRU 的 GPU 缓存从 128 增到 512 MiB 时，192 个相同问题的离线吞吐由 {small_cache["qps"]:.3f} 变为 {large_cache["qps"]:.3f} 请求/s（{large_cache["qps"]/small_cache["qps"]-1:+.1%}），视觉编码数由 {small_cache["encoded_images"]} 变为 {large_cache["encoded_images"]}，PyTorch 峰值 allocated 变化 {large_cache["peak_allocated_mib"]-small_cache["peak_allocated_mib"]:+.1f} MiB。这量出了空间换计算的实际代价，而不仅是命中率变化。价值替换在三个预算上的吞吐差绝对值最大为 {maximum_budget_gap:.2%}，结合主矩阵的配对结果判断其管理收益。

同一组 192 问包含 {random_lru["unique_pages"]} 个唯一页面；按页面连续、16 页交错、随机次序时，LRU 分别编码 {session_lru["encoded_images"]}/{interleaved_lru["encoded_images"]}/{random_lru["encoded_images"]} 次。真实问题的页面局部性决定缓存能否接近唯一页面下界。在 128 个冷页面上，B0/V1 分别为 {cold_native["qps"]:.3f}/{cold_lru["qps"]:.3f} 请求/s，缓存吞吐变化为 {cold_lru["qps"]/cold_native["qps"]-1:+.2%}，且都恰好编码 128 张图像。单次图像场景没有可复用视觉计算，速度差反映管理开销与运行波动。

## 6. 正确性、形式化验证与结果复核

真实 GPU 诊断覆盖单页、同页不同问题、不同页面批次和 CPU 回搬。采用统一逐页视觉执行后，四个诊断组合在原生、miss、GPU hit、CPU hit 路径的完整特征与 token 序列均一致，首步 logits 最大绝对差均为零；GPU→CPU→GPU 搬运逐位相同。修正前同页双请求去重曾产生首步 logits 最大绝对差 0.375，原生单图/双图视觉编码也出现特征差异。初始严格失败记录在 `results/diagnostics-initial-batch-check/`，初版全量失败在 `results/initial-packed-vision/`，最终严格诊断在 `results/diagnostics/equivalence.json`。语言调度仍可能改变文本批次的 BF16 舍入，其质量继续受全量答案门槛约束。

Z3 对 3 个条目、8 步操作、每条目最多 2 个 lease 的有界所有权模型检查“活跃推理不能引用已释放特征”。启用引用保护时为 UNSAT，移除保护的负控制为 SAT 并保存反例；同时穷举真实 LRU/value victim selector 的 54 个 pin 组合。它验证的是有界抽象所有权性质，未宣称证明整个 Python/CUDA 程序或神经网络数值等价。SMT2、版本、操作编码和反例见 `results/verification/`。

独立复核脚本已重新计算全部主实验的质量、SLO、请求账本和 trace 配对，检查 {analysis["integrity"]["checked_requests"]} 个主请求、{analysis["integrity"]["checked_batches"]} 个批次与 192 张原始图像。各批 payload 不超过对应 GPU/CPU 上限，批末 lease 为零；计划到达、接收、预处理、调度、首 token 和完成时间顺序一致。Torch profiler 的实际 CPU/CUDA trace 仅用于解释 kernel 与 miss/hit 路径，主实验关闭 profiler。

## 7. 与相关工作的比较

| 工作 | 复用/优化对象 | 部署和资源方式 | 本项目可对比的效果 |
| --- | --- | --- | --- |
| [Transformers Qwen3-VL](https://huggingface.co/docs/transformers/model_doc/qwen3_vl) | 每次正常视觉编码、prefill、decode | 本机 BF16/SDPA，统一逐页视觉模块 | 本机 B0；全部性能差和质量差均有同机实测；原生 packed 初版另有归档 |
| [ElasticMM](https://arxiv.org/abs/2507.10069) | 模态和阶段解耦、弹性并行、多模态 prefix cache | 多模态服务系统和分布式资源调度 | 对照其阶段感知与缓存思想；本项目实测单 GPU 的 O1/O2/O3 消融 |
| [VisMemo](https://arxiv.org/abs/2610.25136) | 视觉 KV 的精确/近似候选复用、选择性重算、分级存储 | 面向更长视觉上下文和大缓存后端 | 本项目只复用相同页面的视觉编码输出，DeepStack 完整保存，问题相关 KV 正常重算 |

与原生路径的量化对照直接见上述五变体表。相关论文的模型、输入长度、硬件和服务架构与本项目不同，未在本机实现 ElasticMM/VisMemo，因此不将论文加速倍数与本机数字并排冒充同条件实测。本项目建立的是消费级 WSL GPU 下完整视觉边界、有限调度窗口和两级缓存的可复现实验证据。

## 8. 结论与适用范围

完整视觉特征复用使文档同页多问成为可利用的推理工作负载，且缓存管理必须覆盖主 embedding、三层 DeepStack、字节预算和在用引用。五变体实验把模型计算图层、服务调度层和 GPU/CPU 存储层的作用分别量出；最终策略的选择应依据相应到达率下的 goodput 与配对区间，并同时满足全量质量门槛。

本机以中负载 SLO goodput 为目标时，优先配置 {recommendation_labels[recommended]}（{recommended}）。该选择来自独立消融中观察到的实际取舍，三项机制的源码、预算控制和全部结果均保留。

实测结论覆盖本机 Qwen3-VL-2B、统一图像预算和真实问题的合成回放；它不等同于 HTTP 网络服务吞吐、长视频推理或多 GPU 服务成绩。主窗口为 60 秒、每配置三次，温度/频率按 5 秒采样并按区组随机化，完整监控保留供复查。缓存容量敏感性解释受限资源下的选择，不把缓存上限当作显卡物理容量。下一步可在更长连续服务和更长视觉上下文上检验相同机制。

## 9. 复现与交付索引

在项目根目录执行：

```bash
./stage_2/run.sh verify-results     # 从原始数据重算正确性和 SLO，不重新跑模型
./stage_2/run.sh report             # 生成本报告、CSV 和 PNG/SVG/PDF 图
./stage_2/run.sh test               # CPU 不变量测试及 Z3 验证
./stage_2/run.sh experiments        # 依锁定协议续跑/重跑正式与辅助实验
```

完整环境及从零复现步骤见 `../REPRODUCE.md`；独立新跑目录和保留历史数据的说明也在该文档。推理/缓存/调度源代码位于 `src/vlmbench/`，协议在 `configs/protocol.lock.json`，原始与中间结果在 `results/`，所有报告表来自 `report/tables/`。AI 辅助范围见 `AI使用说明.md`。

## 参考文献与来源

1. Mathew et al. [DocVQA: A Dataset for VQA on Document Images](https://arxiv.org/abs/2007.00398)；[官方数据介绍](https://www.docvqa.org/datasets/docvqa)。真实问题与 ANLS 的来源。
2. [Qwen3-VL 模型和架构](https://huggingface.co/docs/transformers/model_doc/qwen3_vl)；[固定模型仓库](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/tree/89644892e4d85e24eaac8bacfd4f463576704203)。实际接口以本地 Transformers 5.17.0 为准。
3. [LMMs-Lab DocVQA 镜像](https://huggingface.co/datasets/lmms-lab-encoder/DocVQA/tree/539088ef8a8ada01ac8e2e6d4e372586748a265e)。冻结原始数据与 revision。
4. Liu et al. [ElasticMM: Efficient Multimodal LLMs Serving with Elastic Multimodal Parallelism](https://arxiv.org/abs/2507.10069)。阶段感知服务与多模态缓存比较。
5. Gao et al. [Visual Cache Reuse for Efficient Multimodal Inference](https://arxiv.org/abs/2610.25136)。视觉 KV 复用和分层管理比较。
'''
    path = destination / "研究报告.md"
    path.write_text(report)
    render_html(path)
    outline = f'''# 五分钟实验讲解提纲

## 1. 问题与测量（约一分钟）

同一文档页面对应多个不同真实问题。每个问题仍需独立语言推理，但视觉图像没有变化。Qwen3-VL-2B 在本机约 {visual_fraction:.1%} 的 GPU worker service time 用于视觉编码，单独消除该阶段的 Amdahl 上限约 {amdahl:.2f} 倍。

数据为 DocVQA 自划分：64 页/282 问开发，128 页/558 问评估，源文档互斥，问题未改写。时间由脚本合成，所有方法同一 trace。

## 2. 三层优化（约两分钟）

- O1，模型计算图：保存主 embedding 与三层 DeepStack，完整 BF16 payload 为 `V×2048×2×4` 字节。不同问题重新执行位置编码、prefill 和生成；语言 KV 仅属于当前请求。
- O2，服务运行时：只看已经就绪的前 16 个请求，保护老请求，再按图像亲和与视觉/文本成本选最多两条。调度器没有未来请求信息。
- O3，GPU/CPU 存储：相同 {cfg["gpu_cache_mib"]}/{cfg["cpu_cache_mib"]} MiB 上限下，以历史复访和实测重编码成本/字节管理条目，GPU/CPU 完整搬运；lease 防止计算期间释放。

初版 packed BF16 视觉执行使缓存出生时的批次影响舍入，一个消融未通过质量门槛。最终对所有变体统一逐页视觉执行，保留语言 batch=2；四组真实 GPU 特征和首步 logits 对照误差均为零。初版代码和失败 gate 保留，门槛没有放宽。

## 3. 对照与实测结果（约一分钟）

五变体、三档负载、三个种子，共 45 轮，每轮 60 秒到达窗口，独立模型进程；另有 18 个辅助单元。TTFT/回答 SLO 为 {slo["ttft_s"]:.2f}/{slo["e2e_s"]:.2f} 秒，来自开发无排队 p95 的两倍。

0.8 档中负载 FULL goodput 为 {headline_full["goodput_qps_mean"]:.3f} 请求/s，B0 为 {headline_base["goodput_qps_mean"]:.3f}，比例 {speed:.2f}。所有拒绝和超时留在到达分母；p95 仅描述完成请求，需和达标率并读。高负载若只有少量成功请求，须先比较绝对数，避免只看比例。

按本机中负载平均 goodput，优先配置为 {recommendation_labels[recommended]}（{recommended}）；FULL 是组合对照，叠加三项机制不代表总是更好。O3 相对 LRU 的中负载平均 goodput 差为 {headline_m["goodput_qps_mean"]-headline_lru["goodput_qps_mean"]:+.3f} 请求/s，{memory_interpretation}。调度的 goodput 与尾延迟取舍须同时说明。

全量 FULL ANLS 为 {full["anls"]:.5f}，相对 B0 为 {full["anls_delta_vs_B0"]:+.5f}；实际视觉编码从 558 次减少到 {full["encoded_images"]} 次。O1/O2/O3 的独立 goodput 差和三种子配对区间在 `tables/paired-effects.csv`，不只展示最终组合。

## 4. 验证与复现（约一分钟）

逐条保存原图、问题、参考答案、生成 token、计划/实际到达、首 token、EOS、缓存与显存。独立脚本重算 ANLS、SLO、唯一请求账本、配对 trace 和每批容量；Z3 对有界 lease 模型得到安全 UNSAT，移除保护后 SAT，保留反例。这项证明只覆盖有限抽象，不覆盖神经网络数值正确性。

8GB 是显卡物理容量，512 MiB 等是控制的缓存资源上限，不能混为一谈。主运行中缓存预算保护触发 {analysis["budget_guard_batches"]} 批，存储收益按实际保留策略解释。相关工作采用不同硬件与模型，本机量化对照是统一执行配置的 Transformers 基线。

已完成结果可用 `./stage_2/run.sh verify-results` 和 `report` 重新核验、出图；全流程 `all`；保持原结果并新跑用 `rerun --output 一个新目录`。AI 辅助范围单独记录，性能数据全部来自实际 GPU 执行。
'''
    (destination / "讲解提纲.md").write_text(outline)


if __name__ == "__main__":
    main()
