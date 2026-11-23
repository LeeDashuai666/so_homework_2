#!/usr/bin/env python3
"""Build design-document figures from audited measurements; inference files are read-only."""
from pathlib import Path
import argparse
import base64
import hashlib
import html
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/assets/design"
PILOT = ROOT / "experiments/cross_layer_v2/results/pilot"
BLUE, GREEN, GRAY, INK = "#346b9a", "#28857d", "#697682", "#26374b"


def save(fig, name):
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / (name + "." + ext), dpi=240, bbox_inches="tight", pad_inches=.1)
    plt.close(fig)


def box(ax, x, y, w, h, text, color="#edf3f9", size=10):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.025,rounding_size=.06",
                              facecolor=color,edgecolor=GRAY,linewidth=.8))
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=size,linespacing=1.4)


def arrow(ax, start, end):
    ax.annotate("",xy=end,xytext=start,arrowprops={"arrowstyle":"-|>","color":GRAY,"lw":1})


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    font_path = Path("/mnt/c/Windows/Fonts/msyh.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        family = font_manager.FontProperties(fname=str(font_path)).get_name()
    else:
        family = "Noto Sans CJK SC"
    plt.rcParams.update({"font.family":family,"font.size":10,"axes.unicode_minus":False,
                         "svg.fonttype":"path","pdf.fonttype":42,"svg.hashsalt":"vlm-design",
                         "axes.spines.top":False,"axes.spines.right":False,
                         "text.color":INK,"axes.labelcolor":INK,"figure.facecolor":"white"})
    audit = json.loads((PILOT / "audit-and-analysis.json").read_text())
    assert audit["passed"] and audit["first_round_frozen_sources_unchanged"]
    means = audit["means"]

    fig,ax = plt.subplots(figsize=(7.3,2.85)); ax.set(xlim=(-.1,10),ylim=(-.3,3.7)); ax.axis("off")
    xs = [.02,2.05,4.08,6.11,8.14]
    routes = [
        ("基础路径",2.7,"#f0f2f5",["图像＋当前问题","视觉编码","完整prefill","逐步eager解码","回答"]),
        ("① 视觉特征命中",1.5,"#edf3f9",["相同页面\n新的问题","取完整视觉特征","完整prefill","逐步解码","回答"]),
        ("② 前缀命中＋③ 图解码",.3,"#e9f4ef",["相同页面\n新的问题","恢复公共前缀KV","当前问题\n后缀prefill","CUDA图decode","回答"]),
    ]
    for row,y,color,texts in routes:
        ax.text(.02,y+.62,row,fontsize=10,color=GRAY)
        for i,(x,text) in enumerate(zip(xs,texts)):
            box(ax,x,y,1.7,.55,text,color,size=9.3)
            if i<4: arrow(ax,(x+1.72,y+.275),(xs[i+1]-.03,y+.275))
    ax.text(5,-.18,"前缀命中直接跳过视觉编码；未命中时正常编码并建立前缀。",ha="center",fontsize=9,color=GRAY)
    fig.tight_layout(pad=.2); save(fig,"overview")

    fig,ax = plt.subplots(figsize=(7.3,2.4));ax.set(xlim=(0,12),ylim=(-.2,4.4));ax.axis("off")
    layers = [
        (3.1,"① 完整视觉特征缓存","跳过重复视觉编码","#edf3f9"),
        (1.8,"② 公共前缀KV复用","跳过公共前缀prefill","#edf3f9"),
        (.5,"③ 静态KV＋CUDA Graph","减少逐token提交开销","#e9f4ef"),
    ]
    for y,title,purpose,color in layers:
        box(ax,.3,y,8.8,.85,"",color)
        ax.text(4.7,y+.58,title,ha="center",va="center",fontsize=11)
        ax.text(4.7,y+.21,purpose,ha="center",va="center",fontsize=9.5,color=GRAY)
    arrow(ax,(4.7,3.05),(4.7,2.7));arrow(ax,(4.7,1.75),(4.7,1.4))
    ax.plot([9.4,9.65,9.65,9.4],[3.95,3.95,1.8,1.8],color=BLUE,lw=1)
    ax.text(10.75,2.87,"模型计算层",ha="center",va="center",fontsize=10.5,color=BLUE)
    ax.plot([9.4,9.65,9.65,9.4],[1.35,1.35,.5,.5],color=GREEN,lw=1)
    ax.text(10.75,.92,"GPU执行层",ha="center",va="center",fontsize=10.5,color=GREEN)
    ax.text(5.8,.04,"前缀命中时跳过①，直接恢复②的状态。",ha="center",fontsize=9,color=GRAY)
    fig.tight_layout(pad=.2);save(fig,"layers")

    fig,ax = plt.subplots(figsize=(7.1,2.75));ax.set(xlim=(0,10),ylim=(0,4));ax.axis("off")
    box(ax,.15,3.05,5.1,.62,"固定头部＋图像＋<vision_end>",size=10)
    box(ax,5.55,3.05,4.25,.62,"当前问题＋assistant生成头",color="#f5efe5",size=10)
    ax.text(5.4,3.86,"前缀边界",ha="center",fontsize=9,color=GRAY)
    ax.plot([5.4,5.4],[2.95,3.68],ls="--",color=GRAY,lw=.8)
    box(ax,.25,1.05,3.05,.9,"各层公共前缀KV\n持久缓存／只读",color="#e9f4ef")
    arrow(ax,(2.3,3.02),(1.77,2.0))
    ax.text(.65,2.3,"首次计算后保存",fontsize=9,color=GRAY)
    box(ax,4.35,1.78,5.35,.7,"请求A工作KV：前缀副本＋问题A＋答案A",size=9.3)
    box(ax,4.35,.38,5.35,.7,"请求B工作KV：前缀副本＋问题B＋答案B",size=9.3)
    arrow(ax,(3.33,1.66),(4.3,2.13));arrow(ax,(3.33,1.32),(4.3,.73))
    ax.text(3.43,2.07,"复制",fontsize=9,color=GRAY);ax.text(3.48,.85,"复制",fontsize=9,color=GRAY)
    ax.text(7,.03,"追加只发生在工作区，两个问题的状态互不污染。",ha="center",fontsize=9,color=GRAY)
    fig.tight_layout(pad=.2);save(fig,"prefix")

    chosen = ["STATIC_EAGER","PREFIX","GRAPH","BOTH"]
    labels = ["静态eager","前缀复用","图解码","组合"]
    colors = ["#b6c1cb","#d7a25a",BLUE,GREEN]
    fig,axes = plt.subplots(1,2,figsize=(7.2,2.55))
    for ax,key,scale,ylabel in [(axes[0],"mean_ttft_s",1000,"首token时间 / ms"),
                                (axes[1],"decode_ms_per_token",1,"解码时间 / ms每token")]:
        values = [means[v][key]*scale for v in chosen]
        ax.bar(range(4),values,yerr=[means[v][key+"_sd"]*scale for v in chosen],color=colors,capsize=3,width=.58)
        ax.set_xticks(range(4),labels);ax.set_ylabel(ylabel);ax.set_ylim(0,max(values)*1.3)
        for i,value in enumerate(values):ax.text(i,value+max(values)*.09,f"{value:.1f}",ha="center",fontsize=9)
    fig.tight_layout(pad=.5,w_pad=2);save(fig,"stages")

    variants = ["NATIVE","VISUAL_HF","STATIC_EAGER","PREFIX","GRAPH","BOTH"]
    labels = ["原生","视觉缓存","静态eager","前缀复用","图解码","组合"]
    fig,ax = plt.subplots(figsize=(7.2,2.55))
    values = [means[v]["mean_generation_s"]*1000 for v in variants]
    errors = [means[v]["mean_generation_s_sd"]*1000 for v in variants]
    ax.bar(range(6),values,yerr=errors,color=["#929da9","#7197b2",*colors],capsize=3,width=.58)
    ax.set_xticks(range(6),labels);ax.set_ylabel("完整生成耗时 / ms");ax.set_ylim(0,650)
    for i,(v,e) in enumerate(zip(values,errors)):ax.text(i,v+e+12,f"{v:.1f}",ha="center",fontsize=10)
    fig.tight_layout(pad=.5);save(fig,"generation")

    trace = audit["trace_diagnostics"]
    counts = json.loads((PILOT / "launch-counts.json").read_text())
    fig,axes = plt.subplots(1,2,figsize=(7.2,2.55))
    host = [sum(counts[v]["runtime_launch_calls"].values()) for v in ["eager","graph"]]
    axes[0].bar([0,1],host,color=[BLUE,GREEN],width=.45);axes[0].set_yscale("log")
    axes[0].set(xticks=[0,1],xticklabels=["eager／kernel发射","graph／图发射"],ylabel="主机发射调用数（对数轴）",ylim=(1,70000))
    for i,v in enumerate(host):axes[0].text(i,v*1.35,f"{v:,}",ha="center",fontsize=10)
    kernels = [trace[v]["kernel_events"] for v in ["eager","graph"]]
    axes[1].bar([0,1],kernels,color=[BLUE,GREEN],width=.45)
    axes[1].set(xticks=[0,1],xticklabels=["eager","graph"],ylabel="GPU kernel事件数",ylim=(0,18000))
    for i,v in enumerate(kernels):axes[1].text(i,v+500,f"{v:,}",ha="center",fontsize=10)
    fig.tight_layout(pad=.5,w_pad=2);save(fig,"launches")

    render_document()


def render_document():
    md = ROOT / "docs/设计文档.md"
    body = MarkdownIt("commonmark",{"html":True}).enable("table").render(md.read_text())
    body = body.replace("<!-- pagebreak -->",'<div class="pagebreak"></div>')
    def embed(match):
        file = md.parent / html.unescape(match.group(1))
        if file.is_file() and file.suffix == ".png":
            return '<img src="data:image/png;base64,'+base64.b64encode(file.read_bytes()).decode()+'"'
        return match.group(0)
    body = re.sub(r'<img src="([^"]+)"',embed,body)
    style = "body{max-width:900px;margin:40px auto;padding:0 28px;font:16px/1.85 'SimSun','宋体',serif;color:#111}h1{text-align:center;font-size:24px;font-weight:normal}h2{font-size:22px;font-family:'SimHei','黑体',sans-serif;margin-top:26px}h3{font-size:19px;font-family:'SimHei','黑体',sans-serif}p{text-indent:2em}img{max-width:100%}p:has(img){text-align:center;text-indent:0}table{border-collapse:collapse;width:100%;font-size:14px;margin:18px 0;border-top:2px solid #111;border-bottom:2px solid #111}th{border-bottom:1px solid #111}th,td{padding:7px 10px;text-align:left}pre{padding:12px 18px;background:#f6f6f6;overflow:auto;line-height:1.5}a{color:#346b9a}@page{size:A4;margin:1.7cm 2cm 1.65cm}@media print{body{margin:0;padding:0;font-size:10.5pt}h1{font-size:15pt}h2{font-size:14pt}h3{font-size:12pt}.pagebreak{break-before:page}h2,h3{break-after:avoid}table,img{break-inside:avoid}}"
    md.with_suffix(".html").write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>作业2设计文档</title><style>'+style+'</style><body>'+body+'</body></html>')
    inputs = [PILOT / name for name in ["audit-and-analysis.json","quality-summary.json","timing-blocks.json","launch-counts.json","manifest.json"]]
    framework_analysis=ROOT/'experiments/framework_compare/results/analysis/analysis.json'
    has_framework_figure='assets/design/frameworks.png' in md.read_text()
    if has_framework_figure:
        assert framework_analysis.exists()
        inputs.extend([framework_analysis,ROOT/'experiments/framework_compare/protocol.lock.json'])
    record = {"source_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
              "source_markdown_sha256":hashlib.sha256(md.read_bytes()).hexdigest(),
              "figure_count":len(re.findall(r'^!\[',md.read_text(),flags=re.M)),
              "scope":"Development mechanism validation and independent framework evaluation" if has_framework_figure else "Development mechanism validation; framework evaluation in progress"}
    (OUT / "数据来源.json").write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"figures":str(OUT),"html":str(md.with_suffix('.html'))},ensure_ascii=False))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--html-only',action='store_true',help='Preserve existing figure assets and render current Markdown only')
    args=parser.parse_args()
    render_document() if args.html_only else main()
