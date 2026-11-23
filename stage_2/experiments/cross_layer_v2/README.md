# 第二轮开发验证入口

题目：**面向文档视觉问答的跨层推理优化——多模态前缀状态复用与 CUDA Graph 解码。**

当前完成的是开发集验证。第一轮正式数据、报告、协议、源码和提交包保留；本目录的结果尚未替代最终报告。

- [经验与第二轮方案](../../plan/经验总结与第二轮优化.md)
- [开发验证报告](开发验证报告.md)
- [原始参数和测量代码哈希](results/pilot/manifest.json) · [审计和统计](results/pilot/audit-and-analysis.json)
- [GPU 状态与容量检查](results/runtime-check.json)
- [公共前缀KV与CUDA Graph形式化验证](formal/README.md)：9项安全性质UNSAT、9项故障负控制SAT，36份SMT2独立重放。

## 软件和硬件层次

1. **模型计算复用层：多模态公共前缀 KV。** 只复用 `<vision_end>` 之前的固定头部和页面，后续问题与答案独立；命中时减少语言 prefill。
2. **CPU–GPU/GPU 执行层：静态 KV + CUDA Graph。** 固定缓冲地址，以单次图发射替代每 token 的大量 host kernel 发射，保留真实 EOS 和模型计算。

视觉缓存作为已有强基线；它和前缀缓存不能重复计为两个不同层次。六条执行路径包括原生 HF、HF+视觉缓存和四个静态 KV 消融，防止只用较慢的新参考实现证明加速。

## 前置条件

在本机已有项目环境和固定资产上执行，无需重新安装依赖：

- `stage_2/env.sh` 可激活 `.venv`；PyTorch 2.14.0+cu130、Transformers 5.17.0、支持 CUDA Graph 的 NVIDIA 驱动。
- RTX 4060 Laptop 8GB 或有足够显存的 CUDA GPU，执行期间无其他训练/推理任务。
- `models/qwen3-vl-2b-instruct` 指向固定模型，锁在 `configs/vlm-assets.lock.json`；模型约 4.26GB。
- `data/processed/docvqa/development.jsonl` 和其中引用的本地 PNG 已备齐，使用原开发集 64 页 282 问。
- `configs/protocol.lock.json`、`tools/vlm_runtime.py`、`src/vlmbench` 保持第一轮版本，参数和源码哈希在结果中记录。

资产缺失时，按[项目复现说明](../../REPRODUCE.md)先完成资源准备；不需要重跑第一轮 63 个实验单元来执行本目录的验证。GPU 代码针对本机已安装接口，不把未测试的软件版本称为可复现环境。

## 一键和逐步执行

从仓库根目录执行。读取已有结果、从逐请求记录重新计算质量与计时，并从两个真实 CUDA trace 重数 host 发射调用：

```bash
./stage_2/experiments/cross_layer_v2/run.sh report
```

独立 GPU 状态检查：

```bash
./stage_2/experiments/cross_layer_v2/run.sh check \
  --output stage_2/experiments/cross_layer_v2/results/runtime-check-new.json
```

重跑开发实验与全部检查，输出到**新的目录**：

```bash
./stage_2/experiments/cross_layer_v2/run.sh all \
  --output stage_2/experiments/cross_layer_v2/results/replay-new
```

该命令依次执行状态/边界检查、282 问×六路径质量门槛、16 页 57 问×六路径×三个配对区组计时、八步真实 decode profiler、数据审计、Markdown/HTML/CSV 和 PNG/SVG/PDF 生成。不覆盖已存在的目录；耗时和 GPU 温度取决于设备，进度直接输出。

也可拆开执行：

```bash
./stage_2/experiments/cross_layer_v2/run.sh pilot \
  --output stage_2/experiments/cross_layer_v2/results/pilot-new
./stage_2/experiments/cross_layer_v2/run.sh report \
  --input stage_2/experiments/cross_layer_v2/results/pilot-new \
  --report stage_2/experiments/cross_layer_v2/开发验证报告-new.md
```

逐步命令默认使用已经通过的 `results/runtime-check.json`；若重跑了状态检查，给 report 增加 `--runtime-check 新检查路径`。完整报告要求全部开发问题和预注册的三个计时区组，缩小参数仅用于接口诊断。

## 原始和中间数据

`results/pilot` 保留测量时的 runtime/pilot/smoke 源码 SHA256、全部答案与 token、逐请求生成/TTFT/decode 时间、每区组顺序和结果、图构建费用、实际 profiler trace、质量门槛、审计与 CSV。报告从这些数据生成；不删去慢区组，也不按输出筛选问题。

`verify_runtime.py` 检查真实页面间的前缀身份与存储隔离、跨请求状态、图地址稳定、驱逐后 miss，并用单独合成长后缀检查 960-token 输入加 64-token 输出的静态容量边界。这个合成输入只做正确性检查，不加入性能/质量分数。

热页面结果表示预热后的 B=1 生成阶段延迟，不包含排队和图像预处理。图捕获 6.29s 的启动费用、前缀约 7 倍的状态体积和首次 miss 的代价均单独报告。新正式评估、全局随机/低复用、B=2 和冷启动矩阵按计划另行冻结执行。

实现和报告使用 AI 辅助；AI 不提供生成答案的外部服务，也不替代 Qwen3-VL 的本地实际推理。AI 生成的代码和叙述由开发集门槛、原始日志审计、GPU不变量检查及下述补充形式化验证检验。第一轮Z3的结论单独对应第一轮缓存模型。

2026-11-18已补充针对当前前缀和图执行实现的源码关联Z3验证，入口为`./stage_2/experiments/cross_layer_v2/run.sh formal`；它证明明确接口前提下的状态与数据流性质。上述开发实验当时未包含该证明，补充结果单独保存于`formal/results/`，不改动原测量记录。新的`all`流程会先生成形式化证据，再执行GPU测量。
