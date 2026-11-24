# 系统优化作业 2 · 多模态推理优化

本项目在 WSL2 和 RTX 4060 Laptop GPU 上测量 Qwen3-VL-2B-Instruct 的 DocVQA 同页多问负载。软件侧保留完整视觉特征，由 Qwen3-0.6B 离线选择缓存参数，再按回访间隔和显存成本决定页面是否驻留前缀 KV。GPU 侧使用连续 KV 工作区与 Triton 融合写回。

| 交付项 | 入口 |
| --- | --- |
| 设计文档 | [Word](stage_2/docs/设计文档.docx) · [Markdown](stage_2/docs/设计文档.md) |
| 本轮实验 | [源码与测量入口](stage_2/experiments/adaptive_kv/) · [环境、配置及原始数据](stage_2/experiments/adaptive_kv/results/main/) · [结果摘要](stage_2/experiments/adaptive_kv/results/main/summary.json) |
| 输入与环境 | [DocVQA 数据](stage_2/data/README.md) · [依赖锁](stage_2/requirements.lock) · [模型和配置](stage_2/configs/) |
| 框架对照与验证 | [vLLM 独立对照](stage_2/experiments/framework_compare/框架对比实验报告.md) · [Z3 状态验证](stage_2/experiments/cross_layer_v2/formal/) |
| AI 与复现 | [AI 使用说明](stage_2/docs/AI使用说明.md) · [复现步骤](stage_2/REPRODUCE.md) · [课程要求对照](stage_2/docs/课程要求自查.md) |

在 320 MiB 缓存预算下，交错回访的平均首 token 延迟从固定前缀 KV 的 293.1 ms 降至 189.6 ms，比固定视觉缓存低 6.0%。按页连续访问时，固定前缀 KV 最快。固定 CUDA Graph 提交方式后，融合写回将每解码步的 GPU kernel 从 1796 减至 1684，解码耗时从 18.05 降至 17.89 ms/token。离线 AI 选中默认成本配置，与该配置持平。按两种访问顺序等权计算，AI 所选配置比按开发集延迟直接排序选出的宽配额低 6.0%。

## 快速复算

在 Linux x86_64 或 WSL2 上，从仓库根目录执行。此流程读入保存的请求记录，并核对 Word、实验数据和 Z3 结果。

```bash
python3.12 -m venv stage_2/.venv
stage_2/.venv/bin/python -m pip install -r stage_2/requirements.lock
source stage_2/env.sh
python stage_2/tools/check_submission.py
```

GPU 实验重跑所需的固定模型版本、步骤与前置条件见[复现说明](stage_2/REPRODUCE.md)。
