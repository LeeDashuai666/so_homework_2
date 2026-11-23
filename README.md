# 系统优化作业 2 · 面向文档视觉问答的跨层推理优化

以 DocVQA 的同页多问为负载，在 WSL2 和 RTX 4060 Laptop GPU 上优化 Qwen3-VL-2B-Instruct 推理。应用层复用视觉特征和同页共享前缀 KV，GPU 执行层使用静态 KV 和 CUDA Graph。独立评估对照原生实现、视觉缓存基线和 vLLM 0.32.0。

| 交付项 | 入口 |
| --- | --- |
| 主报告与设计文档 | [Markdown](stage_2/docs/设计文档.md) · [Word](stage_2/docs/设计文档.docx) |
| 研究结果 | [六路径开发消融](stage_2/experiments/cross_layer_v2/开发验证报告.md) · [vLLM 独立对照](stage_2/experiments/framework_compare/框架对比实验报告.md) · [第一轮研究报告](stage_2/report/研究报告.md) |
| 源码 | [优化原型](stage_2/experiments/cross_layer_v2/runtime.py) · [框架适配器](stage_2/experiments/framework_compare/adapters.py) · [第一轮运行时](stage_2/src/vlmbench/) |
| 环境和配置 | [Python 依赖锁](stage_2/requirements.lock) · [模型、数据和主实验配置](stage_2/configs/) · [vLLM 镜像、环境与冻结配置](stage_2/experiments/framework_compare/README.md#固定镜像和资源) |
| 原始和中间数据 | [DocVQA 输入及来源](stage_2/data/README.md) · [开发逐请求结果与 CUDA trace](stage_2/experiments/cross_layer_v2/results/pilot/) · [独立评估逐请求记录](stage_2/experiments/framework_compare/results/formal/) |
| 结果数据 | [独立评估 CSV 和 JSON](stage_2/experiments/framework_compare/results/analysis/) · [第一轮结果](stage_2/results/) |
| 形式化验证 | [Z3 模型、求解结果和复现](stage_2/experiments/cross_layer_v2/formal/README.md) |
| AI 辅助 | [AI 使用说明](stage_2/report/AI使用说明.md) · [本地 AI 选参实验](stage_2/experiments/ai_cache_tuning/README.md) |
| 复现 | [环境准备、测量与复算步骤](stage_2/REPRODUCE.md) |

独立评估覆盖 64 页、269 道真实问题和 9,720 条计时记录。热页场景中，视觉缓存基线与跨层原型的平均端到端耗时分别为 436.0 和 231.6 ms，原型加速 1.88×。vLLM 的 V11、V11E 对照分别为 176.7 和 120.9 ms，所有配置通过既定质量门槛。Z3 验证得到 9 项安全查询 UNSAT、9 项故障对照 SAT，36 份 SMT2 可重放。AI 选参选择了与固定策略相同的容量，实验记录了该结果及决策开销。

## 快速复算

在仓库根目录执行。需 Linux x86_64 或 WSL2、Python 3.12 和锁定依赖。复算已保存结果时无需 GPU、模型权重或 Docker。

```bash
python3.12 -m venv stage_2/.venv
stage_2/.venv/bin/python -m pip install -r stage_2/requirements.lock
source stage_2/env.sh
python stage_2/tools/check_submission.py
```

单项复算、GPU 重新测量、Z3 及 AI 选参的入口与前置条件见[复现说明](stage_2/REPRODUCE.md)。原始图片、问题、模型输出和计时记录随仓库提供。模型权重和 vLLM 镜像按固定版本另行准备。
