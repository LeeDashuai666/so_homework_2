# AI 使用说明

本次作业使用 Codex 辅助检查实验计划、确认环境、编写测试与审阅设计文档。另有一项使用本地 Qwen3-0.6B 的视觉缓存容量选择实验。

## 1. 完善实验计划并补充验证

我以 DocVQA 的同页多问为负载，计划复用视觉特征和公共前缀 KV，并用 CUDA Graph 优化解码。Codex 辅助检查这套方案能否在一张 8 GB 显存的 GPU 上运行，结合前缀 KV 的存储开销和 CUDA Graph 对形状、地址稳定性的要求，检查方案的适用条件。

针对方案里的关键假设，补充了异页不应命中、同页不同问题的请求状态隔离、静态 KV 容量边界和 CUDA Graph 重放检查。实验计划还加入了逐项消融、CUDA trace、不同页面访问顺序的对照以及同一模型的 vLLM 对照。这些实验分别用于判断每项优化的作用，并检查效果是否随访问方式变化。[实验方案](../plan/推理框架对比实验方案.md) · [开发验证报告](../experiments/cross_layer_v2/开发验证报告.md)

## 2. 检查环境是否可用

Codex 辅助核查 WSL2 中的 Python、PyTorch、Transformers、CUDA 和 NVIDIA 驱动，检查现有权重能否离线加载，以及基础图文问答和 CUDA Graph 是否可运行。vLLM 对照复用了本机已有的 0.32.0 Docker 镜像。环境可用性以本地探测输出、烟测和实际推理记录为依据。[环境清单](../env/实验配置清单.md) · [vLLM 环境](../experiments/framework_compare/README.md#固定镜像和资源)

## 3. 辅助编写测试代码

Codex 辅助编写环境烟测、运行时检查、对照测量和数据复算脚本。运行时检查覆盖缓存命中、状态隔离、容量边界和解码输出一致性。测量脚本保存逐请求答案、延迟、缓存计数和 GPU 采样，复算脚本按固定口径从这些记录检验结果。AI 也辅助实现 Z3 的性质检查、故障对照和 SMT2 重放脚本。实测数值由本地推理获得，形式化查询由 Z3 求解。[运行时检查](../experiments/cross_layer_v2/verify_runtime.py) · [原始数据复算](../tools/check_submission.py) · [形式化验证](../experiments/cross_layer_v2/formal/README.md)

## 4. 审阅我手写的设计文档

Codex 辅助梳理我手写的文档，对照代码、实验表格和原始记录查找遗漏与表述错误。检查重点包括计时起点是否一致、加速比是否引用同一组对照、开发消融与独立评估是否区分清楚，以及形式化结论是否写明假设和适用范围。设计文档第 4 章记录了实验方法和结果，第 5 章说明适用范围。[设计文档](设计文档.md)

## 5. 本地 AI 选参实验

实际运行 Qwen3-0.6B，根据开发集统计在 1024 和 16384 token 两档 vLLM 视觉缓存容量中选择一档。选参后用已有的独立 GPU 实测记录评价所选配置。模型选择了 1024 token，与固定策略相同，两种策略对应的平均端到端耗时均为 176.7 ms。本次选参没有带来加速。16384 token 档位作为事后参照，在同一热页负载下的均值为 120.9 ms。

本地模型使用 CPU BF16、2 个线程、PyTorch 2.14.0+cu130 和 Transformers 5.17.0。一次模型前向约 14.8 秒，从启动决策进程到保存结果约 30 秒。权重版本、文件哈希、完整提示词、候选 logits 和实际输出均已保存。复算运行 `./stage_2/experiments/ai_cache_tuning/run.sh`，重新执行模型决策时另需锁定的本地权重。[实验说明与复现步骤](../experiments/ai_cache_tuning/README.md) · [权重锁](../experiments/ai_cache_tuning/policy-model.lock.json) · [决策记录](../experiments/ai_cache_tuning/results/replay/decision.json)
