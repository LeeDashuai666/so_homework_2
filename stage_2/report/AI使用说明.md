# AI 辅助使用说明

本项目使用 Codex 辅助代码开发、实验编排、结果分析和文档写作。当前提交主题为“面向文档视觉问答的跨层推理优化——多模态前缀状态复用与 CUDA Graph 解码”，包含第二轮开发消融及独立 vLLM 框架评估；第一轮实验作为历史材料保留。

## 实际辅助范围

- 阅读本地 Qwen3-VL 和推理框架实现，组织完整视觉特征缓存、公共前缀 KV 复用、静态 KV 与 CUDA Graph 三项优化。前两项属于模型计算层，第三项属于 CPU–GPU／GPU 执行层。
- 辅助实现测量、状态隔离检查、开发质量门槛、协议冻结、随机化重复运行、Docker 启动、显存遥测及结果审计脚本。
- 调用本机 GPU 完成六路径开发消融，以及 64 页 269 问、24 次独立启动、72 个区组、9720 条计时请求的框架对照。模型答案和计时记录来自真实执行，未用 AI 编写替代数据。
- 从原始答案和日志重算 ANLS、延迟、标准差及配对区间，生成图表，辅助撰写设计文档和框架实验报告。原型慢于成熟框架的结果、较慢的第三次重复和首次未命中成本均保留。
- 辅助建立与当前源码关联的前缀KV／CUDA Graph符号模型：9项安全性质UNSAT、9项故障负控制SAT，另有18项前提可满足性检查；保存36份SMT2、反例、源码快照和求解器版本。原有GPU状态检查与答案对照继续验证具体设备执行。第一轮有界lease证明单独保留。

本项目另在第 4.8 节实际运行本地 Qwen3-0.6B 作为离线选参模型。模型根据开发集统计从两档 vLLM 视觉缓存容量中选择一档。选择锁定后回放独立测试页的原始 GPU 测量记录。提示词、候选 logits、权重版本和 CPU 运行环境见[AI 选参实验](../experiments/ai_cache_tuning/README.md)。

## AI 工具和实验环境

AI 开发助手为 Codex。本次交付自查记录的本地客户端为 `codex-cli 0.162.1`，工具版本见 `results/submission-audit/checks.json`。该记录表示自查时可用的客户端，不代表所有历史交互使用同一服务端模型版本。交互式助手的服务端模型版本及生成随机性未作为实验变量锁定。

实验推理模型为本地 `Qwen/Qwen3-VL-2B-Instruct`，revision 为 `89644892e4d85e24eaac8bacfd4f463576704203`。AI 助手用于生成和检查代码，不位于测量请求路径中。复现者运行已经交付的确定版本代码，无需重新让助手生成代码，也无需 Codex 账号、云端推理服务或 API key。

| 用途 | 实际配置与清单 |
| --- | --- |
| 宿主推理 | Python 3.12、Torch 2.14.0+cu130、Transformers 5.17.0；`requirements.lock` 与 `experiments/framework_compare/environment-host-effective.txt` |
| 框架对照 | vLLM 0.32.0 固定 Docker digest；Torch 2.13.0+cu130、Transformers 5.19.0；`experiments/framework_compare/protocol.lock.json` 与 `environment-container.txt` |
| 形式化检查 | `z3-solver==5.4.1.0`，Python接口；当前证明在`experiments/cross_layer_v2/formal/`，第一轮证明在`scripts/formally_verify.py`与`results/verification/` |

完整有效包清单还记录了共享环境中未用于本实验的工具；无需为了复现安装其中的定制 FlashAttention 或 nano-vLLM。宿主实际路径使用 SDPA，vLLM 使用固定镜像内的 FLASH_ATTN。

## 复现和人工核查

从仓库根目录执行：

```bash
# 从框架原始记录重算报告与图表，不加载 GPU 模型
./stage_2/experiments/framework_compare/run.sh report
# 从开发日志重算消融和 profiler 证据
./stage_2/experiments/cross_layer_v2/run.sh report
# 本次提交材料自查：CPU 测试、原始记录审计、SMT2 重放、文档哈希核对
source stage_2/env.sh
python stage_2/tools/check_submission.py
# 单独重放当前前缀／图执行证明，不需要GPU
./stage_2/experiments/cross_layer_v2/run.sh formal
```

真实 GPU 重跑分别使用两个实验目录的 `run.sh all`，新目录要求和环境前置条件见 `REPRODUCE.md`、`experiments/cross_layer_v2/README.md`、`experiments/framework_compare/README.md`。AI 开发交互本身不承诺确定性重放；已交付源代码、输入、协议、生成答案和统计计算可以逐项核查。

## 与任务相关的用户要求

用户提出系统优化课程第二项目需围绕 AI 推理，在 WSL 上完成，避免泛化的常见题目；随后要求改为多模态推理，缩减为三到四项优化，复用已有软硬件环境，并最终完成实验。代码和文档据此围绕“相同文档页面的多个独立真实问题”展开。

提交者应阅读并理解缓存边界、DeepStack、KV 独立性、时间戳、SLO 分母和质量门槛的实现与结果，以便独立讲解研究方法和实验结论。
