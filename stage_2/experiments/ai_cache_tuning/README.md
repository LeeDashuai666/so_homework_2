# AI 辅助视觉缓存容量选择

Qwen3-0.6B 根据开发数据执行了一次离线选参。它选择 1024 token，与固定策略相同。这次试验耗费约 30 秒 CPU 进程时间（单次前向 14.77 秒），新增 GPU 启动 0 次。GPU 仍被另一个工作负载占用，因此使用第 4.7 节已留存的真实 vLLM 请求记录验证选择。

## 协议与结果

固定策略沿用 1024 token。AI 在 1024 与 16384 token 中选一档。它读取的是 V11 在原开发集 282 问的真实统计，以及按固定哈希从开发集挑出的 16 页图像的 11,880 个视觉 token。开发请求按页连续，218 次回访均未重新编码。提示中提供了目标访问方式为 16 个已预热页面的交错回访，以及 6.75 GiB 的 PyTorch 峰值 allocated 预算。完整输入和提示词在 `results/replay/observation.json` 与 `decision.json`。候选字母的 logits 分别为 A=29.125、B=28.625，贪心选择 A。

记录决策后，使用与开发集源文档互斥的已留存热场景来回放。三个独立进程各 72 道相同问题，共 216 条记录每档。固定策略与 AI 策略选择了同一档位，因此引用同一份记录。16384 档位作为事后参照。ANLS 从原始答案重新计算，门槛为相对固定档位最多下降 0.005。

| 配置 | encoder 容量 / token | 平均 TTFT / ms | 平均端到端 / ms | 平均视觉编码次数 / 72 问 | 峰值 allocated / GiB | ANLS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 固定策略与 AI 策略 | 1024 | 86.3 | 176.7 | 36.7 | 5.78 | 0.9088 |
| 事后参照 | 16384 | 30.3 | 120.9 | 0.0 | 5.96 | 0.9088 |

开发集的连续同页访问和评估集的跨页交错访问不同。开发数据的 0 次回访编码无法直接反映交错访问中的缓存驱逐。结果适用于这个访问场景和一次决策，记录中的端到端延迟均来自已完成的 GPU 实测。

## 复现

在仓库根目录执行以下命令，使用已保存的决策和原始文件复算，无需加载权重或占用 GPU。

```bash
./stage_2/experiments/ai_cache_tuning/run.sh
```

在新目录重新运行 AI 决策并复算。默认使用本机现有的 `/home/lxy18/infer-lab/models/Qwen3-0.6B`，可用 `--model` 指向同一快照的其他目录。

```bash
./stage_2/experiments/ai_cache_tuning/run.sh \
  --model /home/lxy18/infer-lab/models/Qwen3-0.6B \
  --output stage_2/replays/ai-cache-new
```

复现使用 `stage_2/.venv` 中的 Python 3.12、PyTorch 2.14.0+cu130 和 Transformers 5.17.0。选参模型使用 CPU BF16、2 线程、Qwen3 非思考模板。对单 token 候选字母取最大 logit。决策阶段需要约 1.5 GB 权重及本次约 1.69 GiB 的进程峰值 RSS。已有权重、环境和原始记录时可离线运行。模型为 `Qwen/Qwen3-0.6B` 的 `c1899de289a04d12100db370d81485cdf75e47ca` 版本。资产文件及 SHA256 在 `policy-model.lock.json`，提示词、logits、环境和测得用时在 `results/replay/decision.json`。原始评估记录哈希在 `summary.json`，vLLM Docker 镜像和 GPU 环境由第 4.7 节实验协议固定。
