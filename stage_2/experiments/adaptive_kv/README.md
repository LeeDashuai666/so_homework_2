# 文档视觉问答的缓存选择与 GPU 写回

本实验为 B=1 的 Qwen3-VL-2B-Instruct 推理提供两项对照。软件侧在 320 MiB 持久负载预算内，比较固定视觉特征、固定前缀 KV 和按成本选择表示。GPU 侧固定 CUDA Graph，分别测量原静态 KV、连续恢复、连续恢复与 Triton 写回融合。

## 环境和步骤

- 实测使用 WSL2 和 RTX 4060 Laptop GPU。Python 3.12 依赖见 [锁文件](../../requirements.lock)，核心版本为 PyTorch 2.14.0+cu130、Transformers 5.17.0、Triton 3.8.0。驱动、系统内核和图构建耗时见 [评估环境](results/main/eval-environment.json)。
- 逐问生成使用 Qwen3-VL-2B-Instruct 的固定权重与处理配置。DocVQA 图片及清单使用仓库的固定数据版本。下载及安装步骤见 [项目复现说明](../../REPRODUCE.md)。
- 离线选参使用 Qwen3-0.6B，revision 为 `c1899de289a04d12100db370d81485cdf75e47ca`。CPU BF16、2 个线程、关闭 thinking，输出在候选字母中取最高 logit。设 `STAGE2_POLICY_MODEL` 指向本地权重目录。程序检查 [权重和配置的 SHA256](../ai_cache_tuning/policy-model.lock.json)。

从仓库根目录运行。

```bash
# 读出已保存的 GPU 和 AI 记录，重算结果和 trace 计数
./stage_2/experiments/adaptive_kv/run.sh report

# 在一个新目录完整运行
STAGE2_POLICY_MODEL=/path/to/Qwen3-0.6B \
  ./stage_2/experiments/adaptive_kv/run.sh all stage_2/replays/adaptive-kv-new
```

分步模式为 `dev`、`ai`、`eval`、`report`，第二个参数均为同一个结果目录。开发阶段创建目录；AI 阶段产生冻结决策，评估阶段核对开发记录与协议的哈希。已保存记录可在具备锁定 Python 环境的 CPU 机器复算。

## 数据和结果

| 文件 | 用途 |
| --- | --- |
| [protocol.json](results/main/protocol.json) | 两套数据成员、访问顺序、成本参数、区组和源码哈希 |
| [calibration.json](results/main/calibration.json) | 8 页配对测量，计算前缀热命中收益和建立前缀的成本 |
| [dev-requests.jsonl](results/main/dev-requests.jsonl) | 开发集逐请求数据 |
| [observation.json](results/main/observation.json) · [decision.json](results/main/decision.json) | AI 看到的开发指标、提示词、logits、所选参数与直接排序结果 |
| [eval-requests.jsonl](results/main/eval-requests.jsonl) | 独立集逐请求时延、输出及缓存计数 |
| [hardware.json](results/main/hardware.json) · [原始 trace](results/main/) | 72 条 GPU 消融计时、设备状态检查和三组压缩 CUDA trace |
| [summary.json](results/main/summary.json) | 按场景汇总的时延、质量和资源结果 |

开发和评估各使用 16 页 44 道真实问题，两个集合的源文档和像素身份互斥。每个集合的交错与连续场景包含同样的问题。软件 TTFT 从 GPU 输入就绪计到首 token，生成时间继续计到文本输出。硬件计时比较热前缀后的 8 个固定解码步，profiler 单独采集同一热请求的 2 次图重放。
