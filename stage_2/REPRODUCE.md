# 实验复现说明

所有命令从仓库根目录执行。仓库提供锁定的 DocVQA 页面、问题、逐请求输出、计时、GPU 采样和 CUDA trace。

## 1. 环境准备

使用 Linux x86_64 或 WSL2、Python 3.12。在新机器上创建项目虚拟环境并安装精确版本：

```bash
python3.12 -m venv stage_2/.venv
stage_2/.venv/bin/python -m pip install -r stage_2/requirements.lock
source stage_2/env.sh
```

本机宿主为 PyTorch 2.14.0+cu130、Transformers 5.17.0、Triton 3.8.0，完整软件见[依赖锁](requirements.lock)和[宿主实测环境](experiments/framework_compare/environment-host-effective.txt)。形式化验证使用 `z3-solver==5.4.1.0`。GPU 测量要求可用的 NVIDIA CUDA GPU、兼容驱动和足够显存，本实验为 WSL2、驱动 592.00 和 RTX 4060 Laptop 8 GB。vLLM 对照另需支持 `--gpus all` 的 Docker、固定镜像和容器内可访问的模型快照。

## 2. 从原始记录复算

准备 Python 环境后运行下面的检查。它核对 Word 和 Markdown 哈希，并复算本轮缓存选择、AI 决策与 GPU trace，以及已保存的框架对照和 Z3 证据。输出写入 `stage_2/results/submission-audit/`。CPU 复算使用上述 Python 环境。

```bash
python stage_2/tools/check_submission.py
```

需要单项重算时，从以下命令选择：

```bash
./stage_2/experiments/cross_layer_v2/run.sh report
./stage_2/experiments/framework_compare/run.sh report
./stage_2/experiments/cross_layer_v2/run.sh formal
./stage_2/experiments/adaptive_kv/run.sh report
./stage_2/run.sh verify-results
./stage_2/run.sh report
```

本轮保存 1144 条开发和独立评估逐请求记录、72 条 GPU 消融计时和三份压缩 trace。早一轮 vLLM 对照保存 9720 条计时记录。本轮主实验从 GPU 输入就绪开始计时，vLLM 对照从 CPU 输入就绪开始。

## 3. GPU 重新测量

先把锁定的 Qwen3-VL-2B-Instruct 权重放入共享缓存并核验。完成后项目内的模型路径为 `stage_2/models/qwen3-vl-2b-instruct`。已缓存权重用 `--offline --model-only`。

```bash
./stage_2/run.sh prepare-vlm --http --model-only
```

[模型及数据锁](configs/vlm-assets.lock.json)给出模型版本、路径和哈希。仓库中的冻结图片和 JSONL 已足够执行正式测量。只有重新导出 DocVQA 划分时才需下载完整 validation Parquet，步骤见[数据来源](data/README.md)及下文。

### 本轮缓存优化、离线选参与 GPU 写回

准备 [Qwen3-0.6B 权重锁](experiments/ai_cache_tuning/policy-model.lock.json)指定的本地快照。已有权重时将 `STAGE2_POLICY_MODEL` 指向本地目录。也可按锁下载：

```bash
python - <<'PYTHON'
import json
from pathlib import Path
from huggingface_hub import snapshot_download

lock = json.loads(Path("stage_2/experiments/ai_cache_tuning/policy-model.lock.json").read_text())
snapshot_download(repo_id=lock["name"], revision=lock["revision"],
                  allow_patterns=list(lock["sha256"]),
                  local_dir="stage_2/models/Qwen3-0.6B")
PYTHON
```

GPU 侧使用同一 CUDA 设备运行开发和独立评估。Triton 首次编译需要系统 C 编译器。离线选参在 CPU 上使用 BF16、2 个线程及单 token 候选。

```bash
export STAGE2_POLICY_MODEL="$PWD/stage_2/models/Qwen3-0.6B"
./stage_2/experiments/adaptive_kv/run.sh all stage_2/replays/adaptive-kv-new
```

入口依次执行开发 GPU 测量、CPU 选参、独立 GPU 评估与复算。分步执行时，对同一个新目录依次传入 `dev`、`ai`、`eval` 和 `report`。参数目录和实测环境见 [实验说明](experiments/adaptive_kv/README.md)，提示词和模型输出见 [决策记录](experiments/adaptive_kv/results/main/decision.json)。

### 早一轮对照

- 开发消融。目标目录需要是新目录。命令顺序执行状态检查、GPU 生成及计时，然后经审计生成报告。

  ```bash
  ./stage_2/experiments/cross_layer_v2/run.sh all \
    --output stage_2/replays/cross-layer-new
  ```

- vLLM 独立评估。先按[框架对照复现说明](experiments/framework_compare/README.md#异机环境准备与路径)准备固定镜像和 Hugging Face 缓存路径，再按[工作副本步骤](experiments/framework_compare/README.md#运行)归档副本中的旧结果和冻结记录。随后运行副本中的 `./stage_2_framework_replay/experiments/framework_compare/run.sh all`，自动执行开发质量门槛、协议冻结、串行 GPU 测量和统计。直接运行现有仓库的 `./stage_2/experiments/framework_compare/run.sh all` 会复用该仓库已记录的完成单元。
- 第一轮。用 `./stage_2/run.sh rerun --output stage_2/replays/first-round-new` 创建新结果树，依次运行质量、45 轮主矩阵、18 个辅助实验、核验与报告。约 45×60 秒到达窗口，另外需要 GPU 初始化、排空和质量评估时间。

若要重新下载全量分片并导出第一轮数据，在单独的工作副本里备份原有 `data/processed/docvqa`、`data/traces`、`configs/protocol.lock.json` 与 `results` 后，再执行 `./stage_2/run.sh prepare-data`、`calibrate`、`experiments` 和 `report`。新划分清单和新协议需对应同一测量，原结果依赖仓库里的原始哈希。

## 4. 形式化验证

`./stage_2/experiments/cross_layer_v2/run.sh formal` 使用现有 Python 环境重放 36 份 SMT2。要重新生成符号模型和反例，使用同一入口的 `--output 新目录`。版本、源码快照、SMT2、负控制反例和证明适用条件见[验证说明](experiments/cross_layer_v2/formal/README.md)。第一轮 lease 模型入口是 `./stage_2/run.sh test`。
