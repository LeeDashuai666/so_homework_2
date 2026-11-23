# 项目环境

> 2026-11-16 已转向[多模态文档视觉问答](../plan/多模态推理项目计划.md)。本文件描述已搭建的公共基础与旧 BGE/SciFact 资产，已有 smoke 不等于 VLM 验收；新模型资产和视觉问答检查待补充。

本机采用**项目虚拟环境 + 现有推理环境引用**的方式。`stage_2/.venv` 通过一个 `.pth` 文件读取 `/home/lxy18/venvs/infer/lib/python3.12/site-packages`，只把缺少的依赖装到项目目录。PyTorch、CUDA 用户态库、Transformers、Triton、数据和绘图库直接复用，原推理环境不做安装、升级或卸载。

运行环境与该 base 路径有关，保留原环境才能继续使用。`setup_env.py` 会对照保存的清单检查 base 的版本，发现变化时停止并说明差异；不会自动升级 base。

## 日常使用

从仓库根目录执行：

```bash
source stage_2/env.sh
python -c 'import torch; print(torch.__version__, torch.cuda.get_device_name(0))'
./stage_2/run.sh smoke-rerank
```

`env.sh` 限制分词和数值库线程，使用 1 个 Inductor 编译线程，并把非交互绘图缓存放在项目内。这些是环境初始设置，后续正式实验需校准并在配置中记录。

当前 `run.sh` 提供环境操作；正式性能实验的 `full` / `reproduce-results` 尚未实现。

## 重建与资产准备

```bash
# 检查并复用 base，优先链接 uv 的本地缓存，安装缺少的固定版本包
./stage_2/run.sh setup --profile rerank

# 首次准备：固定模型 revision，下载后校验；也可直接复用已有缓存
./stage_2/run.sh prepare-rerank --http

# 已准备好的机器可以离线校验模型和数据
./stage_2/run.sh prepare-rerank --offline

# 基础 CUDA、编译、profiler、数据、标准 SDK 和实际模型图重放
./stage_2/run.sh smoke-rerank

# 重采机器环境
./stage_2/run.sh probe --output stage_2/env/machine-latest.json
```

模型仅下载 `model.safetensors`，没有再下载等价的 `pytorch_model.bin` 或 ONNX 权重。文件保存在现有 Hugging Face 共享缓存，`models/bge-reranker-base` 是固定 revision 的符号链接。HTTP 备用下载器按范围续传，最终必须通过发布文件的 SHA-256；默认 Hub 下载仍可使用 Xet。[Hub 缓存和下载环境配置](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables)。

SciFact 使用 BEIR 发布的 zip，检查上游 MD5，并保存完整 SHA-256 和解压文件清单。正式测量不需要联网。

如需重建新增依赖的离线 wheel 包，可在可联网时执行以下命令；当前共享 base 不包含在 wheelhouse 中：

```bash
source stage_2/env.sh
python -m pip download --no-deps --require-hashes --only-binary=:all: \
  -r stage_2/env/extra-requirements.lock -d stage_2/env/wheelhouse
./stage_2/run.sh setup --profile rerank --offline
```

这些命令会准备一份安装包供搬运，不是日常运行所需步骤；本次没有为了打包而再下载全部 wheel。

## 清单与验证记录

- `base-packages.json` / `base-constraints.txt`：构建前现有推理环境的版本清单。
- `extra-requirements.lock`：项目新增依赖，固定版本及发布 wheel SHA-256。
- `environment.json`：本机复用方式和路径。
- `effective-packages.json`：项目实际依赖及安装位置。
- `../requirements.lock`：项目依赖闭包的精确版本，不包括无关的 nano-vLLM 和本地定制 FlashAttention。
- `assets-manifest.json`：模型 revision、原始数据和每个文件的哈希。
- `smoke-latest.json`：完整环境验收结果；`smoke-core.json` 是下载期间进行的底层能力检查。
- `profiler-smoke.json`：实际 CPU/CUDA 时间线，可用兼容 Chrome trace 的查看器打开。
- `../logs/`：构建、下载和检查日志，本地保留，不纳入源码提交。

`../requirements.lock` 供另一台 Python 3.12 / Linux x86_64 机器准备独立环境，CUDA wheel 使用列出的 PyTorch 索引。该路线需要下载大型库，本次没有执行；当前通过验收的是复用本机 base 的路线。驱动和 GPU 硬件信息另外保存在机器快照中。

## CPU 计数器与开发工具

复用系统 GCC/G++、base 的 Ninja，以及第一个项目保存的 perf 二进制与其配套动态库。项目 `perf` 包装脚本只为该命令设置库搜索路径，未修改系统配置。

```bash
source stage_2/env.sh
perf stat -e cycles:u,instructions:u,branches:u,branch-misses:u \
  python -c 'print(sum(range(1000000)))'
```

CPU 计数器用于解释预处理开销；GPU 阶段使用 CUDA Events 与已验证的 PyTorch Profiler。当前没有另装完整 CUDA Toolkit、Nsight Systems、vLLM 或 TEI 大镜像。已有 vLLM 镜像可留给后续框架对照，当前主实现不依赖它。

环境检查中的 Z3 样例只证明求解器可调用，模型图样例只验证推理结果和重放能力；它们不是项目的正式形式化验证或性能结果。
