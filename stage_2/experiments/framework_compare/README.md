# 推理框架独立评估入口

复用`infer-lab`实验5.3中已有的vLLM Docker镜像，不向宿主推理环境安装vLLM，不重新下载模型。运行状态以`results/development/quality-gate.json`、`protocol.lock.json`与`results/formal/complete.json`为准。

## 对照与数据

七种主配置：HF原生H0、HF视觉缓存H1、本项目完整组合P，以及vLLM的APC/图开关V00、V10、V01、V11。额外的V11E只扩大encoder缓存容量，作为辅助容量对照。三个独立进程种子，每次执行热多问、含首次miss的会话和每页一次三个场景，共21次主启动、3次辅助启动、72个场景区组。

新评估集从已有固定DocVQA分片导出64页269问，源文档和RGB均与旧开发集/评估集互斥；所选页面的全部真实问题保留。开发质量门槛只读原282问，之后固定新评估协议再生成评估答案。首/末token计时均从驻留CPU图片与提示就绪开始，不与旧GPU输入就绪计时直接混算。

## 固定镜像和资源

已有镜像：`vllm/vllm-openai:v0.32.0`，实际按不可变image ID运行：

```text
sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34
```

容器Torch 2.13.0+cu130、Transformers 5.19.0；宿主沿用项目环境Torch 2.14.0+cu130、Transformers 5.17.0。模型为固定revision的Qwen3-VL-2B-Instruct，全GPU BF16，不启用量化或推测解码。B=1、上下文1024、新token上限64、真实EOS，原始图片、聊天模板和视觉预算一致。

容器禁用网络，以只读方式挂载项目及Hugging Face缓存，只为本实验目录提供写入。CPU数据预处理缓存0.125GiB，KV池1648MiB；本实验`max_num_batched_tokens=1024`时，派生的encoder缓存容量为1024 token，V11E设为16384 token。后者通过API配置对象设置容量，不改框架源码，encoder单步计算预算仍1024。1024是本实验设置下的容量，不代表所有vLLM默认配置。主测量使用公开的in-process引擎选项，避免引入HTTP层；不修改`infer-lab`既有服务容器。

## 运行

前提：Docker GPU访问正常，已有固定镜像、本地固定权重、随包提供的冻结页面／JSONL及`stage_2/.venv`可用。保留`data/manifest.json`时会复用所选数据，无需六份完整Parquet；仅重新导出选页时需要这些分片。从仓库根目录执行：

```bash
./stage_2/experiments/framework_compare/run.sh all
```

程序复用通过完整性检查的已有完成单元；遇到非空但未完成的结果目录会停止，保留失败证据。不要覆盖旧目录或将不同源码的结果混入冻结协议。各单元有独立JSON参数、源码快照、日志和GPU每2秒遥测。

逐步执行可使用`prepare`、`development`、`experiments`、`report`。需要真正重新执行全部推理时，在仓库根目录创建工作副本，将副本中的本实验结果和锁文件归档后运行。以下要求`stage_2_framework_replay`尚不存在，原目录保持不动，模型软链接和既有环境继续复用：

```bash
cp -a stage_2 stage_2_framework_replay
mkdir -p stage_2_framework_replay/experiments/framework_compare/archive-original
mv stage_2_framework_replay/experiments/framework_compare/results \
   stage_2_framework_replay/experiments/framework_compare/protocol.lock.json \
   stage_2_framework_replay/experiments/framework_compare/options.lock.json \
   stage_2_framework_replay/experiments/framework_compare/archive-original/
./stage_2_framework_replay/experiments/framework_compare/run.sh all
```

保留副本的根`configs/protocol.lock.json`、`cross_layer_v2`代码/开发证据和本实验冻结图片/问题，它们是测量依赖。该命令重新执行开发验收、诊断和24次独立启动；编译缓存可继续复用，初始化时间采用缓存产物存在的口径。单次接口诊断使用`launch.py --variant ... --phase preflight --output 新目录`，不会覆盖现有记录。

## 异机环境准备与路径

宿主Python的安装入口见根`REPRODUCE.md`：可复用已有基础环境，或用`requirements.lock`创建独立Python 3.12环境。`environment-host-effective.txt`还记录了共享base中未调用的工具，并非要求安装其定制FlashAttention；宿主实际使用SDPA。冻结页面及问题已在包内，模型权重用固定资产锁准备：

```bash
./stage_2/run.sh prepare-vlm --http --model-only
```

Docker镜像二进制不随源码包分发；若目标机没有原镜像，可按已记录digest获取，然后核对image ID：

```bash
docker pull vllm/vllm-openai@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34
docker image inspect sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34
```

这里提供异机命令，本轮自查未重新pull，也未执行全新机器安装。离线转移可在原机`docker save`固定image ID，目标机`docker load`后再核对。运行镜像还需要目标Docker支持`--gpus all`，相关驱动／Docker GPU集成属于前置条件。

冻结的`launch.py`使用`/home/lxy18/.cache/huggingface`作为只读挂载源和HF_HOME。在其他用户名的机器上，先将固定模型快照放入该位置，或在有目录权限的前提下把该路径链接到目标机实际HF缓存；模型链接须可在容器内相同路径解析。以相同用户名和目录布局恢复最直接。若选择改`launch.py`中的路径，应在新的工作副本先修改、再重新执行开发验收和协议冻结；旧测量的源码哈希须保持原值。本仓库工作目录可不同，模型缓存约束不可忽略。

`results/profile`包含首个CPU可见token之后固定八个输出间隔的CUDA profiler。框架调度可能提前执行下一步，该窗口不保证恰好覆盖八次GPU模型调用。固定输出数只用于机制诊断；主质量与性能数据全部按真实EOS停止。Python图探针不能保证捕获所有预绑定函数，也可能计入提前执行的步骤，CUDA trace重新计数是窗口内发射调用的依据。

## 文件

- `prepare_data.py`：仅按源文档/页ID哈希划分新的评估集，不依据质量或性能筛选。
- `common.py`、`adapters.py`、`run_engine.py`：统一逻辑输入、相同CPU计时边界和真实推理。
- `launch.py`、`execute.py`：串行GPU编排、开发门槛、协议冻结、源码快照与进度记录。
- `data/`：原始页面、真实问题和冻结划分清单。
- `results/`：开发/正式逐请求记录、质量门槛、源码、trace及中间统计。
- `cache/`：临时编译产物，复现结果与交付包不依赖将其打包。

旧的`cross_layer_v2`实现和测量不修改；宿主基线在本实验适配器中使用已解码图片，并通过观测原实现内部时钟取得首token绝对时间。vLLM采用命名worker extension读取执行计数，记录内容只用于诊断，不替代模型输出。
