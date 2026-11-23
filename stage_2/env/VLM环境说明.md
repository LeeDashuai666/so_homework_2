# 多模态文档视觉问答运行环境

本环境服务于[当前多模态课题](../plan/多模态推理项目计划.md)。继续复用原有 PyTorch/CUDA、Transformers、torchvision、Pillow 和分析工具；不更换驱动、不重复安装大型框架，也不需要为图像输入额外安装 qwen-vl-utils、视频解码器或完整 CUDA Toolkit。

2026-11-16 本机 **9/9 验收通过**，包含 BF16/SDPA 下 batch=1/2、完整 DeepStack 特征往返、同页两问的生成 token 对照与异页 miss。该轮 PyTorch 峰值 allocated 为 4.35 GiB、reserved 为 4.40 GiB，检查结束时设备可用约 2.50 GiB。原有 98 个 base 包未改变，本轮没有新增安装包。

证据：[验收报告](vlm-smoke-latest.json)、[配置快照](vlm-environment.json)、[模型与数据文件清单](vlm-assets-manifest.json)、[base 完整性](vlm-base-integrity.json)。

## 1. 固定配置

| 配置 | 当前设定与来源 |
| --- | --- |
| Python | 项目 `.venv`，引用原有 `~/venvs/infer`，详见 `environment.json` |
| 模型 | Qwen/Qwen3-VL-2B-Instruct |
| revision | `89644892e4d85e24eaac8bacfd4f463576704203` |
| 权重与配套文件 | `../configs/vlm-assets.lock.json`，只取一份 safetensors 及 processor/tokenizer/chat template |
| 执行设备 | 单张 GPU，全部模型参数在 GPU，关闭自动 CPU/磁盘卸载路径 |
| 精度 / attention | BF16 / SDPA；与模型配置一致，原生支持由 smoke 检查 |
| 图像预算 | min 128、max 768 视觉 token；32 像素空间因子，对应像素面积 131072–786432 |
| 输入与输出 | 左侧 padding；输入 token 守卫 2048；greedy、正常 EOS、最多新生成 64 token |
| CPU 线程 | 起始 2 个；沿用 `env.sh` 的公共线程设置 |
| smoke 数据 | 3 张真实 DocVQA 页面、8 个不同问题，包含同页多问；只用于环境与接口验收 |

精确参数位于 [vlm-runtime.json](../configs/vlm-runtime.json)。这是已配置的环境起点，不表示在完整 DocVQA 上最优。正式实验需要另外固定 split、SLO、缓存和调度参数。

模型文件位于现有 Hugging Face 共享缓存，项目 `models/qwen3-vl-2b-instruct` 为固定 revision 的符号链接。原 BGE 权重、SciFact 数据和原验收记录保留，避免破坏旧项目。

## 2. 直接使用

从仓库根目录执行：

```bash
source stage_2/env.sh

# 多模态完整环境验收；默认 ./stage_2/run.sh 也执行此项
./stage_2/run.sh smoke-vlm

# 运行已准备的真实页面和第一个问题
./stage_2/run.sh ask-vlm --sample 0

# 指定自己的文档图片与问题，使用相同模型配置
./stage_2/run.sh ask-vlm --image /absolute/path/document.png --question 'What is the total amount?'
```

推理和 smoke 均只读本地模型文件，不需要联网。`ask-vlm` 每次启动独立进程加载一个模型；它是运行示例，不是已实现的多请求服务。

## 3. 环境与资产复现

```bash
# 复用 base，并校验当前 VLM 需要的固定依赖
./stage_2/run.sh setup --profile vlm

# 联网准备：按范围续传模型并验证 SHA-256，准备真实小样本
./stage_2/run.sh prepare-vlm --http

# 已准备好的机器只做离线校验，不再次下载
./stage_2/run.sh prepare-vlm --offline

# 保存新机器快照
./stage_2/run.sh probe --output stage_2/env/vlm-machine-latest.json
```

`prepare` 与 `smoke` 现在分别是 `prepare-vlm`、`smoke-vlm` 的别名。旧主题请显式使用 `prepare-rerank`、`smoke-rerank`；旧依赖重建可用 `setup --profile rerank`。

完整依赖版本在 [requirements.lock](../requirements.lock)，当前 VLM 的项目增量依赖在 `extra-requirements.lock`，旧增量锁在 `extra-requirements-rerank.lock`。已安装但新课题不需要的旧包保留，不执行卸载或升级。

## 4. DocVQA 小样本的来源与边界

样本取自 `lmms-lab-encoder/DocVQA` 的 `DocVQA/validation`，revision 为 `539088ef8a8ada01ac8e2e6d4e372586748a265e`，使用数据浏览服务的真实图片导出与人工问题。

准备脚本检查图片来源 URL 中的 revision，并保存本地文件 SHA-256、图片尺寸、页面/问题 ID、参考答案和源 row index。临时签名 URL 不作为离线依赖。重复问题引用同一份页面图片，不重复存储。

数据位于 `data/raw/docvqa-smoke/`；清单中的 `scope` 明确标记为环境小样本。正式评测应另建按文档划分的数据集，并固定原始 Parquet/图片资产，不能把这个小样本当作完整基准或正式质量成绩。

## 5. 多模态 smoke 验证内容

`tools/smoke_vlm.py` 依次验证：

1. 固定模型资产、数据哈希与依赖。
2. 原生 processor、视觉 token 预算、同页相同缓存身份与异页不同身份。
3. 全 GPU BF16 模型加载。
4. 真实页面和问题的正常生成。
5. 主视觉特征和三组 DeepStack 的 GPU→CPU→GPU 往返及 prefill logits 对照。
6. 同页不同问题使用临时特征接口替换后，生成 token 与正常路径一致，视觉编码器调用为零。
7. 不同页面必须 miss 并重新执行视觉编码，避免跨页面污染。
8. 两张不同页面的 batch=2 图文生成。
9. 该配置下的峰值显存与剩余显存余量。

临时视觉边界替换只存在于诊断上下文，退出即恢复。它检查 O1 所需接口，不是生产缓存、调度器或 O3 策略；不跨问题复用语言模型 KV。

验收结果保存为 `vlm-smoke-latest.json`，资产清单为 `vlm-assets-manifest.json`，机器快照为 `vlm-machine-latest.json`。输出答案可人工查看；smoke 不计算完整数据集 ANLS，不构成性能加速或质量达标的结论。

## 6. 调整参数

修改 `configs/vlm-runtime.json` 后，重跑 `smoke-vlm` 并保存新报告。`max_image_tokens` 与 `max_pixels` 应同时调整，满足 `max_pixels = max_image_tokens * 32 * 32`；processor 实际输出的 grid/token 数仍由脚本校验。

更改分辨率、精度、模型 revision 或视觉后端后，视觉缓存身份应变化。正式对照必须使用统一输入配置。显存记录只覆盖实际验收过的样本、batch 和输出上限，后续并发与新输入规模继续做准入和测量。
