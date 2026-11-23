# 项目环境：多模态文档视觉问答

当前环境继续使用项目 `.venv` 和原有 `~/venvs/infer`，复用 PyTorch/CUDA、Transformers、torchvision、Pillow、Triton 和性能工具。VLM 通过 Transformers 原生 processor 处理图像，不需要新增 qwen-vl-utils 或视频解码依赖。

- [多模态环境说明、配置与运行方式](VLM环境说明.md)
- [当前模型与数据锁](../configs/vlm-assets.lock.json)
- [图像、生成和显存检查配置](../configs/vlm-runtime.json)
- [全部依赖版本](../requirements.lock)
- [旧重排环境说明](README-rerank.md)，旧资产和旧验收保留

从仓库根目录使用：

```bash
source stage_2/env.sh
./stage_2/run.sh prepare-vlm --offline
./stage_2/run.sh smoke-vlm
./stage_2/run.sh ask-vlm --sample 0
```

首次准备可用 `./stage_2/run.sh prepare-vlm --http`，已有缓存会被复用；模型校验使用固定文件大小与 SHA-256。`prepare/smoke` 是当前 VLM 入口的别名，旧课题请使用 `prepare-rerank/smoke-rerank`。

`setup --profile vlm` 使用当前最小增量依赖锁；`setup --profile rerank` 可恢复旧增量依赖。重建前检查 base 的固定版本，不修改 base；现有旧包不卸载。两类原始资产清单分别保存在 `vlm-assets-manifest.json`、`assets-manifest.json`，不会互相覆盖。

当前 DocVQA 资产是 3 页 8 问的真实环境验收小样本，不是正式评测集。VLM 验收结果以 `vlm-smoke-latest.json` 为准，旧 `smoke-latest.json` 仍是 BGE 主题的历史记录。推理环境验收不等于完整数据集准确率或系统优化性能结果。
