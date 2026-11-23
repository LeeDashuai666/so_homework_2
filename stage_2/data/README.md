# DocVQA 数据

仓库收录实验使用的 DocVQA 页面原始图片、问题、参考答案及哈希清单。图片源自 UCSF Industry Documents Library，问题和参考答案来自 DocVQA 的人工标注。完整数据源为 [DocVQA 项目](https://www.docvqa.org/datasets) 的单文档视觉问答任务。本实验使用 [lmms-lab-encoder/DocVQA](https://huggingface.co/datasets/lmms-lab-encoder/DocVQA) 固定版本 `539088ef8a8ada01ac8e2e6d4e372586748a265e` 的 validation 分片。镜像数据卡标注 Apache-2.0，随附[许可文本](Apache-2.0.txt)。

- [原开发／评估划分](processed/docvqa/manifest.json)包含 64 页 282 问及 128 页 558 问，二者源文档互斥。图片是对应分片的原始字节，路径在 `development.jsonl` 和 `evaluation.jsonl` 中。
- [独立推理框架评估划分](../experiments/framework_compare/data/manifest.json)包含 64 页 269 问，与原划分源文档和 RGB 哈希均无交集。
- [第一轮到达序列](traces/manifest.json)记录程序生成的负载及其哈希，用于复算到达率与 SLO。
- [数据源锁](../configs/experiment-assets.lock.json)给出分片路径、提交版本、长度和 SHA256。需要重新导出划分时，请按[复现说明](../REPRODUCE.md)准备完整分片。

原论文为 Minesh Mathew、Dimosthenis Karatzas、C. V. Jawahar，*DocVQA: A Dataset for VQA on Document Images*，WACV 2021。
