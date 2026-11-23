# 推理与测量模块说明

本目录实现实际调用 Qwen3-VL 的单 GPU 推理运行时。模型/输入公共辅助代码位于 `../../tools/vlm_runtime.py`，入口是 `../../scripts/experiment.py`。

| 文件 | 职责 |
| --- | --- |
| `engine.py` | 模型与 processor 加载，输入组批，完整视觉边界替换，首 token/EOS 计时，CUDA 阶段事件 |
| `cache.py` | 完整主 embedding/DeepStack payload，GPU/CPU 排他存储，LRU/value 策略，lease 和搬运记账 |
| `scheduler.py` | 已就绪队列中有限 lookahead、等待年龄保护、图像亲和与代价组批 |
| `runner.py` | 固定 trace、独立到达生产者、CPU 预处理线程、GPU worker、有限队列、drain、完整请求账本 |
| `metrics.py` | ANLS/EM、分位数、SLO goodput 和拒绝/错误分母 |

## 一条请求的路径

独立生产者按 trace 的计划时刻产生请求。只有接纳后的请求进入 CPU 图像处理与分词队列；过载时队列拒绝会立即记账，后续请求的计划到达时刻不因 GPU 变慢而后移。

CPU 线程打开原始页面，用冻结 processor 形成图像 patch、问题 token、grid 和 mask。调度器只看已处理好的前 16 条请求，选取最多两条送入 GPU worker。

`generate()` 的视觉边界按如下顺序执行：

1. 对本批页面身份去重，更新历史频率，获取已有 GPU/CPU 条目并加 lease。
2. 将当前批次缺失的唯一图像交给原生视觉编码器，保存主 embedding 和三层 DeepStack，按每页独立 clone。
3. 按原请求顺序装配视觉张量，继续正常的语言 prefill 和 greedy decode。
4. 首生成 token 到 CPU 时计 TTFT；EOS 解码完成时计完整答案。
5. 本批生成结束后释放所有 lease。计数与 payload 不能在仍被本批引用时提前释放。

B0 在视觉边界直接调用原生批量编码，不使用缓存或批内图像去重。所有变体执行相同 CPU 图像处理和输入组批，每个完成请求均运行正常语言生成。

## 两级缓存约束

每个 payload 包含四个 BF16 张量，字节数为张量实际 `numel × element_size` 之和。GPU 与 CPU 层按互斥归属管理，CPU 晋升时复制回 GPU；当前批次可以持有未准入 GPU 字典的临时 tensor 引用。缓存容量约束仅指持久 payload，不混同 GPU 参数、KV、输入张量与激活。

所有策略共用按批 KV/workspace 安全预算。O3 的收益来自价值/字节准入淘汰策略，分数使用过去频率和实测重编码成本，不知道未来 trace。锁定模型 revision、图像处理和数据后，相同页面可跨问题复用视觉结果；问题相关语言 KV 不跨请求复用。

## 计时和质量口径

CPU 的计划到达、接收、预处理、就绪、调度、首 token 和完成时刻均使用同一单调时钟。视觉与首个语言 prefill 使用 CUDA Events。完成质量评分在测量结束后执行，避免 Levenshtein 计算占据被测 GPU worker。

普通请求内 KV cache 开启；跨请求的语言 KV、最终答案缓存和降分辨率策略均未实现。BF16 的原生视觉输出可能随视觉批次形状产生舍入差异，因此实际张量搬运一致性和全量答案质量分别验证，不能把完整 payload 的精确保存等同于任意批次之间 token 逐位相同。

开放环 goodput 同时约束 TTFT 与完整回答延迟；分母为固定到达窗口。离线模式用于质量和容量测量，其同时可用请求的排队分位数不用于在线延迟结论。
