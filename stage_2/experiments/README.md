# 实验

| 目录 | 验证内容 |
| --- | --- |
| [cross_layer_v2/](cross_layer_v2/) | 六路径开发消融、GPU 状态检查、CUDA trace 和 [Z3 形式化验证](cross_layer_v2/formal/) |
| [framework_compare/](framework_compare/) | 原型与 vLLM 的冻结对照、独立 DocVQA 输入、原始请求日志与统计结果 |
| [ai_cache_tuning/](ai_cache_tuning/) | Qwen3-0.6B 的离线选参、提示词、决策和原始记录回放 |

各目录的 README 给出运行入口和对应环境。统一步骤见[复现说明](../REPRODUCE.md)。
