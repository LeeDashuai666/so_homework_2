# 实验

| 目录 | 验证内容 |
| --- | --- |
| [adaptive_kv/](adaptive_kv/) | 成本缓存选择、AI 离线选参、Triton 写回消融、GPU 状态检查与实测 trace |
| [framework_compare/](framework_compare/) | 原型与 vLLM 的冻结对照、独立 DocVQA 输入、原始请求日志与统计结果 |
| [cross_layer_v2/](cross_layer_v2/) | 早一轮开发消融、GPU 状态检查及 [Z3 状态验证](cross_layer_v2/formal/) |

各目录的 README 提供运行入口和环境。统一步骤见[复现说明](../REPRODUCE.md)。
