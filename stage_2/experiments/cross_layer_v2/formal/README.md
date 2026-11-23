# 公共前缀KV与CUDA Graph的形式化验证

[结果报告](results/形式化验证报告.md) · [求解结果、前提与反例](results/summary.json) · [验证源码](verify.py) · [求解器环境](results/environment.json)

当前原型的9项安全查询均为UNSAT，对应9项故障负控制均为SAT。安全查询问“在给定前提下，是否存在违反性质的状态”；UNSAT表示模型内不存在这样的状态。每个安全／故障模型的前提还分别求解为SAT，共18项，防止矛盾前提造成空证明。所有查询保存为36份可独立重放的SMT2。

## 证明内容

| 方向 | 性质 | 源码对应 |
| --- | --- | --- |
| 公共前缀KV | P1：结构化缓存身份一致时，模型、图像、grid、前缀token及位置所决定的状态一致 | `Runtime.__init__/prepare`，实际源码检查和冻结SHA256 |
| 公共前缀KV | P2：若一层的前缀不依赖后缀，因果注意力和逐token运算保持此性质 | 任意前缀长度的归纳步，逐层应用于28层；因果网络语义作为前提 |
| 公共前缀KV | P3：后续工作区清空、恢复及任意符号数据覆盖，不改写已保存的前缀 | 实际`_save/_restore`与Transformers的`reset`符号执行；未来K/V为任意符号数组 |
| 公共前缀KV | P4：28层K/V逐元素恢复正确，所有游标回到前缀边界 | 实际`_restore`符号执行 |
| 静态KV／图执行 | G1：准入时预留生成预算后，每步写入在容量内 | 从`prepare`的真实assert抽取谓词；`generate`最多63步decode |
| 静态KV／图执行 | G2：写入当前token后，mask恰好开放有效KV | 实际`_decode`和`StaticLayer.update`符号执行 |
| 静态KV／图执行 | G3：捕获的数据流读取当前缓冲内容时，与eager的单步抽象迁移一致 | `_decode`生成符号数据流，捕获后用当前状态代入，再与重新执行结果比较 |
| 静态KV／图执行 | G4：恢复与解码不改变图绑定的KV、游标、token、delta和columns地址 | 实际`copy_`、`add_`、`index_copy_`及存储标识检查 |
| 静态KV／图执行 | G5：三步预热和图捕获之后，恢复有效KV、游标及首token | 真实`build_graph`控制流程，成功prefill与流执行顺序作为前提 |

张量存储用符号数组建模，地址用不相交的抽象标识建模。全部28层均展开，K/V数组中的单个标量代表任意头和通道的对应元素。证明适用于本实验B=1、1024-token容量和最多64输出token；诊断接口`fixed_steps`也须不超过64。它不把诊断接口的任意参数调用都称为安全。

## 与真实代码及硬件验证的关系

验证器解析冻结的`runtime.py`，直接执行其中的`_save`、`_restore`、`_decode`、`build_graph`方法；同时解析已安装Transformers 5.17.0的`StaticLayer.update`和`CacheLayerMixin.reset`。使用Z3符号张量替换PyTorch张量，以原代码的切片、复制、索引写入、游标更新和mask运算构造查询。原型、验证器、模型配置、依赖源码的SHA256及快照均随结果保存；重放前重新核对。

P1将SHA256键抽象为无碰撞的结构化身份，并不证明密码学无碰撞。P2以因果注意力和局部运算为网络接口前提。G3验证捕获张量数据流在当前状态代入后的迁移；CUDA Graph按接口忠实重放、流按顺序执行和张量算子语义正确属于可信基础。神经网络数值计算用确定性未解释函数表示，不能由此推出BF16逐位等价或GPU驱动／kernel的正确性。

形式模型给出上述条件下的状态与流程保证，真实设备执行由原有`verify_runtime.py`、282问×六路径答案对照及框架独立质量门槛验证。两个证据对应不同问题，结果相互补充。

## 故障负控制

分别模拟缓存键漏掉位置、因果mask被去掉、clone变为别名、恢复游标偏移、准入不预留生成空间、mask多开放一格、图固定旧游标／delta、重新绑定token以及捕获后未恢复KV。每个故障模型均能生成反例；它们是对保护机制的检验，不是正常实验发生的错误。

## 一键复现

从仓库根目录执行，无需GPU、网络、模型权重或API key：

```bash
./stage_2/experiments/cross_layer_v2/run.sh formal
```

已有结果时核对源码及36份SMT2哈希，再独立加载SMT2求解。完全重新生成查询、反例与报告，使用新的目录：

```bash
./stage_2/experiments/cross_layer_v2/run.sh formal --output stage_2/replays/formal-new
```

前提是已按根`REPRODUCE.md`准备Python环境；本机复用`z3-solver==5.4.1.0`与Transformers 5.17.0，无需新增安装。开发消融的`run.sh all --output 新目录`现已在真实GPU测量之前自动生成一份形式化验证证据。根交付自查也自动重放当前证明。
