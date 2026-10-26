# 混合检索协调层步骤 4 验证记录

## 实现范围

新增 `HybridKnowledgeClient`：异步向量召回与线程内 BM25 并行，RRF 融合后截取候选，调用注入的 Reranker，校验映射并返回 Top-K。默认每路 50、RRF 常数 60、重排候选最多 30；工具侧未来返回数约束为 1..10，默认 5。查询非空且最多 1000 字符，只有向量分支应用既有 Qwen 查询指令。

新增 `RerankHit`、`HybridRanking`、`HybridHit`、`HybridSearchResult` 和相关能力协议。每个输出仍使用原 Passage，向量分数、BM25 分数、各路排名、RRF 分数、重排分数独立保存；不改变 Passage.score。

重排器必须为全部候选各返回一次合法下标和有限分数。遗漏、重复、越界及 bool 下标拒绝；同分按文档编号与块号稳定排序。双路为空时不调用重排。所有证据必须属于请求集合，即使两路都错误返回同一个其他集合，也会被拒绝。

## 生命周期与错误处理

- 任一路失败立即取消另一分支，等已拥有的工作清理完再报告异常。不静默回退为 RRF 或空结果。
- 同步 BM25 线程不能强行中断，使用 shield 并等待线程退出；重复父任务取消不跳过清理。
- 阶段任务和等待分离，只向阶段操作传播一次取消，后续取消等待清理完成，覆盖向量及重排异步 finally。
- 对内置 KnowledgeClient 使用混合链路专用 `_JoinedKnowledgeClient` 适配；其 get_collection 和 search 同步读库在取消后等待线程退出。KnowledgeClient 新增私有 `_call_store` 接缝，默认仍执行原来的 asyncio.to_thread，旧工具的接口和默认取消行为不变。
- 自定义 DenseRetriever 必须遵守“取消返回前清理自己拥有的工作”契约；协调层不能回收供应商自行隐藏且不登记的后台任务。
- 默认全流程超时 30 秒。超时后清理同步线程可能超出预算，不能把它当作硬杀线程时限。
- 记录 hybrid.request、hybrid.stage、hybrid.response/error/canceled。阶段日志包含两路结果、融合全候选与截断数量、重排结果、状态和耗时，使用同一个 hybrid call_id 关联。

## 验证证据

本步新增 29 项测试，使用线程/异步屏障与真实 EventStore；证明流程和协议正确，不代表实际重排质量。

独立审查发现两个重要取消缺陷，均补充复现并修复：

1. 父任务重复取消打断 dense 的异步 finally：新增测试先失败，改为 shield 后通过；重排的相同边界也已覆盖。
2. 内置 KnowledgeClient 的同步 get_collection/search 在线程未完成时就报告取消：两个边界在禁用适配器时均失败，恢复专用回收适配器后通过。

其他验证覆盖双路并行、查询格式隔离、30 候选与 Top-K、空结果、不合法重排响应、集合绑定、同分排序、依赖失败、超时、同步线程回收及错误终态。

组合本步、BM25、RRF、Embedding 及两项检索诊断测试：**91 passed**。代码、格式和类型检查在提交前运行；最终结果见交付消息及本地实施日志。

扩展 core/workflows 与检索诊断回归：**221 passed, 4 skipped, 1 deselected**。4 项跳过是未启用真实 PostgreSQL 的测试；排除的 1 项为既有 Windows 符号链接权限测试。另在收集时排除已知依赖 fcntl 的 test_config_api.py、test_session_delete.py、test_routing.py，未宣称整套后端通过。

## 复现

```text
<PY> -m pytest backend/tests/core/test_hybrid_knowledge.py -q
<PY> -m pytest backend/tests/core/test_hybrid_knowledge.py backend/tests/core/test_rrf.py backend/tests/core/test_lexical.py backend/tests/core/test_embeddings.py backend/tests/scripts/test_fusion_benchmark.py backend/tests/scripts/test_lexical_benchmark.py -q
```

`<PY>` 使用项目虚拟环境 Python；测试无需启动模型服务或数据库。

本轮未加载真实 Cross-Encoder，没有新模型依赖，没有注册 hybrid_search_documents，没有修改 s1～s4 或原工具定义。下一步是为 Reranker 接口提供真实模型适配并验证质量、资源与延迟。
