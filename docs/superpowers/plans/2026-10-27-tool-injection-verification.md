# 混合检索工具接入验收（2026-10-27）

本次完成步骤 6：注册可选 `hybrid_search_documents`。步骤 7 的全链路质量对照尚未完成，默认开关保持关闭。

## 接入行为

- s2/s3/s4 在运行时有混合检索能力时注册新工具；旧工具的参数、说明、默认数量与行为保持。s1 文件未修改。
- 新工具参数 query 为非空白、最多 1000 字符；limit 默认 6，严格整数 1–10，拒绝额外字段。
- query 仅做空白归一化；协调器仅为向量一路添加 Qwen 检索指令，BM25 和重排使用原始查询。
- 语义/BM25 各取 50，RRF 取 30，Cross-Encoder 打分，再沿用已验证的 0.75 重排权重策略。
- 返回 query/passages，可附空结果或限制的 notice。片段只包含 id/title/text/rerank_score/final_score；原始模型分数与实际最终排序分数分别保存，完整来源留在证据和日志。
- 新旧工具共享阶段预算，`hybrid:` 单独去重；同一查询可分别使用新旧工具一次。s4 跨轮保存去重记录。
- AgentTurn 按片段 ID 登记一次证据，工具结果保留本次独立排名；旧工具先命中同一片段不会覆盖新分数。原生 tool_call_id 和工具载荷进入检查点，重启可恢复。

## 装配与开关

`Settings.hybrid_reranking_config` 默认为 None。配置路径相对主配置文件解析，可由 `FLOWPAY_HYBRID_RERANKING_CONFIG` 覆盖。

启用时资源所有者读取独立重排配置，要求 `local_files_only=true`，同步准备模型和 BM25 索引。缺少依赖、本地权重、设备或集合会中止启动并关闭知识连接。每轮只创建追踪绑定客户端，共享模型、互斥锁、词法索引和只读连接；不共享 tracer。

本地配置已给出可取消注释的启用项。默认 Docker 镜像不包含可选重排依赖，也未配置 GPU；本次没有变更或启动整套部署服务。

## 自动验证

新增 17 个测试：工具投影与非法参数、共享模型/独立追踪、配置开关与路径、启动失败清理、s2/s3/s4 接入/去重/预算/证据/重启、同批调用共享上限。

在设置 `FLOWPAY_DATASET_ROOT=D:\Code\RAG\flowpay-dataset\flowpay-dataset` 后执行：

```powershell
backend/.venv/Scripts/python.exe -m pytest backend/tests/core backend/tests/workflows backend/tests/scripts/test_fusion_benchmark.py backend/tests/scripts/test_lexical_benchmark.py backend/tests/scripts/test_reranking_benchmark.py backend/tests/scripts/test_ranking_policy.py --ignore=backend/tests/core/test_config_api.py --ignore=backend/tests/core/test_session_delete.py --ignore=backend/tests/workflows/test_routing.py -k 'not test_manifest_file_and_path_changes_are_rejected' -q
```

结果：**280 passed，4 skipped，1 deselected**。4 项为默认未启用的 PostgreSQL 集成测试。3 个排除文件依赖 Windows 缺少的 fcntl；取消选择的用例需要 Windows 创建符号链接权限。没有把这些项目计入通过。

后端源码及新增测试的 Ruff 检查、改动文件格式检查均通过；对应 BasedPyright 为 **0 errors / 0 warnings**。全仓 BasedPyright 另有既存 scripts/start.py 和 tests/scripts/test_start.py 警告，本次未修改它们。

独立审查未发现 Critical/Important 缺陷。最终排序分数作为工具新增字段已同步到计划。

## 实际知识库与 GPU 验证

通过真实 SessionManager 和 s2 工具循环执行了一次 `hybrid_search_documents`：

- 知识集合：`flowpay_docs_v1_qwen3_06b`，真实 PostgreSQL，只读账号；启动准备真实 BM25 索引。
- 重排：本地 `BAAI/bge-reranker-v2-m3` 固定 revision，CUDA/float16，真实推理。
- 问题：另一个 Account 的 Payment 返回 FP-4041 时，能否断言 Payment ID 不存在。
- 结果：completed，预算计 1 次，返回 6 条完整来源证据；第一条为 `doc:KB-TS-012#c007`，直接说明 FP-4041 不能证明资源删除或其他账户没有同名 ID。
- 日志中各有一次 hybrid.response 和 rerank.response；结果与检查点已落入 SQLite。

原始产物：`var/benchmark/tool-injection/report.json`、`events.sqlite`。验证脚本：`var/implementation/verify_tool_injection.py`，均为本地忽略产物。

这次使用已记录的真实查询向量回放，工具选择由受控聊天模型指定，未重新调用嵌入服务或外部聊天模型。因此它证明实际数据库、GPU、工具注入、预算与引用链路可运行，不证明真实聊天模型的工具选择能力或端到端生成质量。测试后已停止临时启动的 PostgreSQL，恢复原状态。
