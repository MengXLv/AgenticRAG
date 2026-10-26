# BM25 步骤 2 实施与验证记录

执行范围：按用户要求跳过步骤 1，仅实现步骤 2。新增 BM25 能力尚未注册为模型工具；RRF 与 Cross-Encoder 尚未接入。

## 实现

- `BM25Retriever` 按 collection_id 懒加载只读片段并缓存，提供 prepare 预热；并发首次访问只构建一次。
- PostgreSQL 新增 list_passages：仅允许 ready 集合，按 doc_id/chunk_no 返回完整来源，不加载向量。
- 文档和查询共用分词，中文使用 jieba，保留英文标识、错误码、下划线字段和版本号，英文统一小写。
- BM25 分数存放在 LexicalHit，不覆盖 Passage.score；只返回有共享 token 的候选，负/零分匹配仍保留。
- 集合内容错误、重复片段、数据库异常不伪装为零命中，初始化失败不缓存。
- 原 search_documents/read_document、Runtime 装配和 s1～s4 工作流未更改。

## 验证方法

确定性验证：中文与技术词目标进入 Top-3，无匹配为空，负分匹配保留，集合隔离，building 拒绝，索引复用、并发及稳定排序，输入证据不变，源异常后可重试。

数据库验证：独立临时 pgvector 0.8.2 / PostgreSQL 17 容器，使用现有迁移创建 reader/builder 角色；验证按集合读取、顺序、元数据/定位符保留及 BM25 真实查询。只读连接不增加新权限。

性能夹具：数据包原有 `data/index/knowledge_chunks.jsonl`，100 篇文档、809 个现成片段，没有重新切割。该索引与正式导入分块可能不同，因此报告不代表正式在线集合。输入文件 SHA256：

`0f421439595277b84c647af8298109eeb0f7bebd2483a06e54077ff7558a0557`

夹具写入临时测试集合 `bm25_dataset_prebuilt_fixture`，三维测试向量仅用于满足 ready 约束，没有调用模型、没有修改正式集合。初始化性能包含只读查询、分词词典加载、BM25 构建；不包含建立数据库连接池之前的耗时。

## 性能实测

运行环境：Windows 11、Python 3.12.2。查询为 HTTP 410、FP-4021、account_id、幂等键冲突，各预热一次，再各测 30 次，总计 120 次。

| 项目 | 结果 |
| --- | --- |
| 全量片段读取次数 | 1 |
| 独立进程未做内存插桩的初始化 | 503.31 ms |
| 热查询 p50 / p95 | 0.156 / 0.298 ms |
| 独立内存测量进程的 tracemalloc 峰值 | 55,997,934 bytes，约 53.4 MiB |
| 开启 tracemalloc 的初始化 | 21.43 s，存在显著插桩开销，不能当正常启动延迟 |

内存数字只包含 prepare 期间可由 tracemalloc 跟踪的分配，不是完整进程 RSS。查询延迟是本次小规模夹具结果，包含 25% 零命中查询；不外推到大语料或高并发。account_id 在该夹具没有命中，未用其他片段填充；其他三个查询均有候选。字段名匹配能力由独立构造语料测试验证。

由于步骤 1 已跳过，本次没有冻结 SLA，也不声明相对纯向量检索有质量收益。正式集合容量、延迟目标以及召回收益另行验收。

原始报告：`var/benchmark/bm25-step2.json`（带内存插桩）、`var/benchmark/bm25-step2-timing.json`（未插桩）。这些本地产物不提交 Git。

## 复现入口

在仓库根目录，使用已安装依赖的 Python；Windows 为 `backend/.venv/Scripts/python.exe`，Linux 为 `backend/.venv/bin/python`。

```text
<PY> -m pytest backend/tests/core/test_lexical.py backend/tests/scripts/test_lexical_benchmark.py -q
<PY> scripts/benchmark/lexical.py --collection <ready集合ID> --query "HTTP 410" --query "FP-4021" --repeats 30 --no-memory --output var/benchmark/bm25-timing.json
```

性能脚本默认从 `FLOWPAY_KNOWLEDGE_DSN` 读取只读连接串，可用 --dsn-env 指定其他变量；省略 --no-memory 测内存，需要独立进程运行，不将连接串写入报告。PostgreSQL 测试还需显式配置 FLOWPAY_TEST_POSTGRES 及独立测试库的 reader/builder/admin 环境变量。

## 验证限制

- Windows 原有 fcntl 导入限制阻断 `test_config_api.py`、`test_session_delete.py`、`test_routing.py` 收集。
- 扩展回归首次运行 169 passed / 1 failed，失败是既有 `test_manifest_file_and_path_changes_are_rejected` 创建符号链接缺少 Windows 权限。未修改该测试或运行时来掩盖结果。
- 原有核心/工作流与新增代码的最终针对性结果，在最终交付消息及实施记录中列出；不能把局部通过写为全套后端通过。
