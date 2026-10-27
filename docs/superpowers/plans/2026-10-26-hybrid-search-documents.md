# hybrid_search_documents Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> 本次仅交付计划，不启动实现或委派。后续执行时以当前环境实际提供的技能和用户选择为准。

**Goal:** 新增可选工具 `hybrid_search_documents`，完成向量与 BM25 召回、RRF 融合、Cross-Encoder 重排，保留原工具行为。

**Architecture:** 新增独立混合检索协调层、词法索引和重排接口，通过 Runtime 注入工作流。复用现有 PostgreSQL 片段与向量能力；排名信息与原文证据分离。先建立独立检索评测，再逐层接通，最后提供模型工具。

**Tech Stack:** Python 3.12、PostgreSQL/pgvector、现有 asyncio/httpx/pytest；第一版计划使用进程内 `rank_bm25` 与中文分词器。模型适配和依赖版本在真实重排验证步骤固定，不在本文虚构已验证配置。

**Spec:** 本文“范围与验收约定”记录当前会话需求和待验证工程假设；本文尚未执行，不代表指标已达到。

## Global Constraints

执行调整（2026-10-26）：用户要求跳过步骤 1，直接开始步骤 2。本轮仅实施 BM25；质量对照、冻结预算不作为本轮前置条件。步骤 2 的性能数值作为测量报告，不宣称通过尚未定义的 SLA。详见 `2026-10-26-bm25-verification.md`。

- 工具名固定为 `hybrid_search_documents`；`search_documents`、`read_document` 的参数、返回语义和执行路径保持不变。
- 不改变文档切割策略，不重新生成已有文档向量。允许读取现有片段建立词法索引。
- s1 保留现有纯向量基线；新工具仅在显式启用时加入 s2/s3/s4。
- workflows 只处理编排、提示词、预算和业务决策；SDK、SQL、索引、模型推理与追踪实现在 core。
- 新工具只接受 query 和 limit；limit 默认 5，严格整数 1..10；query 非空且最多 1000 字符。
- 初始实验参数：每路最多 50 条，RRF 常数 60，融合后最多 30 条，最终最多 limit 条；参数必须满足 rerank_limit >= 最大允许的 limit。
- 新工具收到原始查询。仅向量分支应用现有 Qwen 查询指令，BM25 和重排不使用该指令前缀。
- 两路必须使用同一 ready collection；内部身份键为 (collection_id, doc_id, chunk_no)。不跨集合混合。
- 不把相似度、BM25、RRF、重排分数解释为答案置信度；不修改旧 Passage.score 的含义。
- 新工具默认关闭；关闭时不加载 BM25 索引或重排模型，不增加旧链路的外部请求。
- 系统异常不得伪装为空结果。第一版任一路或重排失败即让新工具报告失败，不自动静默降级；用户取消继续向上传播。

## Review Focus

1. 中英混排及 FP-4021、account_id、版本号不能被分词破坏到无法精确匹配：步骤 2 固定测试。
2. BM25 无匹配及双路全空不能产生填充候选：步骤 2、4、5 固定测试。
3. 相同片段由旧工具先返回后，新排名不能被 PassageSet 首次登记值覆盖：步骤 4、6 固定测试。
4. 相同 doc_id 属于不同集合时不能串用索引和证据：步骤 2、3、4 固定测试。
5. 超时、取消和冷启动不能产生泄漏任务、反复加载模型或未配对工具消息：步骤 4、5、6 固定测试。

## 范围与验收约定

- 每步产物分为“确定性正确性证据”和“真实数据实验报告”。假模型证明协议与排序正确，不能证明真实模型有收益。
- 进程内 BM25 是待验证方案。步骤 2 必须报告实际片段数、初始化时间、峰值内存和单次查询 p50/p95；容量不符合运行环境时停在该步，调整词法后端，不以小样本测试宣称可扩展。
- 固定评测集合来自现有 benchmark 的文档问题，排除必须依赖业务记录才能回答的问题；另建最小术语/否定/多轮指代诊断集。开发集和验收集按场景分开，不能改同一题措辞后跨集合复用。
- 复用 scripts/benchmark/evidence.py 的小节对齐方式，按 distinct 文档小节计算证据 Recall@K；排名指标先把同一小节的多个片段折叠，避免重复块虚增收益。空相关标注样本不计入相关性均值，单独报告。
- 报告 Recall@50、融合候选 Recall@30、最终 Recall@5、MRR@5、nDCG@5、证据全覆盖率、候选数、各段 p50/p95、错误率及初始化内存。排名指标采用二元相关标签。
- 纯检索先固定同一 query 比较策略，避免 Agent 查询差异混入算法效果；多轮 query 构造另作端到端验证。
- 真实延迟验收线在步骤 1 按目标机器和业务预算写入实验配置，执行前冻结。本文不凭空指定毫秒值；未冻结预算不得宣布性能验收通过。
- 每一步先增加能检出该类故障的测试并确认失败，再实现、确认通过，独立审查和提交。纯报告步骤不要求制造无意义失败。

## 文件职责

新增文件：
- `core/knowledge/hybrid_contracts.py`：词法召回、候选排名、混合结果和重排接口。
- `core/knowledge/tokenization.py`：文档与查询共用的中文/技术词处理。
- `core/knowledge/lexical.py`：按集合构建及复用的进程内 BM25 索引。
- `core/knowledge/fusion.py`：无外部依赖的 RRF。
- `core/knowledge/hybrid.py`：两路调用、融合、重排、事件与错误生命周期。
- `core/knowledge/hybrid_tools.py`：新工具参数校验及模型可见结果投影。
- `core/llm/reranking/`：具体重排模型适配、设置和生命周期。
- `scripts/benchmark/retrieval.py`：直接检索评测，不调用生成模型。

以上 core 路径均相对于 `backend/src/flowpay/`。现有 `knowledge/store.py` 增加按集合读取索引所需片段的只读方法；原 search/read_document 不改。资源装配、Runtime、AgentTurn、配置和阶段文件只增加新能力入口。

## 验证命令约定

以下命令是实施后新增或使用的验收入口，不表示当前均已存在或执行通过。在仓库根目录执行；Windows 使用 `backend/.venv/Scripts/python.exe`，Linux 使用 `backend/.venv/bin/python`，下文简写为 `<PY>`。新增评测入口必须支持显式配置和输出目录，不自行打印密钥。

### 步骤 1：固定评测与旧链路基线

**Files:** 新增 `scripts/benchmark/retrieval.py`、`backend/tests/scripts/test_retrieval_benchmark.py`；复用 `scripts/benchmark/evidence.py`。

**Interfaces:** `evaluate_rankings(expected: set[str], ranked: list[str], cutoff: int) -> RetrievalMetrics`；CLI 支持 `--strategy dense|bm25|rrf|hybrid`、`--config`、`--cases`、`--output`，本步只实现 dense，其余明确报告尚不可用。

- [ ] 固定 cases.jsonl（question_id/query/relevant_sections/split/category），保存集合身份、模型版本、配置及题目文件摘要；记录机器、冷/热启动定义和延迟预算。
- [ ] 测试手算排名：相关集合 {a,b}、排名 [x,a,b]，Recall@2=0.5，MRR@2=0.5；重复小节只计算一次；零标注单独报告。
- [ ] 实现独立 dense 跑批，不触发聊天生成；输出逐题排名、聚合指标和耗时。
- [ ] 执行 `<PY> -m pytest backend/tests/scripts/test_retrieval_benchmark.py -q`，所有指标测试通过。
- [ ] 在同一固定题集运行 dense 两次，排名在相同环境下可复现；保存差异和已有失败，形成 baseline 报告。

**通过标准：** 题集与预算已冻结；指标可手算复核；有真实基线，现存失败未被隐去。**提交：** `test: add retrieval evaluation baseline`。

### 步骤 2：单独验证 BM25

**Files:** 新增 `hybrid_contracts.py`、`tokenization.py`、`lexical.py`；修改 `knowledge/store.py` 的只读片段枚举能力；测试 `backend/tests/core/test_lexical.py`。

**Interfaces:** `LexicalHit(passage: Passage, score: float)`；`LexicalRetriever.search(collection_id: str, query: str, limit: int) -> list[LexicalHit]`；`PostgresVectorStore.list_passages(collection_id: str) -> list[Passage]`。枚举得到的 Passage.score 为非检索占位值，不携带 BM25 分数。

- [ ] 增加中文、FP-4021、HTTP 410、account_id、v1.2.3 的可解释最小语料测试，断言目标片段进入 Top-3；原始技术 token 保留。
- [ ] 测试查询无共享 token 时结果为空；匹配判定不能直接等同于 BM25 分数大于零，避免 IDF 形式影响召回。
- [ ] 测试两个集合存在同名文档时隔离；未就绪集合拒绝初始化；查询期间不重新枚举全部片段或重建索引。
- [ ] 实现并执行 `<PY> -m pytest backend/tests/core/test_lexical.py -q`；真实运行 `--strategy bm25`，报告内存、初始化与热查询性能。

**通过标准：** 确定性用例全过，索引每集合每生命周期只建一次；实际资源满足冻结预算。真实术语切片结果逐题可查。**提交：** `feat: add collection-scoped BM25 retrieval`。

### 步骤 3：验证 RRF 融合

**Files:** 新增 `fusion.py`、`backend/tests/core/test_rrf.py`；补充候选结构与评测 rrf 策略。

**Interfaces:** `RankedCandidate` 包含 Passage、dense_score/dense_rank、bm25_score/bm25_rank、rrf_score；`fuse_rrf(dense: list[Passage], lexical: list[LexicalHit], rank_constant: int = 60) -> list[RankedCandidate]`。

- [x] 手算测试 dense=[A,B]、bm25=[B,C]：B=1/62+1/61，A=1/61，C=1/62，排序 B>A>C。
- [x] 测试一路为空、双路为空、同一路重复项只记一次、不同集合候选拒绝混合；同分按 doc_id/chunk_no 稳定排序。
- [x] 实现并执行 `<PY> -m pytest backend/tests/core/test_rrf.py -q`。
- [x] 真实运行 rrf：记录两路 Top-50、并集和融合 Top-30；已标注样例报告锚点的召回/截断损失，未标注样例保持未知，不宣称全量证据覆盖。

步骤 3 验证记录见 `2026-10-26-rrf-verification.md`。同路去重后重新编号，首次命中的原分数被保留；融合函数返回完整列表，由调用方截取 Top-N。实际验收使用 2 个历史真实查询向量和真实只读知识集合，没有生成新向量。

**通过标准：** 算法测试全过；报告可定位每个证据在哪一路命中、是否被融合截断。此时不要求虚构 RRF 必然优于 dense。**提交：** `feat: add traceable RRF fusion`。

### 步骤 4：验证混合协调层与结果身份

**Files:** 新增 `hybrid.py`、`backend/tests/core/test_hybrid_knowledge.py`；扩展 hybrid_contracts.py，不修改旧 DocumentRetriever 协议。

**Interfaces:** `RerankHit(candidate_index: int, score: float)`；`Reranker.rank(query: str, passages: list[Passage]) -> Awaitable[list[RerankHit]]`；`HybridHit(passage: Passage, ranking: HybridRanking)`；`HybridSearchResult(hits: list[HybridHit])`；`HybridKnowledgeClient.search(collection_id: str, query: str, limit: int = 5) -> Awaitable[HybridSearchResult]`。

- [x] 使用屏障式假检索器验证两路确实重叠执行，不用脆弱的固定 sleep 时间判断；仅向量侧收到 Qwen 指令前缀。
- [x] 使用可控假重排器验证 RRF 30 条上限、按重排分数返回最终 limit 条；零候选时不调用重排；同分稳定排序。
- [x] 测试重排候选索引越界、重复、遗漏及非有限分数必须报错；不能把外部响应错配到原文。
- [x] 验证失败或取消时另一分支被取消并等待回收；记录 hybrid.request/response/error/canceled，并分开记录每路、融合、重排的排名与耗时。
- [x] 断言输入 Passage 和旧工具返回分数未被修改，HybridRanking 独立保存；执行 `<PY> -m pytest backend/tests/core/test_hybrid_knowledge.py -q`。

步骤 4 验证记录见 `2026-10-26-hybrid-coordination-verification.md`。默认全流程超时 30 秒；取消/超时后等待同步读库结束可能超过此值，它不是强制杀线程的硬截止时间。真实 Cross-Encoder 尚未接入。

**通过标准：** 假依赖能证明完整链路、身份映射和异常生命周期正确；明确尚未证明真实重排质量。**提交：** `feat: orchestrate hybrid retrieval with ranking provenance`。

### 步骤 5：接入并验证真实 Cross-Encoder

**Files:** 新增 `core/llm/reranking/` 设置及适配、`backend/tests/core/test_reranking.py`；按需修改 pyproject.toml 和配置解析；模型实验结果写入 var/benchmark 的运行目录。

**Interfaces:** 实现步骤 4 的 Reranker。适配器返回输入候选的索引和分数，模型只加载一次或复用服务连接；同步推理不得阻塞事件循环。

- [x] 固定模型 ID/revision、库版本、设备、max_length、batch_size、截断和超时，再运行按场景族隔离的开发诊断与留出诊断。
- [x] 用假适配响应验证顺序映射、批处理、超时、取消与错误传播；真实查询验证错误码和条件规则，人工复核覆盖退化样例。
- [x] 执行 `<PY> -m pytest backend/tests/core/test_reranking.py -q`；对固定真实 RRF 候选比较已标注正例召回和证据组覆盖。因现有标注不完整，不生成正式 nDCG。
- [x] 分开报告模型加载、首次推理、热态 p50/p95，确认模型重复查询不重新加载，输入预算和截断日志可追查。

2026-10-27 状态：模型适配及真实实测完成，**严格质量验收未通过，不默认启用**。指定证据组覆盖上升，但留出诊断中指定正例 Recall@5 下降，且无完整正式 nDCG 标注。详见 `2026-10-27-cross-encoder-verification.md`；不得将条件性 judged-pool nDCG 当正式 nDCG。

**通过标准：** 协议测试全过；目标机达到冻结资源/延迟预算；真实 hybrid 在验收集相对 rrf 的 nDCG@5 提升且 Recall@5 不下降。未满足时不进入默认启用，保留失败样本和参数结果。**提交：** `feat: add verified cross-encoder reranking`。

### 步骤 6：暴露新工具与可选装配

**Files:** 新增 `hybrid_tools.py`；修改 `execution/runtime.py`、`execution/resources.py`、`agent/turn.py`、配置 settings/loader、s2/s3/s4 和启动校验；新增 `backend/tests/core/test_hybrid_tools.py` 并扩展对应 workflow/config 测试。

**Interfaces:** `HybridSearchArguments(query: str, limit: int = 5)`；Runtime 新增独立 `search_hybrid_knowledge(query: str, *, limit: int = 5) -> HybridSearchResult`；AgentTurn 新入口登记 hit.passage 并保留原始 ranking。旧入口签名不变。

- [ ] 新工具结果仅暴露 query、passages；每条片段给出 id/title/text/rerank_score，来源留在证据与日志，其他阶段分数留在追踪。工具说明明确混合召回并重排。
- [ ] 参数验证：空白 query、超长 query、额外字段、bool/小数 limit、超出 1..10 均拒绝。
- [ ] 配置默认关闭；关闭时工具定义与旧版一致且零新资源加载；启用但依赖不可用时启动明确失败，不假装工具可用。
- [ ] s2/s3/s4 注册新工具，继续共用既有预算；去重键使用 hybrid 命名空间，允许同一 query 在旧工具与新工具各执行一次。s4 跨轮保留这一记录。
- [ ] 测试同一片段先经旧工具再经新工具返回，证据只有一份，新工具重排顺序/分数正确；模型收到工具结果与 tool_call_id 配对，重启检查点可恢复。
- [ ] 执行 `<PY> -m pytest backend/tests/core/test_hybrid_tools.py backend/tests/core/test_config_api.py backend/tests/workflows -q`，核对数据集跳过项，不把 skip 当完整验收。

**通过标准：** 开关两种模式均有运行证据；旧工具契约与既有阶段预算保持；新工具可真实调用、引用可展开。**提交：** `feat: expose optional hybrid_search_documents tool`。

### 步骤 7：全链路对照与上线门槛

**Files:** 完善 retrieval.py、相关测试及 README；在同一运行目录保存 dense/bm25/rrf/hybrid 的配置、逐题结果、聚合结果与回归样本。

- [ ] 在同一集合、题集、机器和并发条件下跑四策略；每个策略相同热身次数，至少 3 轮热态查询，报告逐题变化与差异区间；查询重复不能被计作新增独立样本。
- [ ] 区分检索指标与生成任务通过率；额外跑新工具关闭/启用的 s2/s3/s4 场景，记录实际选了哪个工具和调用数，不能把未调用新工具的结果算成新算法验证。
- [ ] 检查旧工具默认模式无新增模型请求、现有功能无新增失败；运行 `<PY> -m pytest backend/tests -q`、ruff、basedpyright 及前端既有检查。真实 PostgreSQL/HTTP/SSE 验证单独运行并留存结果。
- [ ] 在文档写明配置、依赖、冷启动、错误行为、资源预算与关闭方式；关闭开关后验证旧工具可正常工作。

**通过标准：** 确定性正确性与旧链路回归全部通过；hybrid 验收集 nDCG@5 相对 dense 和 rrf 均提升，Recall@5 不下降，术语切片证据覆盖不下降，且达到冻结延迟预算。差异区间不足以支撑提升时，结论记为“收益未证实”，工具保留实验开关，不宣称改造成功或默认启用。

**提交：** `docs: record hybrid retrieval acceptance and rollout controls`。

## 执行与产物管理

- 每步提交前检查差异，只提交本步代码和必要测试。真实运行报告放在 var/benchmark/<run-id>/，不把凭据、模型权重或大索引提交到 Git。
- 失败时停在所属步骤定位：召回问题不靠重排掩盖，真实模型失败不靠假模型验收替代。
- 顺序为 1 → 2 → 3 → 4 → 5 → 6 → 7；单步验收不代表整体完成。
- 本次交付只验证计划的结构、路径和覆盖范围；未安装依赖、未执行上述未来测试或真实性能评测。
