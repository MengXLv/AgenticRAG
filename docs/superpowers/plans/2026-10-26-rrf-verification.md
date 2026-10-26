# RRF 步骤 3 实施与验证记录

范围：新增两路排名融合及诊断回放；未接入混合检索协调层、重排或模型工具，现有工作流与工具不变。

## 实现约定

- `fuse_rrf(dense, lexical, rank_constant=60)` 按两路各自从 1 开始的排名求和 `1 / (constant + rank)`。
- 同路重复只保留首次命中，去重后重新编号；两路均命中的片段累加两次贡献。
- 身份键是 collection_id/doc_id/chunk_no；跨集合或同身份原文、标题、ID、来源冲突时拒绝融合。
- 向量分数和 BM25 分数不参与融合计算，各路分数、排名与融合分数分别保存在 RankedCandidate 中；不改变原 Passage。
- 同分按 doc_id/chunk_no 稳定排序。函数返回全部去重候选，调用方截取 Top-N。
- 诊断报告区分未召回和截断丢失；未提供标注时保持 null，不把未知写成零损失。

## 确定性验证

19 项算法测试覆盖手算公式、单路/双路空结果、重复项、稳定排序、分数隔离、非法常数、非有限分数、集合隔离、来源冲突和 Top-N。

7 项诊断测试覆盖候选丢失分类、无标注与空标注、非法截断参数、JSON 回放。与步骤 2 的测试一起运行：**51 passed**。本步改动的 ruff、格式检查、basedpyright 通过；独立审查未发现明确缺陷，独立复跑本步测试 **26 passed**。

## 真实两路召回验收

临时启动原 PostgreSQL 容器，使用已有 reader 账号只读查询集合 `flowpay_docs_v1_qwen3_06b`；复用 SQLite 事件中的两个真实历史查询向量和模型身份。存储层再次核对向量空间，向量 Top-50 的前 10 条均与历史 Top-10 一致。BM25 使用相同查询内容，仅去除 Qwen 查询指令前缀。未改文档、切割、向量或集合，未调用 Embedding 服务。

| 样例 | 向量数量 | BM25 数量 | 去重并集 | Top-N | 截断数量 |
| --- | --- | --- | --- | --- | --- |
| HTTP 410 问题 | 50 | 50 | 70 | 30 | 40 |
| 带历史的追问 | 50 | 50 | 72 | 30 | 42 |

人工查看的 HTTP 410 锚点 `doc:KB-BR-015#c004` 在第一个样例中向量排名 6、BM25 排名 2、融合排名 2，没有被截断。该锚点只是具体验收证据，不是穷举相关性标签。第二个样例未设置标签，相关证据损失记为未知；不能将两条样例外推为质量收益结论。

报告保留每条候选的两路排名、原分数、RRF 分数、是否进入 Top-N；回放核心结果与真实只读查询报告逐字段一致。

本地原始产物（不提交 Git）：

- `var/benchmark/rrf-step3-input.json`：真实两路结果，不包含向量或凭据。
- `var/benchmark/rrf-step3-live.json`：真实只读查询报告。
- `var/benchmark/rrf-step3-replay.json`：独立 CLI 回放报告。

## 复现

```text
<PY> -m pytest backend/tests/core/test_rrf.py backend/tests/scripts/test_fusion_benchmark.py -q
<PY> scripts/benchmark/fusion.py --input var/benchmark/rrf-step3-input.json --top-n 30 --output var/benchmark/rrf-step3-replay.json
```

`<PY>` 为项目虚拟环境 Python。输入是 FusionCase 的 JSON 数组；每项包含 case_id、query、dense（Passage 列表）、lexical（LexicalHit 列表），expected_ids 可省略。该命令只读取快照并输出报告，不访问模型或数据库。

本轮未重复运行已知受 Windows fcntl 和符号链接权限限制的整套后端测试，测试通过声明仅覆盖列明范围。
