# 重排退化修复结果

## 改动

新混合检索不再让 Cross-Encoder 完全接管最终排序。对固定 RRF Top-30 的位次倒数先验和模型 logits 分别做逐查询 min-max 归一化，最终分数为：

`0.25 × normalized_RRF_position_prior + 0.75 × normalized_model_logit`

位次先验为 `1/(60+rank)`；原向量分数、BM25 分数、RRF 分数、模型分数均保留。新增 rrf_rank、rerank_rank、final_score 供追踪。final_score 只表示当前候选池内的排序值，不是置信概率。

新混合链路默认返回 **6** 条，取代之前的 5 条；调用方显式要求 5 条时仍严格返回最多 5 条。旧 search_documents、read_document、文档切割、文档向量、模型版本和推理配置不变。没有训练模型，也没有注册新工具。

## 已知退化

| 关键证据 | 原 RRF 位次 | 纯模型位次 | 修复位次 |
| --- | ---: | ---: | ---: |
| FP-BM-SR-0081：429 诊断边界 | 4 | 6 | 6 |
| FP-BM-MD-0102：支付终态限制表 | 3 | 6 | 5 |

两条均进入新的默认返回预算。另修复纯模型极值端点：归一化不能因浮点精度坍缩改变 weight=1 的原始模型排序，端点直接使用原始排名。

## 选择过程与预算

先比较 K=5/6/8、排名融合以及归一化分数融合。初始 K=5、weight=.5 虽恢复两条问题证据，却在扩展检查中使 SR-0101 的指定组覆盖退化，因此未作为最终策略。

此前看过的 24 道题均只作诊断和调参，不再称未见测试。选择条件是：在同 K 与 RRF 比较时，平均指定组覆盖改善、指定正例召回不下降，并且没有新增逐题指定组覆盖退化；优先较小的 K。最终固定 K=6、模型权重 .75，然后另取未参与这轮选择的 8 个场景做确认。

前一轮新增的 8 题已被查看，也不算最终未见确认组。它们的初始五片段结果原样保留，没有隐藏失败或反复称为留出。

## 原 24 题诊断，同为 Top-6

统一应用现有标签及两道已知问题的独立补审标签；仍是部分标注，不是穷尽判断。

| 指标 | RRF | 纯 Cross-Encoder | 修复 |
| --- | ---: | ---: | ---: |
| 平均指定证据组覆盖@6 | 59.72% | 70.83% | 64.58% |
| 指定正例 Recall@6 | 40.18% | 49.00% | 44.73% |
| 全部指定组覆盖题目比例 | 45.83% | 54.17% | 50.00% |
| 相对 RRF 的逐题指定组覆盖退化数 | — | 1 | 0 |
| 平均输出片段 token | 673.5 | 723.6 | 725.7 |

修复选择的是减少关键证据退化的折中，未声称所有均值优于纯模型。片段数从 5 增为 6；相对原纯模型 Top-5 的平均 607.8 token，修复平均约增加 19.4%。不将增加上下文预算的收益冒充同预算提升。

## 最终新题确认，同为 Top-6

最终策略固定后，确认组为另外 8 个场景族，与此前已查看的 32 题场景族均不重叠。涉及幂等冲突/过期、Sandbox 限流、Growth 持续与突发限额、Webhook 计数、Customer 删除和 Email 身份。四道题在推理前明确幂等作用域、时间或 Production 条件，未补入隐藏版本或答案到查询。

指定证据组根据原文在推理前冻结。真实只读召回每路 50、RRF 取 30，使用同一预训练模型给全部候选打分。三种策略的 Top-6 合并去重后打乱，独立审阅者在不查看方法、分数和位次的情况下逐条判断 **66** 个片段，再统一应用到所有策略。审阅者是独立模型代理，不是人工双人金标。

| 指标 | RRF | 纯 Cross-Encoder | 修复 |
| --- | ---: | ---: | ---: |
| 平均指定证据组覆盖@6 | 87.50% | 87.50% | 87.50% |
| 指定正例 Recall@6 | 70.76% | 75.22% | 77.01% |
| 已知正例 MRR@6 | 0.8542 | 0.9167 | 0.9167 |
| 相对 RRF 的逐题指定组覆盖退化数 | — | 0 | 0 |
| Top-6 未判断片段数 | 0 | 0 | 0 |
| 平均输出片段 token | 756.4 | 776.9 | 782.1 |

新题未出现相对两种基线的指定组覆盖退化；部分题仍未覆盖全部指定证据。66 个判断仅覆盖三策略的合并 Top-6，不是全语料标签，正式全量 nDCG 继续保持未知。8 道场景题不能证明所有问题都会改善。

重排仍对同一 30 个候选推理，无额外模型调用。最终融合本身在这组样本单次最多约 0.095 ms；生成上下文增量已单独报告。

## 验证与复现

- 数值、映射、有限性、端点、模型等分、已知真实回归分数重放、默认六条及显式五条上限都有测试。
- 最终选定范围回归：**263 passed, 4 skipped, 1 deselected**；ruff、格式与 basedpyright 检查通过（零错误、零警告）。4 项为 opt-in PostgreSQL，排除项是既有 Windows 符号链接权限测试，另在收集时排除 3 个既有 fcntl 依赖文件。
- 缓存输入 SHA256、候选顺序、问题 ID 与模型排序重放必须匹配；额外标签对所有策略统一应用。支持 Windows UTF-8 BOM 标签文件。
- 独立代码审查发现极值端点 P2，已补测试先失败后通过；之后仅继续新题标签审阅，没有将修复后的自检冒充第二次独立代码审查。
- 保留既有 Windows fcntl/符号链接权限及 opt-in PostgreSQL 测试限制，最终计数见交付消息与实施记录。

```text
<PY> -m pytest backend/tests/core/test_final_ranking.py backend/tests/core/test_hybrid_knowledge.py backend/tests/scripts/test_ranking_policy.py -q
<PY> -m scripts.benchmark.ranking_policy --input var/benchmark/ranking-repair-confirmation-new-input.json --scores var/benchmark/ranking-repair-confirmation-scores/report.json --judgments var/benchmark/ranking-repair-confirmation-judgments.json --output var/benchmark/ranking-repair-confirmation-final.json
```

主要本地产物（var 下不提交 Git）：

- `ranking-repair-freeze.json`：初始候选比较范围；`ranking-repair-confirmation-freeze.json`：最终参数、新场景、题面与事先证据组。
- `ranking-repair-confirmation-new-input.json`：真实候选，SHA256 `e03a57397389bad6ea790334b8aed9a1520acd4f838fde7b5143ed60bea38e6b`。
- `ranking-repair-confirmation-scores/`：完整模型输出、版本、token 计数和事件。
- `ranking-repair-confirmation-blind-pool.json`、`ranking-repair-confirmation-judgments.json`、`ranking-repair-confirmation-review.md`：盲审材料。
- `ranking-repair-original-final.json`、`ranking-repair-confirmation-final.json`：按当前生产函数重放的统一预算结果及各输入文件摘要。

所有数据库访问使用 reader，只生成查询向量。两个临时启动的服务容器已恢复停止状态。
