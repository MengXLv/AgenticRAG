# 最终排序切换为纯 Cross-Encoder

本记录保留切换时 Top-6 的验证结果。随后默认返回量已按用户要求改为 Top-7，见 [Top-7 验证记录](2026-10-27-top-seven-verification.md)；纯模型排序策略不变。

按用户明确选择，当前链路为：语义/BM25 各 Top-50 → RRF Top-30 → Cross-Encoder 按原始分数降序 → Top-K。默认 K=6，显式 1–10 保持原样。

## 实现与协议

- `HybridKnowledgeClient` 移除 rerank_weight 构造参数，线上固定纯模型排序；RRF 只筛选候选，不参与最终分数或同分打破规则。
- `rank_candidates` 默认模型权重为 1；直接使用原始 logits 排序，避免归一化导致极小分差消失。模型同分时按文档编号、分块编号稳定排序。
- `final_score` 等于 `rerank_score`。两者保留原工具结果字段结构，原始 RRF 分数、位次及来源继续保留在追踪，`Passage.score` 不变。
- 请求追踪的 final_ranking 改为 cross_encoder，不再记录线上融合权重。工具说明和 README 同步说明最终排序方式。
- 显式加权排序仅保留供历史实验对照；`scripts/benchmark/ranking_policy.py` 默认 0.75 是历史修复重放，不代表线上策略。
- 四策略评测默认随线上切换为纯模型。热态测量新增 measured_rerank_weight 绑定；缺失或与当前策略不一致时拒绝统计。旧融合方案耗时不能标为新策略时延。

原工具、新工具开关、召回数量、模型版本、输入截断、预算、来源与取消机制均沿用此前实现；文档切割未修改。

## 冻结分数重放

使用此前真实 PostgreSQL 候选与 CUDA logits，固定相同 24 道已查看诊断题、15 个场景族及 Top-6，仅替换最终排序策略。

| 指标 | 历史 0.75 融合 | 当前纯 Cross-Encoder |
| --- | ---: | ---: |
| 平均指定证据组覆盖@6 | 64.58% | 70.83% |
| 指定正例 Recall@6 | 44.73% | 49.00% |
| 完整覆盖全部指定组的题目比例 | 50.00% | 54.17% |
| 已覆盖指定证据组数量 | 28/48 | 30/48 |
| Top-30 内但最终未完整覆盖的指定组 | 14 | 12 |
| 相对 RRF 的逐题指定组覆盖退化数 | 0 | 1 |

保留逐题退化记录。上述是相同部分标签的已见题重放，不是新留出测试；正式 nDCG 仍为 null，未重新测量 GPU/在线延迟。

新产物为 `var/benchmark/cross-encoder-final-order/input.json`、`report.json`。输入已移除旧融合时延，performance 为 null。源输入 SHA256 为 `d8aa5352b8cb50a4c9aafc7217903f0bc8f51a6375d1ba7660a7a75428e10654`。

```powershell
backend/.venv/Scripts/python.exe -m scripts.benchmark.retrieval --input var/benchmark/cross-encoder-final-order/input.json --output var/benchmark/cross-encoder-final-order/replayed-report.json
```

## 验证范围

新增反例确保：RRF 首位与模型首位冲突时跟随模型、模型同分不回退 RRF、最终分数等于原始 logits。真实 SessionManager → s2 → 工具 → 生产协调器集成测试检查片段顺序、模型分数和完整引用回填。

历史回归分数继续用显式 0.75 重放，名称已注明历史策略；默认 Top-6 与显式 Top-5 仍有测试。旧测量权重和当前策略不一致、或缺少测量策略信息时均拒绝使用对应耗时。

最终选定范围回归：**299 passed、4 skipped、1 deselected**。Ruff、格式和源码及变更测试的 BasedPyright 检查通过，零错误、零警告。独立审查未发现重要缺陷。

测试范围沿用此前 Windows 限制：4 项为 opt-in PostgreSQL，取消选择项需要符号链接权限；收集阶段排除依赖 fcntl 的 test_config_api.py、test_session_delete.py、test_routing.py，未计为通过。s2/s3/s4 阶段回归设置了业务数据集路径，均在选定范围中执行。
