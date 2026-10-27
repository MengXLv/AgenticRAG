# 默认 Top-7 调整

用户指定默认返回 Top-7。当前链路为语义/BM25 各 Top-50 → RRF Top-30 → Cross-Encoder 按原始分数排序 → 默认 Top-7。显式 limit 仍严格接受 1–10，原有 search_documents 和 read_document 保持原样。

统一默认常量、能力协议、工具参数 Schema、工具提示词、运行时与工作流调用均已同步。历史 ranking_policy 评测脚本显式固定历史 K=6、weight=0.75，避免默认值变化篡改历史对照口径。四策略评测跟随当前默认 K=7。

## 同一冻结结果的 Top-6 / Top-7 对照

复用同一批 24 道既有诊断题、15 个场景族的真实候选与原始 CUDA 分数，仅扩大最终返回数量。返回顺序保持原样，Top-7 的前六条与 Top-6 完全一致。

| 指标 | Top-6 | Top-7 |
| --- | ---: | ---: |
| 平均指定证据组覆盖 | 70.83% | 72.22% |
| 指定正例 Recall | 49.00% | 53.55% |
| 全部指定组完整覆盖的题目比例 | 54.17% | 54.17% |
| 已覆盖指定证据组数 | 30/48 | 31/48 |
| Top-30 中仍未完整返回的指定组 | 12 | 11 |
| 平均返回片段 token | 723.63 | 848.25 |
| 平均返回正文字符 | 1,469.75 | 1,718.96 |

覆盖率增加约 1.39 个百分点、Recall 增加约 4.55 个百分点，新增完整覆盖一个证据组；完整覆盖全部组的题目数未增加。两路召回遗漏的 2 个组、RRF 截断丢失的 4 个组不受返回数量变化影响。

平均片段 token 增加 **17.22%**。token 数来自原模型 tokenizer 记录的原文长度，不包含系统提示词、工具协议或回答生成，不能直接换算为整轮费用增长。重排仍处理相同的 30 个候选，没有新增重排调用；生成上下文更长，本次没有重新测量端到端延迟。

这些是部分标签的已见诊断题重放，正式 nDCG 保持 null。不能据此推断所有问题都改善。

## 产物与验证

本地产物：`var/benchmark/top-seven/top6.json`、`top7.json`。源输入为 `var/benchmark/cross-encoder-final-order/input.json`，原文 token 记录来自 `var/benchmark/reranking-step5-final/events.sqlite`。

测试检查默认七条、显式六条/五条、工具 Schema 默认值、s2/s3/s4 收到的实际 limit、旧工具默认十条、评测返回前缀一致性。计时元数据新增 measured_limit，只有 K 与重排权重同时匹配时才允许使用热态测量，防止把 Top-6 耗时标为 Top-7 实测。

最终选定范围回归：**300 passed、4 skipped、1 deselected**。Ruff、格式与源码及变更测试的 BasedPyright 检查通过，零错误、零警告。沿用既有 Windows 测试限制：4 项为 opt-in PostgreSQL，1 项需符号链接权限；收集阶段排除依赖 fcntl 的 test_config_api.py、test_session_delete.py、test_routing.py，未计为通过。
