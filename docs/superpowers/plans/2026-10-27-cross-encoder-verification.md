# Cross-Encoder 步骤 5：实现及实测

结论：预训练重排器已接入并在 RTX 4060 Laptop 上实际运行；没有训练或微调。保留实验状态，未注册新工具或改动原工作流。当前严格质量门槛未通过。

## 固定推理配置

| 项目 | 配置 |
| --- | --- |
| 模型 | BAAI/bge-reranker-v2-m3 |
| revision | 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e |
| 设备 | RTX 4060 Laptop，8 GB；CUDA FP16 |
| 库版本 | torch 2.6.0+cu124、transformers 4.57.6、tokenizers 0.22.2 |
| 最大输入 / 查询预算 | 512 / 128 token |
| 批大小 / 推理超时 | 4 / 30 秒 |
| 加载策略 | 仅本地固定 revision、safetensors、trust_remote_code=False |
| 分数 | 原始单 logit，不当作概率或答案置信度 |

实现采用[模型作者提供的文本对推理方式](https://huggingface.co/BAAI/bge-reranker-v2-m3)。Windows 可选依赖使用 [PyTorch 官方 CUDA 12.4 安装源](https://pytorch.org/get-started/previous-versions/)，对应本机驱动 566.26；不升级系统驱动。

`core/llm/reranking` 中配置、推理后端、异步客户端分离。客户端复用一个模型实例并串行占用设备；批间检查取消，等待当前同步推理退出后写终态。排队任务也可取消。模型加载失败不缓存半成品，不静默回退 CPU。

查询保留头部最多 128 token；片段保留剩余输入预算内的头部；显式计入模型配对特殊 token，右侧 padding。日志保存查询和片段的原始/实际 token 数。超过时限后仍可能因等待在途推理清理而超过 30 秒，不宣称硬杀 GPU 任务。

## 真实候选与标注口径

从现有 ready 集合以 reader 账号读取，文档、切割和文档向量未改动。24 道原样可见查询经已有 Qwen 模型生成查询向量，每路取 50 条，经 RRF 固定每题最多 30 条。模型摘要先校验；查询内容不混入标签或答案。

使用已有 dev-v1 的 24 题、192 条部分相关性判断。15 个场景族按预先固定规则切分为 development 13 题、holdout 11 题；同一场景族不跨分区。这里的 holdout 仍来自原 dev，不是官方 test，也不是独立复核后的穷尽判断。输入、查询、qrels 与模型配置的 SHA256 在推理前保存于 freeze 文件。未按结果调参。

正式 `ndcg5` 全部为 null。`judged_pool_ndcg5` 只比较候选中已判断片段的相对排序，会过滤未知片段，不能解释为实际 Top-5 nDCG。主报告列明指定证据组覆盖、指定正例召回、已知正例 MRR 与未判断结果数量。

## 最终运行结果

| 指标 | RRF | 重排后 |
| --- | ---: | ---: |
| 开发诊断：平均指定证据组覆盖@5 | 64.10% | 70.51% |
| 开发诊断：全部必需组覆盖的题目比例@5 | 38.46% | 53.85% |
| 开发诊断：指定正例 Recall@5 | 42.18% | 49.62% |
| 留出诊断：平均指定证据组覆盖@5 | 43.94% | 56.06% |
| 留出诊断：全部必需组覆盖的题目比例@5 | 36.36% | 45.45% |
| 留出诊断：指定正例 Recall@5 | 29.94% | 26.90% |
| 留出诊断：已知正例 MRR@5 | 0.3470 | 0.4848 |

指定正例包含多条可替代证据，而证据组按 any-of/all-of 判断问题所需证据；因此二者可能方向不同。仍按原计划保留“Recall 不下降”的门槛，不因为另一指标改善就宣称通过。

覆盖退化已人工查看：

- **FP-BM-SR-0081**：RRF 前五包含“错误本身不能证明套餐降级，需确认环境、套餐、Usage、共享额度及重试放大”的诊断规则；模型更偏向相似历史案例，将该直接规则挤出 Top-5，证据组覆盖从 0.5 降到 0。
- **FP-BM-MD-0102**：RRF 前五包含 succeeded 是终态、不允许 confirm 的状态限制表；重排后挤出该片段，指定证据组覆盖从 1 降到 0.5。

这些样例没有发生输入截断，不能将退化归因于 token 上限。正式全量 nDCG 尚不可计算，现有部分标注也不足以证明所有未判断结果不相关。

## 性能与后端校验

最终运行 24 题，每题 30 候选；每题先推理一次，再计 3 次热态推理（72 次热态样本）。

| 项目 | 最终运行 |
| --- | ---: |
| 模型加载次数 | 1 |
| 模型加载（不含 torch import） | 5.80 s |
| 首次推理 | 623.6 ms |
| 热态 p50 / p95 | 223.7 / 395.9 ms |
| GPU 峰值 allocated（含模型） | 1,405,673,984 bytes，约 1.31 GiB |
| 查询 / 片段截断次数 | 0 / 0 |

早期同参数运行 p95 为 259.2 ms，最终重复运行为 395.9 ms，质量排序指标一致。分位数有本机运行波动，不承诺生产 SLA。模型下载耗时与权重文件体积不计入加载和推理耗时。

独立真实后端检查：短文本对与官方 tokenizer 配对编码的 logits 最大差异为 **0.0**；HTTP 410 规则分数高于 HTTP 429 干扰规则。长输入实测 query 201→128 token、passage 1501→380 token，配合 4 个特殊 token 正好 512。

## 测试与审查

- 本步 23 项确定性测试：批次下标、加载复用、并发调用、空输入、加载失败重试、非法分数、超时、重复取消、配置校验、部分标注指标和分区隔离。
- 最终扩展回归 **244 passed, 4 skipped, 1 deselected**；ruff、格式、basedpyright 与 uv lock 检查通过。
- 独立审查发现重复使用 output 会累加历史事件的 P2；复现测试先得到 model_loads=2，改为每次独立会话后通过。报告现在携带 event_session_id。
- 未重复收集已知依赖 fcntl 的三个 Windows 不兼容文件；符号链接权限用例仍单独排除，真实 PostgreSQL 测试默认跳过。最终回归计数以交付消息和实施日志为准。

## 使用及复现

在仓库根目录安装可选依赖：

```text
uv sync --project backend --extra reranking
```

模型下载到独立配置中的缓存目录，固定 revision；常规推理默认 local_files_only=true，不自动下载。`backend/configs/reranking.toml` 仅供显式验证/装配，不会启用聊天工具。CPU 使用时同时改 device=cpu、dtype=float32。

模型预取示例（模型权重不提交 Git）：

```python
from huggingface_hub import snapshot_download
snapshot_download(
    "BAAI/bge-reranker-v2-m3",
    revision="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
    cache_dir="var/models/huggingface",
    allow_patterns=["*.json", "sentencepiece.bpe.model", "model.safetensors"],
)
```

```text
<PY> -m pytest backend/tests/core/test_reranking.py backend/tests/scripts/test_reranking_benchmark.py -q
<PY> scripts/benchmark/reranking.py --input var/benchmark/reranking-step5-input.json --config backend/configs/reranking.toml --output var/benchmark/reranking-step5-final --repeats 3
```

本地原始产物：`reranking-step5-input.json`、`reranking-step5-freeze.json`、`reranking-step5-query-vectors.json`，以及 `var/benchmark/reranking-step5-final/report.json` 和 events.sqlite。真实后端校验在 `var/benchmark/reranking-step5/real-backend-check.json`。这些文件保存在 var，不提交 Git。

下一步质量工作：补审各策略 Top-K 的未判断片段及退化样例，固定新标注版本后重评。当前不自动训练、不默改验收门槛，也不默认启用重排工具。
