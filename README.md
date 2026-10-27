# FlowPay Agentic RAG 教学项目

围绕一家虚构支付公司 FlowPay 的客服场景，分阶段（s1—s4）手写 Agentic RAG：
从一次检索一次回答，到模型自己决定检索几轮、用哪个工具。

## 前置条件

| 需要什么 | 说明 |
| --- | --- |
| Docker Desktop | 数据库、嵌入服务、后端和前端都在容器里跑 |
| 聊天模型密钥 | 任何 OpenAI 兼容服务都行，默认用 DeepSeek |
| 教学数据包 | 单独下发的 `flowpay-dataset-v1.zip` 或 `.tar.gz`，不随仓库分发 |
| 约 8 GB 磁盘 | 镜像约 6.5 GB（其中 ollama 自己占 4.5 GB）、嵌入模型 640 MB、数据包解开后约 160 MB |

本机**不需要**预装 Python、Node 或 Ollama。嵌入模型由 `make up` 自动下载。

## 快速开始

```bash
cp .env.example .env     # 然后填下面两项
make up                  # 首次视网络而定，主要花在拉 4.5 GB 的 ollama 镜像和 640 MB 的模型
```

`.env` 里只有两项必填：

- `FLOWPAY_MODEL_API_KEY`：聊天模型密钥
- `FLOWPAY_DATASET_ROOT`：数据包位置，填 `.zip` / `.tar.gz` 都行，首次启动自动解开

跑完打开 <http://127.0.0.1:5173/>，接口在 <http://127.0.0.1:8000/>。

想跑别的阶段：`STAGE=s2 make up`（可选 s1—s4）。

## `make up` 做了什么

每一步都可以重复执行，已经完成的会跳过：

1. 校验 `.env`，把数据包解开成绝对路径交给 compose
2. 起 PostgreSQL（pgvector）
3. 导入知识库：100 篇文档、712 个向量，直接从数据包的 dump 恢复，不用自己算
4. 起 Ollama，并确保嵌入模型 `qwen3-embedding:0.6b` 就位
5. 核对模型 digest 与 `backend/configs/local.toml` 的 `embedding.revision` 是否一致
6. 起后端与前端

模型存在 `flowpay_ollama` 这个 docker 卷里，`make down` 不会删，所以只下载一次。

## 为什么必须有嵌入服务

数据包省掉的是**建库**：100 篇文档切成 712 块，每块的向量都算好了。

但用户每提一个问题，这个问题本身也要现场嵌成向量，才能和库里那 712 个向量比相似度。
这一步没法预先做——没人知道用户会问什么。所以 Ollama 是运行期的必需服务，
不是可选加速项。

两边的向量必须出自同一个模型才可比。`KnowledgeClient.search` 在嵌入之前会用
`same_vector_space` 核对集合记录的模型身份（provider / model / revision / base_url /
dimensions / request_options），对不上直接报错，不会让检索悄悄退化成噪声。

这也是为什么**嵌入服务的端口必须是 11434**：集合里的向量是在该端口建的，
端口参与向量空间身份比较（主机名不参与，`127.0.0.1`、`host.docker.internal`、
`ollama` 会被归一成同一个部署）。

## 检索对照评测

`scripts/benchmark/retrieval.py` 可在同一 K 下重放 Dense、BM25、RRF 与 Hybrid，
区分召回缺失、RRF 截断和最终选择造成的证据缺口，并按场景族估计配对差异区间。
运行方式、真实测量范围和当前瓶颈见
[四策略检索对照记录](docs/superpowers/plans/2026-10-27-retrieval-comparison-verification.md)。

## 提示词维护

各阶段系统提示词、动态上下文、工具说明与公共纠错指令已独立到
[`backend/src/flowpay/core/prompts/`](backend/src/flowpay/core/prompts/README.md) 的 TOML 文件。
修改后重启后端；新建会话使用更新后的系统提示词。

## 可选混合检索工具

`hybrid_search_documents` 已接入 s2–s4，与原有 `search_documents`、`read_document`
并存，共用阶段的工具预算。默认关闭；s1 仍使用原来的固定检索流程。

在已能运行原检索的后端环境中，安装 `uv sync --extra reranking`，预先准备
`backend/configs/reranking.toml` 固定版本的本地权重，然后取消
`backend/configs/local.toml` 中 `hybrid_reranking_config = "reranking.toml"` 的注释并重启。
也可设置 `FLOWPAY_HYBRID_RERANKING_CONFIG=reranking.toml`；路径相对主配置文件解析。
该推理配置当前使用 CUDA/float16，需要可用的 NVIDIA GPU。
默认 Docker 镜像未安装 reranking extra，也未分配 GPU，需准备对应运行环境后再启用。

新工具执行语义/BM25 各 Top-50 → RRF Top-30 → 预训练 Cross-Encoder 按分数排序 → Top-K，
默认返回 7 条，允许显式指定 1–10 条。模型与索引在启动时准备并跨轮复用；
缺少权重、依赖、设备或集合时启动失败，聊天请求不下载模型、不现场训练。
删除配置项及环境覆盖后重启即可关闭。

当前最终排序仅使用 Cross-Encoder 分数，RRF 仅筛选候选；
[排序切换验证](docs/superpowers/plans/2026-10-27-cross-encoder-final-order-verification.md)与
[默认 Top-7 的效果及上下文增量](docs/superpowers/plans/2026-10-27-top-seven-verification.md)。

验证范围及运行证据见 [工具接入验收记录](docs/superpowers/plans/2026-10-27-tool-injection-verification.md)。

## 常用命令

```
make up        起全栈（BUILD=1 强制重建镜像）
make down      停容器，数据卷保留
make logs      跟踪后端与前端日志
make help      列出全部目标
```

## 换聊天模型

在 `.env` 里填这两项，留空则用 `backend/configs/local.toml` 的默认值：

```bash
FLOWPAY_MODEL_BASE_URL='https://api.openai.com/v1'
FLOWPAY_MODEL_NAME='gpt-4.1-mini'
```

嵌入模型不要换：数据包里的向量和它绑定，换了就得自己重建整个集合。

## 宿主机直跑（进阶）

要改 `backend/src/flowpay/core/` 或者需要断点调试时走这条，否则上面四个命令就够了。
这条路径需要本机自己装好 uv、Node >= 22.12 和 **Ollama**：

```bash
make setup SETUP_ARGS=--download-model   # 装依赖、建库，并下载嵌入模型到宿主机
make run STAGE=s1                        # 后端
make frontend                            # 前端
```

`make setup` 只起 PostgreSQL 容器，嵌入服务用宿主机那个（`127.0.0.1:11434`）。
`make doctor` 可以只做只读检查，不装东西也不调用聊天模型。

## 常见问题

**提问时报嵌入相关的错**  
`docker compose ps` 看 ollama 是否在跑，`make logs` 看后端报什么。
正常情况下 `make up` 已经确保模型就位，不该出现。

**启动时提示 digest 与 revision 不符**  
上游更新了同名模型标签。新模型和数据包里的向量不在同一空间，不能混用；
请使用数据包配套的模型版本，不要改 `revision` 绕过检查。

**端口被占**  
需要 5173、8000、55432 三个端口。Ollama 的 11434 不对外发布，
不会和你本机已装的 Ollama 冲突。

**想从头再来**  
`docker compose down -v` 删掉数据卷（知识库和嵌入模型都会没，下次 `make up` 重新准备）。
