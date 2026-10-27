# 提示词资源

模型指令与流程代码分别维护。修改相应 TOML 文本，无需修改工作流实现。

| 文件 | 内容 |
| --- | --- |
| `s1.toml` | 朴素 RAG 系统指令、资料与历史对话模板 |
| `s2.toml` | 自主检索系统指令 |
| `s3.toml` | 多源调查系统指令 |
| `s4.toml` | 多轮协作系统指令、条件变更与追问反馈 |
| `agent.toml` | 预算收尾、引用纠正、输出纠正指令 |
| `tools.toml` | 工具说明、参数说明、空结果与限制反馈 |
| `embedding.toml` | Qwen 查询指令与输入模板 |

统一调用方式：

```python
from flowpay.core.prompts import prompt

system = prompt("s2", "system")
feedback = prompt("agent", "output_correction", error=str(error))
```

模板只支持 `{name}` 具名占位符；代码必须提供完全匹配的参数。字面花括号使用
`{{` 和 `}}`。参数中的花括号、JSON、中文和换行按原样插入，不会再次解释为模板。
不要在资源中加入密钥或运行时数据。

资源通过 `importlib.resources` 从 Python 包读取，随 wheel 分发，不依赖启动目录。
每个进程缓存一次，修改后须重启后端。新系统指令对新建会话生效；已有会话保留
检查点中的系统消息。普通 UI 文案、字段枚举、预算数值和错误类型仍由代码管理。

嵌入指令也会影响查询向量；修改 `embedding.toml` 属于检索行为变更，应重新评测。
本次迁移保留原文本和空白，没有修改查询输入或模型参数。
