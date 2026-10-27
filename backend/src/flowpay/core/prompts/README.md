# 提示词资源

阶段系统提示词和动态上下文位于 `../../workflows/s1.py` 至 `s4.py`，与执行流程一起阅读和修改。
本目录只维护公共工具说明、纠错指令和嵌入模板。

| 文件 | 内容 |
| --- | --- |
| `agent.toml` | 预算收尾、引用纠正、输出纠正指令 |
| `tools.toml` | 工具说明、参数说明、空结果与限制反馈 |
| `embedding.toml` | Qwen 查询指令与输入模板 |

统一调用方式：

```python
from flowpay.core.prompts import prompt

feedback = prompt("agent", "output_correction", error=str(error))
```

模板只支持 `{name}` 具名占位符；代码必须提供完全匹配的参数。字面花括号使用
`{{` 和 `}}`。参数中的花括号、JSON、中文和换行按原样插入，不会再次解释为模板。
不要在资源中加入密钥或运行时数据。

资源通过 `importlib.resources` 从 Python 包读取，随 wheel 分发，不依赖启动目录。
每个进程缓存一次，修改后须重启后端。新系统指令对新建会话生效；已有会话保留
检查点中的系统消息。普通 UI 文案、字段枚举、预算数值和错误类型仍由代码管理。

嵌入指令也会影响查询向量；修改 `embedding.toml` 属于检索行为变更，应重新评测。
阶段提示词迁回代码时保留原文本和空白，没有修改查询输入或模型参数。
