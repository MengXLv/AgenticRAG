"""从安装包加载 UTF-8 提示词；保留空白并严格校验模板参数。"""

import tomllib
from functools import lru_cache
from importlib.resources import files
from string import Formatter

from flowpay.core.json import require_object, require_text

GROUPS = frozenset({"agent", "tools", "embedding"})


@lru_cache(maxsize=len(GROUPS))
def _templates(group: str) -> dict[str, str]:
    if group not in GROUPS:
        raise ValueError(f"Unknown prompt group: {group}")
    resource = files("flowpay.core.prompts").joinpath(f"{group}.toml")
    values = require_object(tomllib.loads(resource.read_text(encoding="utf-8")))
    templates = {key: require_text(value) for key, value in values.items()}
    if not templates or any(not text.strip() for text in templates.values()):
        raise ValueError(f"Empty prompt resource: {group}")
    return templates


def prompt(group: str, key: str, **values: str) -> str:
    """加载并填充具名模板；缺失资源或参数错误直接抛出，不回退到内嵌文本。"""
    templates = _templates(group)
    if key not in templates:
        raise ValueError(f"Unknown prompt key: {group}.{key}")
    text = templates[key]
    fields: set[str] = set()
    for _, name, spec, conversion in Formatter().parse(text):
        if name is None:
            continue
        if not name.isidentifier() or spec or conversion:
            raise ValueError(f"Unsupported prompt placeholder: {group}.{key}: {name}")
        fields.add(name)
    if fields != values.keys():
        raise ValueError(
            f"Prompt parameters for {group}.{key}: expected {sorted(fields)}, got {sorted(values)}"
        )
    return text.format_map(values)
