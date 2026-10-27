"""固定模型身份和推理预算，不在普通聊天请求中下载权重。"""

import math
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from pydantic import TypeAdapter

from flowpay.core.json import require_object, require_text


@dataclass(frozen=True)
class RerankingSettings:
    model_id: str
    revision: str
    device: Literal["cpu", "cuda"] = "cpu"
    dtype: Literal["float32", "float16"] = "float32"
    max_length: int = 512
    max_query_tokens: int = 128
    batch_size: int = 4
    timeout_seconds: float = 30
    cache_dir: Path = Path("var/models/huggingface")
    local_files_only: bool = True

    @classmethod
    def load(cls, path: Path) -> "RerankingSettings":
        """从独立推理配置读取，缓存目录相对配置文件解析。"""
        with path.open("rb") as stream:
            raw = require_object(cast(object, tomllib.load(stream)))
            fields = dict(require_object(raw["reranking"]))
        if unknown := set(fields) - set(cls.__dataclass_fields__):
            raise ValueError(f"Unknown reranking fields: {sorted(unknown)}")
        fields["cache_dir"] = (
            path.resolve().parent
            / require_text(fields.get("cache_dir", "../../var/models/huggingface"))
        ).resolve()
        return TypeAdapter(cls).validate_python(fields)

    def __post_init__(self) -> None:
        if not self.model_id.strip() or not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("Model identity must include a full pinned revision")
        if self.device not in {"cpu", "cuda"} or self.dtype not in {"float16", "float32"}:
            raise ValueError("Unsupported reranking device or dtype")
        if self.device == "cpu" and self.dtype != "float32":
            raise ValueError("CPU inference requires float32")
        for value, low, high in (
            (self.batch_size, 1, 32),
            (self.max_length, 16, 8192),
            (self.max_query_tokens, 1, self.max_length - 5),
        ):
            if type(value) is not int or not low <= value <= high:
                raise ValueError("Invalid batch_size or token budget")
        if (
            isinstance(self.timeout_seconds, bool)
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive")
