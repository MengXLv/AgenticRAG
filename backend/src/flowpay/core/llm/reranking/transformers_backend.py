"""按 BGE 官方方式联合编码文本对，直接输出相关性 logits。"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from torch import Tensor

from .contracts import BatchPrediction
from .settings import RerankingSettings


class _Tokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...
    def num_special_tokens_to_add(self, pair: bool) -> int: ...
    def prepare_for_model(
        self,
        ids: list[int],
        pair_ids: list[int],
        *,
        add_special_tokens: bool,
        truncation: bool,
        return_attention_mask: bool,
    ) -> Mapping[str, list[int]]: ...


class _ModelConfig(Protocol):
    num_labels: int
    pad_token_id: int | None


class _Output(Protocol):
    logits: "Tensor"


class _Classifier(Protocol):
    config: _ModelConfig

    def to(self, device: str) -> object: ...
    def eval(self) -> object: ...
    def __call__(
        self, *, input_ids: "Tensor", attention_mask: "Tensor", return_dict: bool
    ) -> _Output: ...


class TransformersPairScorer:
    def __init__(self, settings: RerankingSettings) -> None:
        # 可选包只在真实加载时导入，旧链路和协议测试不依赖 torch。
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if settings.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Configured CUDA device is unavailable; no silent CPU fallback")
        self.settings: RerankingSettings = settings
        self.tokenizer: _Tokenizer = cast(
            _Tokenizer,
            AutoTokenizer.from_pretrained(
                settings.model_id,
                revision=settings.revision,
                cache_dir=str(settings.cache_dir),
                local_files_only=settings.local_files_only,
                trust_remote_code=False,
            ),
        )
        self.model: _Classifier = cast(
            _Classifier,
            AutoModelForSequenceClassification.from_pretrained(
                settings.model_id,
                revision=settings.revision,
                cache_dir=str(settings.cache_dir),
                local_files_only=settings.local_files_only,
                trust_remote_code=False,
                use_safetensors=True,
                torch_dtype=torch.float16 if settings.dtype == "float16" else torch.float32,
            ),
        )
        if self.model.config.num_labels != 1:
            raise ValueError("Cross-Encoder must output one score per pair")
        pad_token_id = self.model.config.pad_token_id
        if pad_token_id is None:
            raise ValueError("Cross-Encoder must define a padding token")
        self.pad_token_id: int = pad_token_id
        self.model.to(settings.device)
        self.model.eval()

    def predict(self, query: str, passages: list[str]) -> BatchPrediction:
        import torch

        query_ids = self.tokenizer.encode(query, add_special_tokens=False)
        special = self.tokenizer.num_special_tokens_to_add(pair=True)
        budget = min(self.settings.max_query_tokens, self.settings.max_length - special - 1)
        if budget < 1:
            raise ValueError("Model special tokens exceed configured input budget")
        used_query = query_ids[:budget]
        lengths: list[int] = []
        used_lengths: list[int] = []
        encoded: list[Mapping[str, list[int]]] = []
        for text in passages:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            used = ids[: self.settings.max_length - special - len(used_query)]
            lengths.append(len(ids))
            used_lengths.append(len(used))
            encoded.append(
                self.tokenizer.prepare_for_model(
                    used_query,
                    used,
                    add_special_tokens=True,
                    truncation=False,
                    return_attention_mask=True,
                )
            )
        # 手动按批次最长序列右填充，明确只截取 query/passsage 头部，不依赖隐式截断。
        size = max(len(row["input_ids"]) for row in encoded)
        inputs = {
            "input_ids": torch.tensor(
                [
                    row["input_ids"] + [self.pad_token_id] * (size - len(row["input_ids"]))
                    for row in encoded
                ],
                device=self.settings.device,
            ),
            "attention_mask": torch.tensor(
                [
                    [1] * len(row["input_ids"]) + [0] * (size - len(row["input_ids"]))
                    for row in encoded
                ],
                device=self.settings.device,
            ),
        }
        with torch.inference_mode():
            logits = (
                self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    return_dict=True,
                )
                .logits.reshape(-1)
                .float()
                .cpu()
            )
        scores = [float(logits[i]) for i in range(logits.numel())]
        return BatchPrediction(scores, len(query_ids), len(used_query), lengths, used_lengths)
