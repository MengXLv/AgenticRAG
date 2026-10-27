"""本项目使用的可选 transformers 工厂；结果由本地窄协议描述。"""

class AutoTokenizer:
    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: object) -> object: ...

class AutoModelForSequenceClassification:
    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: object) -> object: ...
