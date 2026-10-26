"""中文分词与完整技术标识共用一套文档、查询处理。"""

import re

from jieba import Tokenizer

# 英文标识保留下划线、连字符和版本号；中文交给词典分词，标点不进入索引。
TERMS = re.compile(r"[A-Za-z0-9]+(?:[._:/-][A-Za-z0-9]+)*|[\u3400-\u4dbf\u4e00-\u9fff]+")
SEGMENTER = Tokenizer()


def tokenize(text: str) -> list[str]:
    """保留技术词整体并统一英文大小写，不使用 embedding 的查询指令。"""
    tokens: list[str] = []
    for match in TERMS.finditer(text):
        term = match.group()
        if term.isascii():
            tokens.append(term.casefold())
        else:
            tokens.extend(SEGMENTER.lcut(term, HMM=False))
    return tokens
