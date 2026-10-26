"""融合两路已有排名；不访问模型或数据库，不改变原文及原分数。"""

import math
from dataclasses import replace

from .contracts import Passage
from .hybrid_contracts import LexicalHit, RankedCandidate


def fuse_rrf(
    dense: list[Passage], lexical: list[LexicalHit], rank_constant: int = 60
) -> list[RankedCandidate]:
    """按去重后的 1 起始排名累加 1/(constant+rank)，返回完整融合列表。

    同路重复只保留首次命中且不占用新的排名。身份由集合、文档和块号组成，
    相同身份的原文及来源必须一致，允许 Passage.score 因召回方式不同而不同。
    调用方在此结果上截取 Top-N，避免融合时丢失候选导致诊断信息不足。
    """
    if type(rank_constant) is not int or rank_constant < 1:
        raise ValueError("RRF rank constant must be a positive integer")
    candidates: dict[tuple[str, str, int], RankedCandidate] = {}
    collection_id: str | None = None
    lanes = (
        (True, [(p, p.score) for p in dense]),
        (False, [(hit.passage, hit.score) for hit in lexical]),
    )
    for is_dense, entries in lanes:
        seen: set[tuple[str, str, int]] = set()
        for passage, score in entries:
            if not math.isfinite(score):
                raise ValueError("Retrieval scores must be finite")
            source = passage.source
            if collection_id is None:
                collection_id = source.collection_id
            if source.collection_id != collection_id:
                raise ValueError("Cannot fuse candidates from different collections")
            key = (source.collection_id, source.doc_id, source.chunk_no)
            previous = candidates.get(key, RankedCandidate(passage))
            original = previous.passage
            if (original.id, original.title, original.text, original.source) != (
                passage.id,
                passage.title,
                passage.text,
                source,
            ):
                raise ValueError("Conflicting evidence for the same passage identity")
            if key in seen:
                continue
            seen.add(key)
            rank = len(seen)
            fused_score = previous.rrf_score + 1.0 / (rank_constant + rank)
            if is_dense:
                candidates[key] = replace(
                    previous, dense_rank=rank, dense_score=score, rrf_score=fused_score
                )
            else:
                candidates[key] = replace(
                    previous, bm25_rank=rank, bm25_score=score, rrf_score=fused_score
                )
    return sorted(
        candidates.values(),
        key=lambda row: (-row.rrf_score, row.passage.source.doc_id, row.passage.source.chunk_no),
    )
