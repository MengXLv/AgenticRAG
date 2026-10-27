"""默认按 Cross-Encoder 原始分数排序；显式权重仅供历史策略对照。"""

import math

from .hybrid_contracts import HybridHit, HybridRanking, RankedCandidate, RerankHit

DEFAULT_RERANK_WEIGHT = 1.0
DEFAULT_LIMIT = 7


def validate_weight(weight: float) -> None:
    if isinstance(weight, bool) or not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("rerank weight must be finite and within 0..1")


def _normalize(values: list[float]) -> list[float]:
    # 先缩放，避免两个有限但相差极大的分数相减后溢出。
    scale = max(abs(value) for value in values) or 1.0
    scaled = [value / scale for value in values]
    low, high = min(scaled), max(scaled)
    return [(value - low) / (high - low) for value in scaled] if high > low else [0.0] * len(values)


def rank_candidates(
    candidates: list[RankedCandidate],
    ranks: list[RerankHit],
    *,
    rerank_weight: float = DEFAULT_RERANK_WEIGHT,
) -> list[HybridHit]:
    """默认按原始模型分数排序，返回完整候选供调用者按原 K 截取。

    原始 RRF/模型分数均保留；纯模型策略 final_score 等于 rerank_score，不是概率。
    显式传入其他权重可重放历史融合策略，线上协调器固定使用纯模型策略。
    候选输入必须已按 RRF 排序，ranker 输出顺序可任意但下标必须完整且唯一。
    """
    validate_weight(rerank_weight)
    if len(ranks) != len(candidates):
        raise ValueError("Rerank response must score every candidate exactly once")
    if not candidates:
        return []
    scores = [0.0] * len(candidates)
    seen: set[int] = set()
    for rank in ranks:
        index = rank.candidate_index
        if type(index) is not int or not 0 <= index < len(candidates) or index in seen:
            raise ValueError("Rerank candidate index is invalid or duplicated")
        if isinstance(rank.score, bool) or not math.isfinite(rank.score):
            raise ValueError("Rerank scores must be finite numbers")
        seen.add(index)
        scores[index] = rank.score
    rerank_order = sorted(
        range(len(candidates)),
        key=lambda i: (
            -scores[i],
            candidates[i].passage.source.doc_id,
            candidates[i].passage.source.chunk_no,
        ),
    )
    rerank_positions = {index: position + 1 for position, index in enumerate(rerank_order)}
    if rerank_weight == 1:
        final_scores = scores
    else:
        priors = _normalize([1.0 / (61 + i) for i in range(len(candidates))])
        normalized_scores = _normalize(scores)
        final_scores = [
            (1 - rerank_weight) * prior + rerank_weight * model
            for prior, model in zip(priors, normalized_scores, strict=True)
        ]
    hits: list[HybridHit] = []
    for index, row in enumerate(candidates):
        hits.append(
            HybridHit(
                row.passage,
                HybridRanking(
                    row.dense_score,
                    row.dense_rank,
                    row.bm25_score,
                    row.bm25_rank,
                    row.rrf_score,
                    scores[index],
                    index + 1,
                    rerank_positions[index],
                    final_scores[index],
                ),
            )
        )
    order = sorted(
        range(len(candidates)),
        key=lambda i: (
            -final_scores[i],
            candidates[i].passage.source.doc_id,
            candidates[i].passage.source.chunk_no,
        ),
    )
    # 端点直接复用原始排名，不能因浮点缩放丢失极小分差而改变纯策略。
    if rerank_weight == 1:
        order = rerank_order
    elif rerank_weight == 0:
        order = list(range(len(candidates)))
    return [hits[index] for index in order]
