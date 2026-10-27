"""真实管理器、检查点和工具循环；受控检索不评价模型检索质量。"""

import json
from pathlib import Path

import pytest

from flowpay.core.config import Settings
from flowpay.core.data_sources import DataSources
from flowpay.core.event_log import EventStore
from flowpay.core.execution.manager import SessionManager
from flowpay.core.execution.resources import RuntimeResources
from flowpay.core.json import JsonObject, parse_json, require_list, require_object
from flowpay.core.knowledge.hybrid_contracts import HybridHit, HybridRanking, HybridSearchResult
from flowpay.core.llm.embedding.ollama import retrieval_input
from flowpay.dataset import dataset_path, snapshot_ready
from flowpay.router import create_stage

from .fixtures import Model, Search, completion, grounded, hit, search_call


def call(identifier: str, query: str) -> JsonObject:
    return {
        "id": identifier,
        "type": "function",
        "function": {
            "name": "hybrid_search_documents",
            "arguments": json.dumps({"query": query}),
        },
    }


class Hybrid:
    def __init__(self) -> None:
        self.queries: list[tuple[str, str, int]] = []

    async def search(self, collection_id: str, query: str, limit: int = 6) -> HybridSearchResult:
        self.queries.append((collection_id, query, limit))
        return HybridSearchResult(
            [HybridHit(hit(), HybridRanking(0.8, 1, 3.0, 1, 0.03, -7.0, 1, 1, 0.95))]
        )


@pytest.mark.parametrize("stage", ["s2", "s3", "s4"])
async def test_registration_evidence_budget_and_checkpoint(tmp_path: Path, stage: str) -> None:
    root = dataset_path()
    if stage != "s2" and (root is None or not snapshot_ready()):
        pytest.skip("business snapshot unavailable")
    model, old, hybrid = Model(), Search(), Hybrid()
    settings = Settings(tmp_path, tmp_path, workflow=stage, knowledge_collection_id="fixture")
    sources = (
        DataSources.from_config(Path(__file__).resolve().parents[2] / "policies/sources.toml", root)
        if root is not None and stage != "s2"
        else None
    )
    resources = RuntimeResources(
        settings, sources=sources, retriever=old.for_run, hybrid_retriever=lambda _: hybrid
    )
    manager = SessionManager(
        EventStore(tmp_path / "events.sqlite"),
        create_stage(stage, model, retrieval_input),
        settings,
        resources,
    )
    try:
        model.responses = [
            completion(calls=[search_call("old", "HTTP 410")]),
            completion(calls=[call("new", "  HTTP   410  "), call("repeat", "HTTP 410")]),
            grounded(),
        ]
        sid = (await manager.create_session())["id"]
        await manager.submit_message(sid, "first", "HTTP 410 后还重试吗？")
        await manager.tasks[sid]
        view = await manager.get_session(sid)
        assert view["status"] == "completed"
        assert hybrid.queries == [("fixture", "HTTP 410", 6)]
        assert old.queries == [(retrieval_input("HTTP 410"), 10)]
        assert view["state"]["tool_calls"] == 2
        assert len(require_list(view["state"]["evidence"])) == 1
        evidence = view["messages"][-1].get("evidence", [])
        assert (
            require_object(require_object(evidence[0])["source"])["path"] == "knowledge/endpoint.md"
        )
        functions = [
            require_object(require_object(t)["function"])
            for t in require_list(model.requests[0]["tools"])
        ]
        assert {f["name"] for f in functions} >= {
            "search_documents",
            "read_document",
            "hybrid_search_documents",
        }
        tools = [
            require_object(m)
            for m in require_list(view["state"]["messages"])
            if require_object(m)["role"] == "tool"
        ]
        assert [t["tool_call_id"] for t in tools] == ["old", "new", "repeat"]
        payload = require_object(parse_json(str(tools[1]["content"])))
        assert require_object(require_list(payload["passages"])[0])["rerank_score"] == -7.0
        assert "source" not in require_object(require_list(payload["passages"])[0])
        # 新管理器从同一 SQLite 重新恢复原生消息和证据。
        await manager.shutdown()
        manager = SessionManager(
            EventStore(tmp_path / "events.sqlite"),
            create_stage(stage, model, retrieval_input),
            settings,
            resources,
        )
        if stage == "s4":
            model.responses = [completion(calls=[call("next", "HTTP 410")]), grounded()]
            await manager.submit_message(sid, "second", "再查一次同一个条件")
            await manager.tasks[sid]
            assert hybrid.queries == [("fixture", "HTTP 410", 6)]
            assert (await manager.get_session(sid))["state"]["tool_calls"] == 0
        else:
            model.responses = [grounded()]
            await manager.submit_message(sid, "second", "依据刚才的资料继续说明")
            await manager.tasks[sid]
        restored = str(model.requests[-1]["messages"])
        assert "rerank_score" in restored and "-7.0" in restored
    finally:
        await manager.shutdown()
        resources.close()


async def test_new_and_old_tools_share_one_budget_even_in_a_batch(tmp_path: Path) -> None:
    model, old, hybrid = Model(), Search(), Hybrid()
    settings = Settings(tmp_path, tmp_path, workflow="s2", knowledge_collection_id="fixture")
    resources = RuntimeResources(settings, retriever=old.for_run, hybrid_retriever=lambda _: hybrid)
    manager = SessionManager(
        EventStore(tmp_path / "e.sqlite"),
        create_stage("s2", model, retrieval_input),
        settings,
        resources,
    )
    model.responses = [
        completion(
            calls=[
                search_call("a", "old"),
                call("b", "new"),
                call("c", "other"),
                call("d", "blocked"),
            ]
        ),
        grounded(),
    ]
    try:
        sid = (await manager.create_session())["id"]
        await manager.submit_message(sid, "first", "HTTP 410")
        await manager.tasks[sid]
        view = await manager.get_session(sid)
        assert view["status"] == "completed"
        assert view["state"]["tool_calls"] == 3
        assert [query for _, query, _ in hybrid.queries] == ["new", "other"]
        messages = [
            require_object(m)
            for m in require_list(view["state"]["messages"])
            if require_object(m)["role"] == "tool"
        ]
        assert [m["tool_call_id"] for m in messages] == ["a", "b", "c", "d"]
        assert "预算已耗尽" in str(messages[-1]["content"])
    finally:
        await manager.shutdown()
        resources.close()
