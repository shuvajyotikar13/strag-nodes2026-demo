"""FastAPI service for naive and temporally grounded graph retrieval."""

import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from strag_engine import build_reasoning_graph

load_dotenv()

NEO4J_DRIVER: Any = None
REASONING_GRAPH: Any = None


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    query_time: datetime | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global NEO4J_DRIVER, REASONING_GRAPH
    try:
        required = ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "OPENAI_API_KEY")
        if all(os.getenv(key) for key in required):
            from neo4j import AsyncGraphDatabase

            NEO4J_DRIVER = AsyncGraphDatabase.driver(
                os.environ["NEO4J_URI"],
                auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
                max_connection_pool_size=50,
                max_connection_lifetime=1800.0,
                connection_liveness_check_timeout=60.0,
                connection_acquisition_timeout=5.0,
            )
            await NEO4J_DRIVER.verify_connectivity()
            REASONING_GRAPH = build_reasoning_graph(
                NEO4J_DRIVER,
                os.environ["OPENAI_API_KEY"],
            )
        yield
    finally:
        if NEO4J_DRIVER is not None:
            await NEO4J_DRIVER.close()
            NEO4J_DRIVER = None
            REASONING_GRAPH = None


app = FastAPI(title="s-TRAG Temporal Retrieval API", lifespan=lifespan)


def _services() -> tuple[Any, Any, str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if NEO4J_DRIVER is None or REASONING_GRAPH is None or not api_key:
        raise HTTPException(status_code=503, detail="Graph service is not configured")
    return NEO4J_DRIVER, REASONING_GRAPH, api_key


def _query_epoch(value: datetime | None) -> float:
    timestamp = value or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.timestamp()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok" if NEO4J_DRIVER is not None else "not_configured"}


@app.post("/compare")
async def compare(request: QueryRequest) -> dict[str, Any]:
    _, graph, _ = _services()
    timestamp = _query_epoch(request.query_time)
    temporal = await graph.ainvoke(
        {"query": request.query, "query_time": timestamp}
    )
    candidates = temporal.get("candidates", [])
    naive = candidates[0] if candidates else None
    naive_result = None
    if naive is not None:
        naive_result = {
            "state_id": naive["id"],
            "description": naive["description"],
            "semantic_score": naive["semantic_score"],
            "valid_from": naive["valid_from"],
            "valid_to": naive["valid_to"],
        }
    return {
        "naive": naive_result,
        "strag": {
            "answer": temporal.get("answer", ""),
            "state_id": (temporal.get("selected") or {}).get("id"),
            "temporal_score": (temporal.get("selected") or {}).get("tas"),
            "lineage_walked": temporal.get("lineage_walked", False),
            "citation": temporal.get("citation"),
        },
    }
