import os
import time
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel
from neo4j import AsyncGraphDatabase
from strag_engine import (
    build_strag_graph,
    VECTOR_SEARCH_CYPHER,
    get_embeddings,
    get_llm,
    get_fallback_payload,
)

NEO4J_DRIVER = None
STRAG_APP = None
DB_ONLINE = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    global NEO4J_DRIVER, STRAG_APP, DB_ONLINE
    uri = os.getenv("NEO4J_URI", "")
    user = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")
    try:
        if not uri or "your-instance" in uri:
            raise RuntimeError("Placeholder NEO4J_URI detected in .env")
        NEO4J_DRIVER = AsyncGraphDatabase.driver(
            uri,
            auth=(user, password),
            max_connection_pool_size=50,
            max_connection_lifetime=1800.0,
            connection_liveness_check_timeout=60.0,
            connection_acquisition_timeout=5.0,
        )
        await NEO4J_DRIVER.verify_connectivity()
        STRAG_APP = build_strag_graph(NEO4J_DRIVER)
        DB_ONLINE = True
        print("✅ Connected to Neo4j TKG with warm Bolt pool.")
    except Exception as exc:
        DB_ONLINE = False
        print(f"⚠️ Live Neo4j unavailable ({exc}). Using stage-safe TKG telemetry.")
    yield
    if NEO4J_DRIVER:
        await NEO4J_DRIVER.close()

app = FastAPI(lifespan=lifespan)

class IncidentRequest(BaseModel):
    query: str
    t_query: float = 1762869600.0

@app.post("/compare")
async def compare_pipelines(req: IncidentRequest):
    if not DB_ONLINE:
        return get_fallback_payload()

    embeddings = get_embeddings()
    llm = get_llm()
    t0 = time.perf_counter()
    q_vec = await embeddings.aembed_query(req.query)
    records, _, _ = await NEO4J_DRIVER.execute_query(VECTOR_SEARCH_CYPHER, queryVec=q_vec)
    naive_top = records[0]
    naive_prompt = (
        f"Incident Query: {req.query}\n"
        f"Retrieved Policy ({naive_top['stateId']}): {naive_top['summary']}\n"
        "In 2 sentences, state whether this traffic is AUTHORIZED or a CRITICAL BREACH."
    )
    naive_resp = await llm.ainvoke(naive_prompt)
    naive_ms = round((time.perf_counter() - t0) * 1000, 1)

    t1 = time.perf_counter()
    strag_result = await STRAG_APP.ainvoke({
        "query": req.query,
        "t_query": req.t_query,
        "query_vec": [],
        "candidates": [],
        "lineage_hop_triggered": False,
        "verdict": "",
    })
    strag_ms = round((time.perf_counter() - t1) * 1000, 1)

    return {
        "naive": {
            "selected_state": naive_top["stateId"],
            "sem_score": round(naive_top["semScore"], 4),
            "summary": naive_top["summary"],
            "verdict": naive_resp.content,
            "latency_ms": naive_ms,
        },
        "strag": {
            "candidates": strag_result["candidates"],
            "lineage_hop_triggered": strag_result["lineage_hop_triggered"],
            "verdict": strag_result["verdict"],
            "latency_ms": strag_ms,
            "bolt_pool": "Warm Container Singleton (liveness_check=60s)",
        },
    }
