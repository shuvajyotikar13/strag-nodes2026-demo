import os
import math
from typing import TypedDict, List, Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, END

load_dotenv()

def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small")

def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)

def compute_tas(valid_from: float, valid_to: float, t_query: float,
                half_life_sec: float = 86400.0, omega: float = 0.15) -> float:
    lam = math.log(2) / half_life_sec
    is_active = (valid_from <= t_query) and (valid_to >= t_query)
    if is_active:
        return math.exp(-0.05 * lam * max(0.0, t_query - valid_from))
    dist = min(abs(t_query - valid_from), abs(t_query - valid_to))
    return omega * math.exp(-lam * dist)

class GraphRAGState(TypedDict):
    query: str
    t_query: float
    query_vec: List[float]
    candidates: List[Dict[str, Any]]
    lineage_hop_triggered: bool
    verdict: str

VECTOR_SEARCH_CYPHER = """
CALL db.index.vector.queryNodes('config_state_vec_idx', 5, $queryVec)
YIELD node AS st, score AS semScore
MATCH (asset:SecurityAsset)-[:HAS_STATE]->(st)
RETURN asset.id AS assetId,
       asset.half_life_sec AS halfLife,
       st.id AS stateId,
       st.summary AS summary,
       st.valid_from AS validFrom,
       st.valid_to AS validTo,
       st.change_ticket AS ticket,
       semScore
"""

LINEAGE_WALK_CYPHER = """
MATCH (stale:ConfigState {id: $staleStateId})-[r:SUPERSEDED_BY*1..3]->(newer:ConfigState)
WHERE newer.valid_from <= $tQuery AND newer.valid_to >= $tQuery
MATCH (asset:SecurityAsset)-[:HAS_STATE]->(newer)
RETURN asset.id AS assetId,
       asset.half_life_sec AS halfLife,
       newer.id AS stateId,
       newer.summary AS summary,
       newer.valid_from AS validFrom,
       newer.valid_to AS validTo,
       newer.change_ticket AS ticket,
       0.80 AS semScore
"""

def build_strag_graph(driver):
    embeddings = get_embeddings()
    llm = get_llm()

    async def retrieve_candidates(state: GraphRAGState) -> GraphRAGState:
        q_vec = await embeddings.aembed_query(state["query"])
        records, _, _ = await driver.execute_query(VECTOR_SEARCH_CYPHER, queryVec=q_vec)
        scored = []
        for r in records:
            tas = compute_tas(r["validFrom"], r["validTo"], state["t_query"], r["halfLife"])
            combined = 0.45 * r["semScore"] + 0.55 * tas
            scored.append({
                "state_id": r["stateId"],
                "summary": r["summary"],
                "ticket": r["ticket"],
                "sem_score": round(r["semScore"], 4),
                "tas": round(tas, 4),
                "combined_score": round(combined, 4),
                "is_stale": r["validTo"] < state["t_query"]
            })
        scored.sort(key=lambda x: x["sem_score"], reverse=True)
        return {**state, "query_vec": q_vec, "candidates": scored, "lineage_hop_triggered": False}

    def staleness_gate(state: GraphRAGState) -> str:
        top_semantic = state["candidates"][0]
        if top_semantic["tas"] < 0.35 or top_semantic["is_stale"]:
            return "walk_lineage"
        return "synthesize"

    async def walk_lineage(state: GraphRAGState) -> GraphRAGState:
        stale_id = state["candidates"][0]["state_id"]
        records, _, _ = await driver.execute_query(
            LINEAGE_WALK_CYPHER, staleStateId=stale_id, tQuery=state["t_query"]
        )
        updated = list(state["candidates"])
        for r in records:
            tas = compute_tas(r["validFrom"], r["validTo"], state["t_query"], r["halfLife"])
            combined = 0.45 * r["semScore"] + 0.55 * tas
            updated.append({
                "state_id": f"{r['stateId']} (via SUPERSEDED_BY)",
                "summary": r["summary"],
                "ticket": r["ticket"],
                "sem_score": round(r["semScore"], 4),
                "tas": round(tas, 4),
                "combined_score": round(combined, 4),
                "is_stale": False
            })
        updated.sort(key=lambda x: x["combined_score"], reverse=True)
        return {**state, "candidates": updated, "lineage_hop_triggered": True}

    async def synthesize(state: GraphRAGState) -> GraphRAGState:
        winning_state = state["candidates"][0]
        prompt = (
            f"You are a SOC Incident Response Agent evaluating an event at UNIX timestamp {state['t_query']}.\n"
            f"Incident Query: {state['query']}\n"
            f"Governing Policy State: {winning_state['state_id']} (Ticket {winning_state['ticket']}): {winning_state['summary']}\n"
            f"Lineage Recovery Triggered: {state['lineage_hop_triggered']}\n"
            "Provide a 2-sentence security verdict stating whether the traffic is AUTHORIZED or a CRITICAL BREACH, citing the state ID."
        )
        resp = await llm.ainvoke(prompt)
        return {**state, "verdict": resp.content}

    wf = StateGraph(GraphRAGState)
    wf.add_node("retrieve", retrieve_candidates)
    wf.add_node("walk_lineage", walk_lineage)
    wf.add_node("synthesize", synthesize)
    wf.set_entry_point("retrieve")
    wf.add_conditional_edges("retrieve", staleness_gate, {
        "walk_lineage": "walk_lineage",
        "synthesize": "synthesize"
    })
    wf.add_edge("walk_lineage", "synthesize")
    wf.add_edge("synthesize", END)
    return wf.compile()

def get_fallback_payload() -> dict:
    return {
        "naive": {
            "selected_state": "state-fw-v1-aug2026",
            "sem_score": 0.9312,
            "summary": "Temporary maintenance rule: ALLOW TCP port 8443 connection from service account sa-billing-prod to database db-ledger-01 for Q3 migration.",
            "verdict": "AUTHORIZED: Traffic matches temporary maintenance rule SEC-1042 permitting sa-billing-prod to connect to db-ledger-01 on TCP port 8443.",
            "latency_ms": 384.2,
        },
        "strag": {
            "candidates": [
                {
                    "state_id": "state-fw-v2-nov2026 (via SUPERSEDED_BY)",
                    "summary": "Zero-Trust Hardening Patch CVE-2026-4412: DENY ALL unauthenticated non-mTLS ingress on port 8443; revoke legacy service account direct access.",
                    "ticket": "SEC-8891",
                    "sem_score": 0.8000,
                    "tas": 0.9914,
                    "combined_score": 0.9053,
                    "is_stale": False,
                },
                {
                    "state_id": "state-fw-v1-aug2026",
                    "summary": "Temporary maintenance rule: ALLOW TCP port 8443 connection from service account sa-billing-prod to database db-ledger-01 for Q3 migration.",
                    "ticket": "SEC-1042",
                    "sem_score": 0.9312,
                    "tas": 0.0001,
                    "combined_score": 0.4191,
                    "is_stale": True,
                },
            ],
            "lineage_hop_triggered": True,
            "verdict": "CRITICAL BREACH: Legacy rule state-fw-v1-aug2026 (Ticket SEC-1042) expired on Aug 15, 2026 and was superseded on Nov 8, 2026 by Zero-Trust patch state-fw-v2-nov2026 (Ticket SEC-8891), which denies all non-mTLS ingress on port 8443.",
            "latency_ms": 412.6,
            "bolt_pool": "Warm Container Singleton (liveness_check=60s)",
        },
    }
