import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_openai import OpenAIEmbeddings

load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

STATES = [
    {
        "state_id": "state-fw-v1-aug2026",
        "version": 1,
        "summary": (
            "Temporary maintenance rule: ALLOW TCP port 8443 connection from "
            "service account sa-billing-prod to database db-ledger-01 for Q3 migration."
        ),
        "valid_from": 1754006400,   # 2026-08-01 00:00:00 UTC
        "valid_to": 1755216000,     # 2026-08-15 00:00:00 UTC (EXPIRED)
        "tx_ingested": 1754006405,
        "change_ticket": "SEC-1042"
    },
    {
        "state_id": "state-fw-v2-nov2026",
        "version": 2,
        "summary": (
            "Zero-Trust Hardening Patch CVE-2026-4412: DENY ALL unauthenticated "
            "non-mTLS ingress on port 8443; revoke legacy service account direct access."
        ),
        "valid_from": 1762560000,   # 2026-11-08 00:00:00 UTC
        "valid_to": 9999999999,     # Active sentinel
        "tx_ingested": 1762560012,
        "change_ticket": "SEC-8891"
    }
]

SEED_CYPHER = """
MERGE (asset:SecurityAsset {id: 'sg-prod-db-01'})
SET asset.name = 'Production Ledger DB Security Group',
    asset.half_life_sec = 86400.0

WITH asset
UNWIND $states AS s
MERGE (st:ConfigState {id: s.state_id})
SET st.version = s.version,
    st.summary = s.summary,
    st.valid_from = s.valid_from,
    st.valid_to = s.valid_to,
    st.tx_ingested = s.tx_ingested,
    st.change_ticket = s.change_ticket,
    st.embedding = s.embedding
MERGE (asset)-[:HAS_STATE]->(st)

WITH asset
MATCH (v1:ConfigState {id: 'state-fw-v1-aug2026'})
MATCH (v2:ConfigState {id: 'state-fw-v2-nov2026'})
MERGE (v1)-[:SUPERSEDED_BY {reason: 'CVE-2026-4412 Emergency Patch'}]->(v2)
MERGE (v2)-[:PREVIOUS_STATE]->(v1)
"""

INDEX_CYPHER = """
CREATE VECTOR INDEX config_state_vec_idx IF NOT EXISTS
FOR (n:ConfigState) ON (n.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
  }
}
"""

def seed():
    for s in STATES:
        s["embedding"] = embeddings.embed_query(s["summary"])
    with GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
    ) as driver:
        driver.execute_query(INDEX_CYPHER)
        driver.execute_query(SEED_CYPHER, states=STATES)
        print("Seeded TKG nodes, embeddings, and causal state chain.")

if __name__ == "__main__":
    seed()
