"""Create the demo bitemporal graph and its vector index in Neo4j."""

import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from neo4j import GraphDatabase

load_dotenv()


def main() -> None:
    required = ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "OPENAI_API_KEY")
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise SystemExit(f"Missing required environment variables: {', '.join(missing)}")

    epoch = lambda value: int(
        datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()
    )
    states = [
        {
            "id": "state-fw-v1-aug2026",
            "valid_from": epoch("2026-08-01"),
            "valid_to": epoch("2026-08-15"),
            "description": (
                "Temporary Q3 migration firewall rule ALLOW TCP port 8443 from "
                "service account sa-billing-prod to database db-ledger-01."
            ),
            "change_ticket": "CHG-8214",
        },
        {
            "id": "state-fw-v2-nov2026",
            "valid_from": epoch("2026-11-08"),
            "valid_to": 9999999999,
            "description": (
                "CVE-2026-4412 Zero-Trust Hardening Patch SEC-8891 denies TCP "
                "port 8443 from sa-billing-prod to db-ledger-01; access is not authorized."
            ),
            "change_ticket": "SEC-8891",
        },
    ]
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.environ["OPENAI_API_KEY"],
    )
    vectors = embeddings.embed_documents([item["description"] for item in states])
    for item, vector in zip(states, vectors, strict=True):
        item["embedding"] = vector

    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    try:
        with driver.session() as session:
            session.run(
                """
                MERGE (asset:SecurityAsset {id: 'sg-prod-db-01'})
                SET asset.name = 'Production ledger firewall',
                    asset.half_life_sec = 86400
                WITH asset
                UNWIND $states AS state_data
                MERGE (state:ConfigState {id: state_data.id})
                SET state.valid_from = state_data.valid_from,
                    state.valid_to = state_data.valid_to,
                    state.description = state_data.description,
                    state.change_ticket = state_data.change_ticket,
                    state.embedding = state_data.embedding
                MERGE (asset)-[:HAS_STATE]->(state)
                """,
                states=states,
            ).consume()
            session.run(
                """
                MATCH (old:ConfigState {id: 'state-fw-v1-aug2026'})
                MATCH (new:ConfigState {id: 'state-fw-v2-nov2026'})
                MERGE (old)-[:SUPERSEDED_BY {reason: 'CVE-2026-4412 Patch'}]->(new)
                MERGE (new)-[:PREVIOUS_STATE]->(old)
                """
            ).consume()
            session.run(
                """
                CREATE VECTOR INDEX configstate_embedding IF NOT EXISTS
                FOR (state:ConfigState) ON (state.embedding)
                OPTIONS {indexConfig: {
                    `vector.dimensions`: 1536,
                    `vector.similarity_function`: 'cosine'
                }}
                """
            ).consume()
            session.run(
                "CALL db.awaitIndex('configstate_embedding', 300)"
            ).consume()
    finally:
        driver.close()


if __name__ == "__main__":
    main()
