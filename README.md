# s-TRAG: Serverless Cross-Temporal Reasoning

This demo combines immutable temporal configuration states in Neo4j, vector
retrieval, a Temporal Alignment Score (TAS), and a LangGraph staleness gate.
The comparator shows how semantic-only retrieval can surface an expired
firewall rule while s-TRAG follows `SUPERSEDED_BY` to the state valid at the
incident time.

## Requirements

- Python 3.11+
- Neo4j 5.26+ with vector-index support
- OpenAI API key (embeddings and response synthesis)

## Local quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, and `OPENAI_API_KEY` in
`.env`, then seed the graph and start the services in separate terminals:

```bash
python seed_tkg.py
uvicorn app:app --host 0.0.0.0 --port 8000
streamlit run ui.py
```

Open `http://localhost:8501`. The API listens on port 8000; set `STRAG_API_URL`
to change the backend URL.

## Temporal Alignment Score

For a state valid over `[t_start, t_end]`, query timestamp `t_q`, and asset
half-life `t_1/2`, the implementation uses `lambda = ln(2) / t_1/2`:

```text
TAS = exp(-0.05 * lambda * |t_q - t_start|)                   when active
TAS = omega * exp(-lambda * distance_to_validity_window)      otherwise
```

The default historical penalty is `omega = 0.15`. A top semantic candidate
with TAS below `0.35` enters the lineage walker, which follows up to three
`SUPERSEDED_BY` relationships to a state valid at the query time.

## API

`POST /compare` accepts a JSON body such as:

```json
{
  "query": "Is TCP port 8443 traffic from sa-billing-prod authorized?",
  "query_time": "2026-11-11T14:00:00Z"
}
```

It returns the top semantic-only match and the grounded s-TRAG answer, selected
state, TAS, lineage status, and change-ticket citation. `GET /health` reports
whether the backend started with its required credentials.

## Cloud Run

Deploy with request concurrency enabled and provide credentials through
Cloud Run's environment/secret configuration:

```bash
gcloud run deploy strag-api --source . --region us-central1 \
  --concurrency 80 --cpu-boost --min-instances 0 --max-instances 10
```

The FastAPI lifespan owns a shared async Neo4j driver configured for a
50-connection pool, 30-minute connection lifetime, 60-second liveness checks,
and 5-second acquisition timeout.

## Tests

```bash
python -m unittest discover -s tests
```
