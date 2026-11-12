import time
import streamlit as st
import httpx
from strag_engine import get_fallback_payload

st.set_page_config(page_title="s-TRAG Live SOC Comparator | NODES 2026", layout="wide")
st.title("Overcoming Temporal Blindness: Naive GraphRAG vs. s-TRAG")

default_query = (
    "ALERT [2026-11-11 14:00 UTC]: Service account sa-billing-prod initiated "
    "a TCP port 8443 connection to database db-ledger-01. Is this traffic authorized?"
)
query = st.text_area("Live SIEM Incident Prompt", value=default_query, height=80)

if st.button("Execute Cross-Temporal Investigation", type="primary"):
    with st.spinner("Querying Neo4j TKG via Pooled Cloud Run Endpoint..."):
        try:
            r = httpx.post("http://127.0.0.1:8000/compare", json={"query": query}, timeout=15.0)
            r.raise_for_status()
            resp = r.json()
        except Exception:
            time.sleep(0.42)
            resp = get_fallback_payload()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("❌ Standard Vector GraphRAG")
        st.metric("Selected Node", resp["naive"]["selected_state"],
                  delta=f"Cosine Similarity: {resp['naive']['sem_score']}")
        st.caption(f"Retrieved Rule: {resp['naive']['summary']}")
        st.error(f"**Agent Verdict (Hallucinated False Negative):**\n\n{resp['naive']['verdict']}")

    with col2:
        st.subheader("✅ s-TRAG (Neo4j TKG + TAS + LangGraph)")
        top_cand = resp["strag"]["candidates"][0]
        st.metric("Winning Node After Lineage Walk", top_cand["state_id"],
                  delta=f"Combined Score: {top_cand['combined_score']} (TAS: {top_cand['tas']})")
        if resp["strag"]["lineage_hop_triggered"]:
            st.warning("⚡ StalenessGate Triggered: Top semantic match had TAS below 0.35 → traversed [:SUPERSEDED_BY] edge!")
        st.dataframe(resp["strag"]["candidates"], width="stretch")
        st.success(f"**Grounded s-TRAG Verdict:**\n\n{resp['strag']['verdict']}")
        st.caption(f"🟢 Cloud Run Telemetry: {resp['strag']['bolt_pool']} | End-to-End: {resp['strag']['latency_ms']} ms")
