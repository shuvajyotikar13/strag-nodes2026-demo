"""Streamlit side-by-side comparator for naive GraphRAG and s-TRAG."""

import json
import os
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st

st.set_page_config(page_title="s-TRAG Comparator", layout="wide")
st.title("s-TRAG: Cross-Temporal Reasoning")
st.caption("Compare semantic-only retrieval with temporally aligned graph reasoning.")

query = st.text_area(
    "SIEM incident",
    value=(
        "ALERT [2026-11-11 14:00 UTC]: Service account sa-billing-prod "
        "initiated a TCP port 8443 connection to database db-ledger-01. "
        "Is this traffic authorized?"
    ),
)
query_time = st.text_input(
    "Query time (ISO 8601)",
    value=datetime(2026, 11, 11, 14, tzinfo=timezone.utc).isoformat(),
)

if st.button("Compare", type="primary"):
    try:
        timestamp = datetime.fromisoformat(query_time.replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        payload = json.dumps(
            {"query": query, "query_time": timestamp.isoformat()}
        ).encode()
        request = Request(
            f"{os.getenv('STRAG_API_URL', 'http://localhost:8000').rstrip('/')}/compare",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=60) as response:
            result = json.loads(response.read())
    except (ValueError, HTTPError, URLError, TimeoutError) as error:
        st.error(f"Could not query the s-TRAG API: {error}")
    else:
        naive, strag = st.columns(2)
        with naive:
            st.subheader("Naive vector retrieval")
            if result["naive"]:
                st.code(result["naive"]["state_id"])
                st.write(result["naive"]["description"])
                st.metric("Cosine similarity", f"{result['naive']['semantic_score']:.3f}")
                st.caption(
                    f"Validity: {result['naive']['valid_from']} – "
                    f"{result['naive']['valid_to']}"
                )
            else:
                st.info("No matching state found.")
        with strag:
            st.subheader("s-TRAG")
            st.write(result["strag"]["answer"])
            st.metric(
                "Temporal alignment score",
                f"{result['strag']['temporal_score']:.3f}"
                if result["strag"]["temporal_score"] is not None
                else "N/A",
            )
            st.write(f"State: `{result['strag']['state_id'] or 'none'}`")
            st.write(f"Lineage walk: **{result['strag']['lineage_walked']}**")
            if result["strag"]["citation"]:
                st.caption(f"Change ticket: {result['strag']['citation']}")
