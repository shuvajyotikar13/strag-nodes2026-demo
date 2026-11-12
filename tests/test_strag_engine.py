import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from strag_engine import build_reasoning_graph, needs_lineage_walk, temporal_alignment_score


class TemporalAlignmentTests(unittest.TestCase):
    def test_active_state_decays_slowly_from_start(self):
        score = temporal_alignment_score(1000, 2000, 1100, 86400)
        self.assertAlmostEqual(score, 0.9996, places=3)

    def test_expired_state_receives_historical_penalty(self):
        score = temporal_alignment_score(1000, 2000, 2000 + 86400, 86400)
        self.assertAlmostEqual(score, 0.075)

    def test_outside_state_window_uses_nearest_boundary(self):
        just_before = temporal_alignment_score(1000, 2000, 999, 86400)
        self.assertGreater(just_before, 0.14)

    def test_invalid_half_life_is_rejected(self):
        with self.assertRaises(ValueError):
            temporal_alignment_score(1000, 2000, 1500, 0)

    def test_historical_penalty_must_be_between_zero_and_one(self):
        for omega in (0, 1):
            with self.subTest(omega=omega), self.assertRaises(ValueError):
                temporal_alignment_score(1000, 2000, 3000, 86400, omega=omega)

    def test_staleness_gate_threshold(self):
        self.assertTrue(needs_lineage_walk({"tas": 0.34}))
        self.assertFalse(needs_lineage_walk({"tas": 0.35}))
        self.assertFalse(needs_lineage_walk(None))


class ReasoningGraphTests(unittest.TestCase):
    def test_stale_semantic_match_walks_to_active_replacement(self):
        old_state = {
            "id": "old",
            "valid_from": 1000,
            "valid_to": 2000,
            "description": "Expired allow rule",
            "change_ticket": "OLD-1",
        }
        new_state = {
            "id": "new",
            "valid_from": 9000,
            "valid_to": 20000,
            "description": "Active deny rule",
            "change_ticket": "NEW-2",
        }

        class Embeddings:
            def __init__(self, **_):
                pass

            async def aembed_query(self, _query):
                return [0.0]

        class Chat:
            def __init__(self, **_):
                pass

            async def ainvoke(self, _messages):
                return SimpleNamespace(content="Traffic is denied.")

        class Result:
            def __init__(self, records=None, record=None):
                self.records = records or []
                self.record = record

            async def data(self):
                return self.records

            async def single(self):
                return self.record

        class Session:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return None

            async def run(self, query, **_):
                if "queryNodes" in query:
                    return Result(
                        records=[
                            {
                                "state": old_state,
                                "half_life_sec": 86400,
                                "semantic_score": 0.95,
                            },
                            {
                                "state": new_state,
                                "half_life_sec": 86400,
                                "semantic_score": 0.70,
                            },
                        ]
                    )
                return Result(record={"state": new_state})

        class Driver:
            def session(self):
                return Session()

        with patch("langchain_openai.OpenAIEmbeddings", Embeddings), patch(
            "langchain_openai.ChatOpenAI", Chat
        ):
            graph = build_reasoning_graph(Driver(), "test-key-not-used")
            result = asyncio.run(
                graph.ainvoke({"query": "Is the connection allowed?", "query_time": 10000})
            )

        self.assertEqual(result["selected"]["id"], "new")
        self.assertTrue(result["lineage_walked"])
        self.assertEqual(result["citation"], "NEW-2")


if __name__ == "__main__":
    unittest.main()
