import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from blackwell.client import CachedChatClient, parse_probability
from blackwell.compression import Compactor
from blackwell.prompts import compression_messages
from blackwell.storage import ROOT, read_json
from run_experiment import analyze


class FakeCounter:
    def count(self, text):
        return len(text)


class FakeChat:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.requests = []

    def complete(self, messages, max_tokens):
        self.requests.append(messages)
        return next(self.outputs), str(len(self.requests))


class PipelineTests(unittest.TestCase):
    def test_probability_parser_rejects_bad_or_ambiguous_output(self):
        self.assertEqual(parse_probability('{"p_yes":0.73}'), .73)
        for bad in ['Yes', '{"p_yes":"0.7"}', '{"p_yes":true}', '{"p_yes":NaN}', '{"p_yes":1.1}', '{"p_yes":0.5,"label":1}', '[]']:
            with self.assertRaises((ValueError, TypeError)):
                parse_probability(bad)

    def test_query_blind_compression_and_length_repair(self):
        client = FakeChat(["too long", "abc"])
        compactor = Compactor(client, FakeCounter())
        result = compactor.compress("ONLY_ALLOWED_CONTEXT", 3, "summary")
        self.assertTrue(result["budget_met"])
        self.assertTrue(result["length_repair_used"])
        self.assertEqual(len(client.requests), 2)
        for messages in client.requests:
            serialized = json.dumps(messages)
            self.assertIn("ONLY_ALLOWED_CONTEXT", serialized)
            self.assertNotIn("PRIVATE_QUESTION_SENTINEL", serialized)
            self.assertNotIn("GOLD_LABEL_SENTINEL", serialized)
        with self.assertRaises(TypeError):
            compactor.compress("context", 3, "summary", question="PRIVATE_QUESTION_SENTINEL")

    def test_over_budget_summary_is_marked_not_silently_truncated(self):
        result = Compactor(FakeChat(["abcdef", "abcde"]), FakeCounter()).compress("abcdefgh", 3, "summary")
        self.assertFalse(result["budget_met"])
        self.assertEqual(result["text"], "abcde")
        self.assertEqual(result["tokens"], 5)

    def test_cache_replays_without_network_or_key(self):
        config = read_json(ROOT / "configs" / "pilot.json")
        response = {"choices":[{"finish_reason":"stop","message":{"content":'{"p_yes":0.7}'}}],"usage":{"total_tokens":10}}
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache") as directory:
            root = Path(directory)
            client = CachedChatClient(config, root)
            with patch.dict("os.environ", {"SILICONFLOW_API_KEY":"not-a-real-key"}), patch("blackwell.client.urlopen", return_value=io.BytesIO(json.dumps(response).encode())) as transport:
                original = client.complete(compression_messages("context", 10), 64, True)
                self.assertEqual(transport.call_count, 1)
            with patch.dict("os.environ", {}, clear=True), patch("blackwell.client.urlopen", side_effect=AssertionError("Network must not be called")):
                cached = CachedChatClient(config, root, offline=True).complete(compression_messages("context", 10), 64, True)
            self.assertEqual(original, cached)
            for file in root.rglob("*.json"):
                text = file.read_text(encoding="utf-8")
                self.assertNotIn("not-a-real-key", text)
                self.assertNotIn("Authorization", text)

    def test_call_limit_stops_before_network(self):
        config = {**read_json(ROOT / "configs" / "pilot.json"), "max_api_attempts":0}
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache") as directory, patch.dict("os.environ", {"SILICONFLOW_API_KEY":"not-a-real-key"}), patch("blackwell.client.urlopen") as transport:
            with self.assertRaises(RuntimeError):
                CachedChatClient(config, Path(directory)).complete([], 64)
            transport.assert_not_called()

    def test_incomplete_run_cannot_generate_a_success_report(self):
        with self.assertRaises(ValueError):
            analyze([], {"sample_size":30})

    @classmethod
    def setUpClass(cls):
        (ROOT / ".cache").mkdir(exist_ok=True)


if __name__ == "__main__":
    unittest.main()

