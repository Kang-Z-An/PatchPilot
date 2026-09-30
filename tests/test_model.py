from __future__ import annotations

import io
import json
import unittest
from unittest.mock import patch

from patchpilot.model import OpenAICompatibleModel


class OpenAICompatibleModelTest(unittest.TestCase):
    def test_sampling_temperature_is_sent_without_changing_messages(self) -> None:
        captured = {}

        def respond(request, *, timeout):
            captured.update(json.loads(request.data))
            return io.BytesIO(b'{"choices":[{"message":{"content":"{}"}}]}')

        model = OpenAICompatibleModel(
            "example-model",
            "test-key",
            "https://example.invalid/v1",
            temperature=0.7,
            thinking_mode="disabled",
        )
        messages = [{"role": "user", "content": "issue"}]
        with patch("patchpilot.model.urllib.request.urlopen", side_effect=respond):
            self.assertEqual(model.complete(messages), "{}")

        self.assertEqual(captured["temperature"], 0.7)
        self.assertEqual(captured["thinking"], {"type": "disabled"})
        self.assertEqual(captured["messages"], messages)
        self.assertEqual(captured["response_format"], {"type": "json_object"})

    def test_default_remains_deterministic_and_invalid_temperature_rejected(self) -> None:
        model = OpenAICompatibleModel("m", "key", "https://example.invalid/v1")
        self.assertEqual(model.temperature, 0.0)
        self.assertIsNone(model.thinking_mode)
        with self.assertRaisesRegex(ValueError, "temperature"):
            OpenAICompatibleModel("m", "key", "https://example.invalid/v1", temperature=2.1)
        with self.assertRaisesRegex(ValueError, "thinking_mode"):
            OpenAICompatibleModel("m", "key", "https://example.invalid/v1", thinking_mode="other")


if __name__ == "__main__":
    unittest.main()
