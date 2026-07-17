from __future__ import annotations

import unittest

from llm_load_tester.modalities import TextHandler


class TextHandlerTests(unittest.IsolatedAsyncioTestCase):
    async def test_modelarts_reasoning_controls_use_total_completion_limit(self) -> None:
        result = await TextHandler().prepare_payload({
            "endpoint": "https://api-ap-southeast-1.modelarts-maas.com",
            "model": "glm-5.2",
            "input_tokens": 16,
            "output_tokens": 64,
            "output_token_parameter": "max_completion_tokens",
            "thinking_mode": "disabled",
        })

        self.assertEqual(result.payload["max_completion_tokens"], 64)
        self.assertNotIn("max_tokens", result.payload)
        self.assertEqual(result.payload["thinking"], {"type": "disabled"})
        self.assertEqual(result.payload["stream_options"], {"include_usage": True})
        self.assertEqual(result.metadata["output_token_parameter"], "max_completion_tokens")

    async def test_default_payload_remains_max_tokens_compatible(self) -> None:
        result = await TextHandler().prepare_payload({
            "endpoint": "http://localhost:8000",
            "model": "mock-model",
            "input_tokens": 16,
            "output_tokens": 32,
        })

        self.assertEqual(result.payload["max_tokens"], 32)
        self.assertNotIn("max_completion_tokens", result.payload)
        self.assertNotIn("thinking", result.payload)
        self.assertNotIn("stream_options", result.payload)


if __name__ == "__main__":
    unittest.main()
