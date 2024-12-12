import unittest
from llm import convert_frontend_str_to_llm, Llm


class TestConvertFrontendStrToLlm(unittest.TestCase):
    def test_convert_valid_strings(self):
        self.assertEqual(
            convert_frontend_str_to_llm("gpt_4_vision"),
            Llm.GPT_4_VISION,
            "Should convert 'gpt_4_vision' to Llm.GPT_4_VISION",
        )
        self.assertEqual(
            convert_frontend_str_to_llm("claude_3_sonnet"),
            Llm.CLAUDE_3_SONNET,
            "Should convert 'claude_3_sonnet' to Llm.CLAUDE_3_SONNET",
        )
        self.assertEqual(
            convert_frontend_str_to_llm("claude-3-opus"),
            Llm.CLAUDE_3_OPUS,
            "Should convert 'claude-3-opus' to Llm.CLAUDE_3_OPUS",
        )
        self.assertEqual(
            convert_frontend_str_to_llm("gpt-4-turbo"),
            Llm.GPT_4_TURBO,
            "Should convert 'gpt-4-turbo' to Llm.GPT_4_TURBO",
        )
        self.assertEqual(
            convert_frontend_str_to_llm("gpt-4o"),
            Llm.GPT_4O,
            "Should convert 'gpt-4o' to Llm.GPT_4O",
        )

    def test_convert_invalid_string_raises_exception(self):
        with self.assertRaises(ValueError):
            convert_frontend_str_to_llm("invalid_string")
        with self.assertRaises(ValueError):
            convert_frontend_str_to_llm("another_invalid_string")


if __name__ == "__main__":
    unittest.main()
