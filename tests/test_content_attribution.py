from __future__ import annotations

from opencode_tokenstats.content_attribution import collect_content_attribution, collect_content_attribution_for_model


class FakeCounter:
    def count(self, text: str) -> int:
        return len(text)


def test_collects_category_totals_and_tool_usage_semantics() -> None:
    messages = [
        {
            "role": "user",
            "info": {"system": "SYS"},
            "parts": [
                {"type": "text", "text": "hello"},
                {"type": "reasoning", "text": "u-reason"},
                {
                    "type": "tool",
                    "tool": "read",
                    "state": {"status": "completed", "output": "file-content"},
                },
                {
                    "type": "tool",
                    "tool": "bash",
                    "state": {"status": "error", "output": "oops"},
                },
            ],
        },
        {
            "role": "assistant",
            "parts": [
                {"type": "text", "text": "answer"},
                {"type": "reasoning", "text": "a-reason"},
                {
                    "type": "tool",
                    "tool": "read",
                    "state": {"status": "completed", "output": {"k": "v"}},
                },
                {
                    "type": "tool",
                    "tool": "read",
                    "state": {"status": "running", "output": "not-done"},
                },
            ],
        },
    ]

    result = collect_content_attribution(messages, token_counter=FakeCounter())

    assert result.totals.system_tokens == 3
    assert result.totals.user_tokens == 5
    assert result.totals.assistant_tokens == 6
    assert result.totals.reasoning_tokens == len("u-reason") + len("a-reason")

    read_stat = next(s for s in result.tool_usage if s.tool_name == "read")
    bash_stat = next(s for s in result.tool_usage if s.tool_name == "bash")

    # call_count includes all tool calls regardless of completion status
    assert read_stat.call_count == 3
    assert bash_stat.call_count == 1

    # output_tokens include only completed outputs
    expected_read_tokens = len("file-content") + len("k: v")
    assert read_stat.output_tokens == expected_read_tokens
    assert bash_stat.output_tokens == 0
    assert result.totals.tool_output_tokens == expected_read_tokens

    # explicit separation from schema/context estimate
    assert result.observed_tools_only is True
    assert result.tool_schema_context_estimate_tokens == 0


def test_ignores_non_assistant_non_user_text_but_counts_system_if_present() -> None:
    messages = [
        {
            "role": "system",
            "info": {"system": "PROMPT"},
            "parts": [{"type": "text", "text": "hidden"}],
        }
    ]

    result = collect_content_attribution(messages, token_counter=FakeCounter())

    assert result.totals.system_tokens == len("PROMPT")
    assert result.totals.user_tokens == 0
    assert result.totals.assistant_tokens == 0


def test_model_based_attribution_exposes_approximate_warning() -> None:
    messages = [
        {
            "role": "assistant",
            "parts": [{"type": "text", "text": "abcd"}],
        }
    ]

    result = collect_content_attribution_for_model(
        messages,
        provider_id="unknown-provider",
        model_id="unknown-model",
    )

    assert result.totals.assistant_tokens == 1
    assert result.approximate_tokenizer_used is True
    assert any("approximate" in w for w in result.warnings)
