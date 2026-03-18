from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

from llmapping import Llmapping


# ---------------------------------------------------------------------------
# Helpers – lightweight fakes that mimic the OpenAI response shape
# ---------------------------------------------------------------------------

@dataclass
class _Choice:
    message: Any = None


@dataclass
class _Message:
    content: str = ""


@dataclass
class _ChatCompletion:
    choices: list = field(default_factory=list)


def _make_client(response_dict: dict | list) -> MagicMock:
    """Return a mock OpenAI client that returns *response_dict* as JSON."""
    content = json.dumps(response_dict, ensure_ascii=False)
    completion = _ChatCompletion(choices=[_Choice(message=_Message(content=content))])

    client = MagicMock()
    client.chat.completions.create.return_value = completion
    return client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestConvertSimple:
    def test_flat_mapping(self):
        expected = {"fullName": "John Doe", "years": 30}
        client = _make_client(expected)
        mapper = Llmapping(client=client)

        source = {"first_name": "John", "last_name": "Doe", "age": 30}
        target = {"fullName": "", "years": 0}

        result = mapper.convert(source, target)

        assert result == expected
        client.chat.completions.create.assert_called_once()

    def test_custom_model(self):
        client = _make_client({"out": 1})
        mapper = Llmapping(client=client, model="gpt-4o")

        mapper.convert({"in": 1}, {"out": 0})

        call_kwargs = client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4o"


class TestConvertNested:
    def test_nested_objects(self):
        expected = {
            "user": {"name": "Alice", "contact": {"email": "alice@example.com"}},
        }
        client = _make_client(expected)
        mapper = Llmapping(client=client)

        source = {"userName": "Alice", "userEmail": "alice@example.com"}
        target = {"user": {"name": "", "contact": {"email": ""}}}

        result = mapper.convert(source, target)
        assert result == expected


class TestConvertList:
    def test_list_target_unwraps_envelope(self):
        """When the target example is a list but the LLM wraps it in a dict,
        the mapper should unwrap it automatically."""
        wrapped = {"items": [{"name": "A"}, {"name": "B"}]}
        client = _make_client(wrapped)
        mapper = Llmapping(client=client)

        source = [{"title": "A"}, {"title": "B"}]
        target = [{"name": ""}]

        result = mapper.convert(source, target)
        assert result == [{"name": "A"}, {"name": "B"}]


class TestEdgeCases:
    def test_empty_source(self):
        expected = {"fullName": "", "years": 0}
        client = _make_client(expected)
        mapper = Llmapping(client=client)

        result = mapper.convert({}, {"fullName": "", "years": 0})
        assert result == expected

    def test_response_format_is_json_object(self):
        client = _make_client({"a": 1})
        mapper = Llmapping(client=client)
        mapper.convert({"x": 1}, {"a": 0})

        call_kwargs = client.chat.completions.create.call_args
        assert call_kwargs.kwargs["response_format"] == {"type": "json_object"}
