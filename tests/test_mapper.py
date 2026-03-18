from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

from llmcast import Llmcast


# ---------------------------------------------------------------------------
# Helpers
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


def _make_client(transform_code: str) -> MagicMock:
    """Return a mock OpenAI client that returns *transform_code* as the response."""
    completion = _ChatCompletion(choices=[_Choice(message=_Message(content=transform_code))])
    client = MagicMock()
    client.chat.completions.create.return_value = completion
    return client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestConvertSimple:
    def test_flat_mapping(self):
        code = (
            "def transform(item):\n"
            "    return {\n"
            '        "fullName": item.get("first_name", "") + " " + item.get("last_name", ""),\n'
            '        "years": item.get("age", 0),\n'
            "    }\n"
        )
        client = _make_client(code)
        mapper = Llmcast(client=client, verbose=True)

        source = {"first_name": "John", "last_name": "Doe", "age": 30}
        result = mapper.convert(source, {"fullName": "", "years": 0})

        assert result == {"fullName": "John Doe", "years": 30}
        client.chat.completions.create.assert_called_once()

    def test_custom_model(self):
        code = "def transform(item):\n    return {'out': item.get('in', 0)}\n"
        client = _make_client(code)
        mapper = Llmcast(client=client, model="gpt-4o", verbose=True)

        mapper.convert({"in": 1}, {"out": 0})

        call_kwargs = client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4o"


class TestConvertNested:
    def test_nested_objects(self):
        code = (
            "def transform(item):\n"
            "    return {\n"
            '        "user": {\n'
            '            "name": item.get("userName", ""),\n'
            '            "contact": {"email": item.get("userEmail", "")},\n'
            "        }\n"
            "    }\n"
        )
        client = _make_client(code)
        mapper = Llmcast(client=client, verbose=True)

        source = {"userName": "Alice", "userEmail": "alice@example.com"}
        result = mapper.convert(source, {"user": {"name": "", "contact": {"email": ""}}})

        assert result == {
            "user": {"name": "Alice", "contact": {"email": "alice@example.com"}},
        }


class TestConvertList:
    def test_list_source_applies_to_each_item(self):
        """When source is a list, the transform function is applied to each item."""
        code = (
            "def transform(item):\n"
            '    return {"name": item.get("title", "")}\n'
        )
        client = _make_client(code)
        mapper = Llmcast(client=client, verbose=True)

        source = [{"title": "A"}, {"title": "B"}, {"title": "C"}]
        result = mapper.convert(source, [{"name": ""}])

        assert result == [{"name": "A"}, {"name": "B"}, {"name": "C"}]
        # LLM은 한 번만 호출되어야 함 (샘플로 함수 생성)
        client.chat.completions.create.assert_called_once()

    def test_list_target_uses_first_item_as_schema(self):
        """When target_example is a list, the first item is used as the schema."""
        code = (
            "def transform(item):\n"
            '    return {"name": item.get("title", ""), "active": True}\n'
        )
        client = _make_client(code)
        mapper = Llmcast(client=client, verbose=True)

        source = [{"title": "X"}]
        result = mapper.convert(source, [{"name": "", "active": True}])

        assert result == [{"name": "X", "active": True}]


class TestCodeExtraction:
    def test_strips_markdown_fences(self):
        code = (
            "```python\n"
            "def transform(item):\n"
            "    return {'a': item.get('x', 0)}\n"
            "```"
        )
        client = _make_client(code)
        mapper = Llmcast(client=client, verbose=True)

        result = mapper.convert({"x": 1}, {"a": 0})
        assert result == {"a": 1}


class TestEdgeCases:
    def test_empty_source(self):
        code = (
            "def transform(item):\n"
            '    return {"fullName": item.get("first_name", ""), "years": item.get("age", 0)}\n'
        )
        client = _make_client(code)
        mapper = Llmcast(client=client, verbose=True)

        result = mapper.convert({}, {"fullName": "", "years": 0})
        assert result == {"fullName": "", "years": 0}

    def test_temperature_is_zero(self):
        code = "def transform(item):\n    return {'a': 1}\n"
        client = _make_client(code)
        mapper = Llmcast(client=client, verbose=True)
        mapper.convert({"x": 1}, {"a": 0})

        call_kwargs = client.chat.completions.create.call_args
        assert call_kwargs.kwargs["temperature"] == 0
