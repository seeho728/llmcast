# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (editable with dev deps)
.venv/bin/pip install -e ".[dev]"

# Run all tests
.venv/bin/pytest tests/

# Run a single test
.venv/bin/pytest tests/test_mapper.py::TestConvertSimple::test_flat_mapping

# CLI usage
.venv/bin/llmcast source.json target.json -v
.venv/bin/llmcast '{"key":"value"}' '{"new_key":""}' --model gpt-4o
cat source.json | .venv/bin/llmcast - target.json
```

## Architecture

**2-step LLM mapping approach:**

1. **Generate** — `_generate_mapping_function()` sends a single source sample + target schema to the LLM, which returns a Python `transform(item)` function as code.
2. **Execute** — `convert()` runs that function via `exec()` on every item in the source data.

This means the LLM is called **once** regardless of dataset size. For list inputs, only the first item is sent as a sample.

**Key files:**
- `llmcast/mapper.py` — `Llmcast` class with `convert()` (public API) and `_generate_mapping_function()` (LLM call)
- `llmcast/cli.py` — CLI entry point, handles JSON string/file/stdin input and SSL workaround (`verify=False`)
- `tests/test_mapper.py` — Mock-based tests using fake OpenAI client that returns pre-written transform code

**Design decisions:**
- `response_format` is NOT used (LLM returns Python code, not JSON)
- `temperature=0` for deterministic function generation
- `_extract_function()` strips markdown fences from LLM output
- List targets use first item as schema; list sources apply transform per-item
