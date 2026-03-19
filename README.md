# llmcast

A Python package that uses LLMs to cast JSON data from one schema to another.

Just provide a target schema example and the LLM automatically generates a transform function, then applies it to the entire dataset. **The LLM is called only once regardless of dataset size.**

## Installation

```bash
pip install llmcast
```

## Usage

### Python API

```python
from openai import OpenAI
from llmcast import Llmcast

client = OpenAI()
mapper = Llmcast(client=client, model="gpt-4o-mini")

source = {"first_name": "John", "last_name": "Doe", "age": 30}
target_schema = {"fullName": "", "years": 0}

result = mapper.convert(source, target_schema)
# {"fullName": "John Doe", "years": 30}
```

List data is also supported. The first item is used as a sample to generate the transform function, which is then applied to all items.

```python
source = [
    {"first_name": "John", "last_name": "Doe"},
    {"first_name": "Jane", "last_name": "Smith"},
]
target_schema = [{"fullName": ""}]

result = mapper.convert(source, target_schema)
# [{"fullName": "John Doe"}, {"fullName": "Jane Smith"}]
```

### CLI

```bash
# File input
llmcast source.json target.json

# Direct JSON string input
llmcast '{"key":"value"}' '{"new_key":""}' --model gpt-4o

# Stdin input
cat source.json | llmcast - target.json

# Specify output file
llmcast source.json target.json -o result.json

# Recursively convert all JSON files in a directory
llmcast ./input_dir target.json -r -o ./output_dir

# Verbose logging
llmcast source.json target.json -v
```

#### CLI Options

| Option | Description |
|--------|-------------|
| `source` | Source JSON (file path, JSON string, or `-` for stdin) |
| `target` | Target schema JSON (file path or JSON string) |
| `--api-key` | OpenAI API key (can also be set via `OPENAI_API_KEY` env var) |
| `--model` | Model to use (default: `gpt-4o-mini`) |
| `-o`, `--output` | Output file path (default: `output.json`) |
| `-r`, `--recursive` | Recursively convert all JSON files in a directory |
| `-v`, `--verbose` | Enable verbose logging |

## How It Works

llmcast operates in two steps:

1. **Generate** — A single source data sample and the target schema are sent to the LLM, which generates a Python `transform(item)` function.
2. **Execute** — The generated function is applied to every item in the source data.

This means even thousands of records require only a single LLM call.

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run a single test
pytest tests/test_mapper.py::TestConvertSimple::test_flat_mapping
```

## License

MIT
