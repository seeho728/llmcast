from __future__ import annotations

import json
import logging
from typing import Any, Union

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """\
You are a data schema mapping expert.
Given a source data example and a target schema example, generate a Python function that converts any data with the source structure into the target structure.

Rules:
1. Define exactly one function: `def transform(item: dict) -> dict:`
2. The function receives a single source item (dict) and returns a single converted item (dict).
3. Map fields by understanding the semantic meaning of field names.
4. Combine or split source fields when necessary (e.g. first_name + last_name -> fullName).
5. If a target field cannot be filled from the source, use the default value from the target example.
6. Handle missing keys gracefully with `.get()`.
7. Return ONLY the Python function code, no markdown fences, no imports, no explanation."""


_SAMPLE_SIZE = 3


def _build_user_prompt(source_sample: Any, target_example: Any) -> str:
    return (
        f"Source data example:\n{json.dumps(source_sample, ensure_ascii=False, indent=2)}\n\n"
        f"Target schema example:\n{json.dumps(target_example, ensure_ascii=False, indent=2)}\n\n"
        "Generate the transform function.\n"
        "The function must accept a single item (dict), NOT a list."
    )


def _extract_function(code: str) -> str:
    """Strip markdown fences if the LLM wraps the code."""
    lines = code.strip().splitlines()
    # Remove leading/trailing ```python / ```
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines)


class Llmapping:
    """Use an LLM to convert data from one schema to another.

    Step 1: Ask the LLM how to map source schema -> target schema (generates a Python function).
    Step 2: Execute that function to perform the actual conversion.
    """

    def __init__(
        self,
        client: Any,
        model: str = "gpt-4o-mini",
        verbose: bool = False,
    ) -> None:
        self.client = client
        self.model = model
        self.verbose = verbose

        if self.verbose:
            logging.basicConfig(
                level=logging.DEBUG,
                format="[%(name)s] %(message)s",
            )
            logger.setLevel(logging.DEBUG)

    def _generate_mapping_function(
        self,
        source_sample: Union[dict, list],
        target_example: Union[dict, list],
    ) -> str:
        """Ask the LLM to generate a transform function using a sample."""
        user_prompt = _build_user_prompt(source_sample, target_example)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        if self.verbose:
            logger.debug("--- Step 1: Generate mapping function ---")
            logger.debug("Model: %s", self.model)
            logger.debug("Source sample:\n%s", json.dumps(source_sample, ensure_ascii=False, indent=2))
            logger.debug("Target schema:\n%s", json.dumps(target_example, ensure_ascii=False, indent=2))

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
        )

        raw_code = response.choices[0].message.content
        code = _extract_function(raw_code)

        if self.verbose:
            logger.debug("Generated code:\n%s", code)

        return code

    def convert(
        self,
        source: Union[dict, list],
        target_example: Union[dict, list],
    ) -> Union[dict, list]:
        """Map *source* data into the structure described by *target_example*.

        1. Samples a few items from source to ask the LLM for a mapping function.
        2. Executes that function on every item in the source data.
        """
        # target_example가 리스트면 첫 번째 아이템을 스키마로 사용
        target_schema = target_example[0] if isinstance(target_example, list) else target_example

        # source에서 샘플 추출 (LLM에는 샘플만 전달)
        if isinstance(source, list):
            source_sample = source[0] if source else {}
        else:
            source_sample = source

        code = self._generate_mapping_function(source_sample, target_schema)

        if self.verbose:
            logger.debug("--- Step 2: Execute mapping function ---")

        namespace: dict[str, Any] = {}
        exec(code, namespace)  # noqa: S102
        transform = namespace["transform"]

        # 리스트면 각 아이템에 적용, 단일이면 바로 적용
        if isinstance(source, list):
            result = [transform(item) for item in source]
        else:
            result = transform(source)

        if self.verbose:
            sample = result[:3] if isinstance(result, list) else result
            logger.debug("Result (sample):\n%s", json.dumps(sample, ensure_ascii=False, indent=2))
            if isinstance(result, list):
                logger.debug("Total items: %d", len(result))

        return result
