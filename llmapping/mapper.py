from __future__ import annotations

import json
from typing import Any, Union


_SYSTEM_PROMPT = (
    "You are a data transformation assistant. "
    "Given source data and a target schema example, produce a JSON object "
    "that maps the source data into the target schema structure.\n\n"
    "Rules:\n"
    "1. Preserve the exact keys and nesting of the target schema.\n"
    "2. Map source values to target fields by understanding the semantic meaning of field names.\n"
    "3. If a target field cannot be filled from the source data, keep the default value from the target example.\n"
    "4. Combine or split source fields when necessary (e.g. first_name + last_name -> fullName).\n"
    "5. Return ONLY the resulting JSON object, nothing else."
)


def _build_user_prompt(source: Any, target_example: Any) -> str:
    return (
        f"Source data:\n{json.dumps(source, ensure_ascii=False, indent=2)}\n\n"
        f"Target schema example:\n{json.dumps(target_example, ensure_ascii=False, indent=2)}\n\n"
        "Convert the source data to match the target schema."
    )


class Llmapping:
    """Use an LLM to convert data from one schema to another."""

    def __init__(
        self,
        client: Any,
        model: str = "gpt-4o-mini",
    ) -> None:
        self.client = client
        self.model = model

    def convert(
        self,
        source: Union[dict, list],
        target_example: Union[dict, list],
    ) -> Union[dict, list]:
        """Map *source* data into the structure described by *target_example*.

        Parameters
        ----------
        source:
            The data to convert.
        target_example:
            An example object (or list of objects) showing the desired output
            schema. Values act as defaults for fields that cannot be mapped.

        Returns
        -------
        dict | list
            The converted data matching the target schema.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(source, target_example)},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )

        content = response.choices[0].message.content
        result = json.loads(content)

        # If the LLM wraps the result in an envelope key, unwrap it so the
        # returned shape matches the target example type (dict vs list).
        if isinstance(target_example, list) and isinstance(result, dict):
            # Attempt to find a list value inside the wrapper dict.
            for value in result.values():
                if isinstance(value, list):
                    return value
            return result

        return result
