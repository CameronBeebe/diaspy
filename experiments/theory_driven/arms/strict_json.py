"""Fail-closed JSON boundary for structured reflection, never task outputs."""

import json
from typing import Any

import dspy
from dspy.clients.base_lm import BaseLM
from dspy.utils.exceptions import AdapterParseError
from pydantic import TypeAdapter


class StrictJSONAdapter(dspy.JSONAdapter):
    """Validate wire responses and latch the first structured parsing failure.

    Share one instance across all structured roles in a compile. GEPA catches
    proposer exceptions and retries them, so callers must also check this latch
    after compile and use ``should_stop`` to prevent later optimizer iterations.
    Task predictions deliberately keep their ambient adapter.
    """

    def __init__(self):
        super().__init__()
        self.failure: AdapterParseError | None = None

    def raise_if_failed(self) -> None:
        if self.failure is not None:
            raise self.failure

    def should_stop(self, state=None) -> bool:
        return self.failure is not None

    def __call__(
        self, lm: BaseLM, lm_kwargs: dict[str, Any], signature: type[dspy.Signature],
        demos: list[dict[str, Any]], inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        # Check before the model call, not only in parse: a valid regenerated
        # response must not erase a failed prosecution, defense, or rewrite.
        self.raise_if_failed()
        try:
            return super().__call__(lm, lm_kwargs, signature, demos, inputs)
        except AdapterParseError as exc:
            self.failure = exc
            raise

    def parse(self, signature: type[dspy.Signature], completion: str) -> dict[str, Any]:
        def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"Duplicate JSON key: {key}")
                result[key] = value
            return result

        def reject_constant(value: str) -> Any:
            raise ValueError(f"Non-JSON constant: {value}")

        try:
            fields = json.loads(completion, object_pairs_hook=unique_object, parse_constant=reject_constant)
            if not isinstance(fields, dict) or fields.keys() != signature.output_fields.keys():
                raise ValueError("Expected exactly the output fields in one JSON object")
            for name, field in signature.output_fields.items():
                # DSPy's parse_value repairs strings for non-string fields. Require
                # native typed values first, including nested lists/model members.
                adapter = TypeAdapter(field.annotation or Any)
                typed = adapter.validate_python(fields[name], strict=True)
                if adapter.dump_python(typed, mode="json") != fields[name]:
                    raise ValueError(f"Field {name} would lose or change JSON members during validation")
        except ValueError as exc:
            raise AdapterParseError(
                adapter_name=type(self).__name__,
                signature=signature,  # type: ignore[arg-type]  # DSPy annotates an instance but expects its signature class.
                lm_response=completion, message=str(exc),
            ) from exc
        return super().parse(signature, completion)
