"""Load a reply schema, and check a reply against it.

`--validate-schema` has to mean the same thing on all three CLIs. Two of them
(claude, grok) take any JSON Schema and accept a lax one; codex rejects a
schema that is not in OpenAI's strict structured-outputs dialect with an HTTP
400 before the turn runs. A schema that works on two backends and 400s on the
third is a parity break, so the strict subset is enforced here, once, for
every backend.

The rules below are the ones that were actually measured against
`codex exec --output-schema` (2026-08-20, codex-cli 0.147.0), not a reading of
a spec: every other shape is passed through. `anyOf` and `$ref`/`$defs` in
particular are *accepted*, because codex accepts them — rejecting a schema all
three backends handle is as much a parity break as accepting one that only two
do, in the more annoying direction. Anything unmeasured that codex still
refuses falls through to its own error message, which the codex backend now
surfaces verbatim.

A reply that fails validation, including one that is not JSON at all, is a
retryable failure rather than a raise: both mean "the reply does not satisfy
the contract", and the caller's retry loop is what that is for.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from jsonschema.exceptions import SchemaError as JsonSchemaError
from jsonschema.exceptions import best_match
from jsonschema.validators import validator_for

from backends.base import OutputSchema

# Appended to the prompt whenever a schema is active, identically on every
# backend. All three CLIs constrain the reply themselves — 23/23 conforming
# replies under adversarial prompting on claude, 4/4 on codex, grok not
# measurable — so this is a hedge, not the mechanism. It stays uniform because
# per-backend prompt text is the exact divergence this interface exists to
# prevent, and because the one backend that could not be trialled must not be
# the one running without a hedge.
SCHEMA_INSTRUCTION = (
    "Reply with JSON conforming to the supplied output schema, and nothing else."
)

# Keywords codex refuses outright, each measured as a 400 with a strict schema
# on both branches. `anyOf` is deliberately absent: it was measured accepted.
FORBIDDEN_KEYWORDS = ("oneOf", "allOf", "not")

# How the root of the document is named in a rejection message. Child paths
# extend it, so a message reads like the codex error it stands in for.
ROOT_PATH = "$"

# Why a schema two backends would have accepted is rejected here anyway.
PARITY_NOTE = "codex rejects it; one schema has to mean the same thing on every backend"

# Long enough to recognise the reply, short enough that neither a stderr line
# nor the repair prompt is dominated by a model that answered with an essay.
EXCERPT_CHARS = 500


class SchemaError(Exception):
    """Base for both schema failures, so callers need one except clause."""


class SchemaLoadError(SchemaError):
    """The schema file is missing, malformed, or outside the strict subset.

    A usage-class mistake: the caller must fix the file. Nothing has run yet.
    """


class ReplyValidationError(SchemaError):
    """The reply does not satisfy the schema. Retryable, and retried.

    `correction` is the part worth showing the model on the next attempt: it
    names what was wrong without repeating the whole reply back at it.
    """

    def __init__(self, message: str, correction: str) -> None:
        super().__init__(message)
        self.correction = correction


def _excerpt(text: str) -> str:
    """Bound model-produced text before it reaches a log line or a prompt."""
    stripped = text.strip()
    if len(stripped) <= EXCERPT_CHARS:
        return stripped
    return f"{stripped[:EXCERPT_CHARS]}…"


def _declared_types(node: dict) -> list[object]:
    declared = node.get("type")
    return declared if isinstance(declared, list) else [declared]


def _is_object_node(node: dict) -> bool:
    """Does this subschema describe a JSON object?

    `properties` alone counts: a schema that lists properties but omits
    `type` is the shape codex rejects for a missing `additionalProperties`
    just the same.
    """
    return "object" in _declared_types(node) or "properties" in node


def _rule_broken(node: dict, where: str) -> str | None:
    """The first strict-subset rule `node` itself breaks, or None.

    Its children are not inspected here; `_first_violation` walks those.
    """
    for keyword in FORBIDDEN_KEYWORDS:
        if keyword in node:
            return f'{where} uses "{keyword}", which is not supported ({PARITY_NOTE})'
    if not _is_object_node(node):
        return None
    if node.get("additionalProperties") is not False:
        return f'{where} must set "additionalProperties": false ({PARITY_NOTE})'
    required = node.get("required", [])
    missing = [name for name in node.get("properties", {}) if name not in required]
    if missing:
        listed = ", ".join(f"'{name}'" for name in missing)
        return (
            f'{where} must list every property in "required"; '
            f"missing {listed} ({PARITY_NOTE})"
        )
    return None


def _subschemas(node: dict, where: str) -> Iterator[tuple[object, str]]:
    """Every child subschema of `node`, with the path to name it by.

    `$defs` entries are walked as written rather than by resolving the `$ref`s
    that reach them: following references means implementing a resolver, and
    cycles, to end up checking the same nodes. An entry nothing references yet
    is checked too — it exists to be referenced, and a lax one is a 400
    waiting for the day it is.
    """
    for name, child in node.get("properties", {}).items():
        yield child, f"{where}.properties.{name}"
    if "items" in node:
        yield node["items"], f"{where}.items"
    for index, branch in enumerate(node.get("anyOf", [])):
        yield branch, f"{where}.anyOf[{index}]"
    for name, child in node.get("$defs", {}).items():
        yield child, f"{where}.$defs.{name}"


def _first_violation(node: dict, where: str) -> str | None:
    """Depth-first, document order, so the message names one stable node."""
    broken = _rule_broken(node, where)
    if broken is not None:
        return broken
    for child, child_where in _subschemas(node, where):
        # A boolean subschema (`{"properties": {"a": true}}`) is legal and has
        # no rules to break.
        if not isinstance(child, dict):
            continue
        broken = _first_violation(child, child_where)
        if broken is not None:
            return broken
    return None


def _load_document(path: Path) -> dict:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SchemaLoadError(f"cannot read schema file {path}: {exc}") from exc
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SchemaLoadError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise SchemaLoadError(f"{path}: a JSON Schema must be a JSON object")
    return document


def load_schema(path: Path) -> OutputSchema:
    """Read `path` and return it in both forms the backends need.

    The path is resolved here rather than left relative: a turn runs with the
    orchestrator's working directory, not the shell's, so a relative path that
    the user could see would be a path codex could not open.
    """
    absolute = path.resolve()
    document = _load_document(absolute)

    validator = validator_for(document)
    try:
        validator.check_schema(document)
    except JsonSchemaError as exc:
        raise SchemaLoadError(
            f"{absolute} is not a valid JSON Schema: {exc.message}"
        ) from exc

    broken = _first_violation(document, ROOT_PATH)
    if broken is not None:
        raise SchemaLoadError(f"{absolute}: {broken}")
    # Checked after the subset rules, so a `{"oneOf": [...]}` root is reported
    # as the unsupported keyword it is rather than as a root that names no
    # type. This last rule is ours, not codex's: a reply is validated as one
    # JSON object, so a schema describing anything else can never be satisfied.
    if not _is_object_node(document):
        raise SchemaLoadError(
            f"{absolute}: the root schema must describe a JSON object; "
            f"a reply is validated as one object"
        )

    # Re-serialised compactly rather than passed through as written: this text
    # becomes one argv entry and one logged line on two of the three backends,
    # and the file's own formatting says nothing the schema does not.
    return OutputSchema(
        text=json.dumps(document, separators=(",", ":"), sort_keys=True),
        path=absolute,
    )


def validate_reply(reply: str, structured: dict | None, schema: OutputSchema) -> dict:
    """Return the reply as an object satisfying `schema`, or raise a retryable.

    `structured` is the backend's parse of the reply, and is None when the
    reply was not a JSON object at all. That is treated exactly like a schema
    violation: both say the reply broke the contract, and neither is worth
    ending the run over while a retry is still owed.
    """
    if structured is None:
        raise ReplyValidationError(
            f"the reply was not a JSON object: {_excerpt(reply)}",
            "the reply was not a JSON object",
        )
    document = json.loads(schema.text)
    validator = validator_for(document)(document)
    # best_match over the first error: under `anyOf`, the first error is
    # "does not match any branch", which names nothing the model can fix.
    error = best_match(validator.iter_errors(structured))
    if error is not None:
        correction = (
            f"the reply did not satisfy the output schema at "
            f"{error.json_path}: {_excerpt(error.message)}"
        )
        raise ReplyValidationError(correction, correction)
    return structured


def compose_schema_prompt(prompt: str) -> str:
    """The prompt the agent sees on the first attempt: user text, then the line."""
    return f"{prompt}\n\n{SCHEMA_INSTRUCTION}"


def repair_prompt(error: ReplyValidationError) -> str:
    """The prompt for a retry: what was wrong, then the same line again.

    The reply itself is not quoted back — the session already holds it, and
    the model's own text is data, not something to re-feed as instruction.
    """
    return f"That reply was rejected: {error.correction}\n\n{SCHEMA_INSTRUCTION}"
