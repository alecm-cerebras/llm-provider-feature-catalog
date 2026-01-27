"""
Structured Outputs Demo (Multi-Provider)

Goal
----
Demonstrate the difference between:

1) PROMPT_JSON (baseline)
   - We only *ask* the model to output JSON in the prompt.
   - We must "best-effort extract" JSON from text (brittle).
   - We validate locally with jsonschema.

2) STRUCTURED_API (true structured outputs)
   - We use the provider's *native* structured output mechanism when available:
     - OpenAI/Cerebras/Groq/Fireworks: response_format={"type":"json_schema", ...}
     - Together: response_format={"type":"json_schema","schema":...}
     - Anthropic: beta structured outputs via client.beta.messages.create(..., output_format={...})
     - Bedrock Converse: enforce schema via toolConfig/tools + toolChoice and read toolUse.input
   - Parsing is simpler and usually more reliable.
   - We still validate locally with jsonschema for parity.

Providers supported:
- cerebras
- openai
- groq
- fireworks
- together
- anthropic
- bedrock (Converse API)

Usage
-----
Run both modes for a provider:
  uv run --env-file .env python examples/python/structured_output.py --provider cerebras --model zai-glm-4.7 --mode both

Run only baseline:
  uv run --env-file .env python examples/python/structured_output.py --provider cerebras --model zai-glm-4.7 --mode prompt_json

Run only structured outputs:
  uv run --env-file .env python examples/python/structured_output.py --provider cerebras --model zai-glm-4.7 --mode structured_api

Notes
-----
- This script intentionally does NOT try to produce "real citations".
  The schema includes `sources`, but we constrain it to be ["none"] always unless you extend
  the script with retrieval. This avoids "hallucinated URLs" and keeps the demo focused.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Literal


# Anthropic structured outputs beta flag name (may change over time)
ANTHROPIC_STRUCTURED_OUTPUTS_BETA = "structured-outputs-2025-11-13"

Mode = Literal["prompt_json", "structured_api"]


# -----------------------------
# Utilities
# -----------------------------

def _require_env(var_name: str) -> str:
    value = os.environ.get(var_name)
    if not value:
        raise SystemExit(f"Missing required environment variable: {var_name}")
    return value


def _extract_first_json_object(text: str) -> dict[str, Any]:
    """
    Best-effort extraction of the first top-level {...} JSON object from a string.
    Used primarily in PROMPT_JSON mode, where models may add extra text.
    """
    if not text:
        raise ValueError("Empty model output")

    stripped = text.strip()

    # Fast path: already a JSON object
    if stripped.startswith("{") and stripped.endswith("}"):
        return json.loads(stripped)

    # Scan for first balanced JSON object by brace counting
    start = stripped.find("{")
    if start == -1:
        raise ValueError("No '{' found in output; cannot locate JSON object")

    depth = 0
    for i in range(start, len(stripped)):
        ch = stripped[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = stripped[start : i + 1]
                return json.loads(candidate)

    raise ValueError("Could not find a complete JSON object in output")


def _schema() -> dict[str, Any]:
    """
    Shared JSON Schema used across all providers.

    IMPORTANT: We constrain `sources` to avoid "hallucinated URLs" in this demo.
    If you add retrieval later, change the schema and prompt accordingly.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "answer": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "sources": {
                "type": "array",
                "items": {"type": "string", "enum": ["none"]},
            },
        },
        "required": ["answer", "confidence", "sources"],
    }


def _anthropic_schema() -> dict[str, Any]:
    """
    Anthropic structured outputs supports a restricted subset of JSON Schema.
    It commonly rejects constraints like minimum/maximum.

    We keep the provider schema compatible here, then enforce the full schema locally.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "answer": {"type": "string"},
            "confidence": {"type": "number"},
            "sources": {
                "type": "array",
                "items": {"type": "string", "enum": ["none"]},
            },
        },
        "required": ["answer", "confidence", "sources"],
    }


def _prompt(user_question: str, mode: Mode) -> str:
    """
    Prompt differs slightly by mode:
    - PROMPT_JSON: we lean harder on instruction ("JSON only") because API isn't enforcing it
    - STRUCTURED_API: API enforces schema, but we still ask for JSON-only for safety

    We also explicitly define "sources" to avoid fabricated citations:
    - Always set sources to ["none"] for this demo.
    """
    schema_text = json.dumps(_schema(), indent=2)

    if mode == "prompt_json":
        return (
            "You MUST output ONLY a single JSON object (no extra text) that matches this JSON Schema:\n"
            f"{schema_text}\n\n"
            "Rules:\n"
            "- Do not include markdown fences.\n"
            "- Do not include any extra keys.\n"
            '- Set "sources" to ["none"] (do not fabricate URLs or citations).\n\n'
            f"Question: {user_question}\n"
        )

    # structured_api mode
    return (
        "Return ONLY a single JSON object that matches this JSON Schema:\n"
        f"{schema_text}\n\n"
        "Do not include markdown fences. Do not include any extra keys.\n"
        'Set "sources" to ["none"].\n\n'
        f"Question: {user_question}\n"
    )


def validate_json(data: dict[str, Any]) -> None:
    """
    Local validation (belt-and-suspenders).
    Even if the provider enforces schema, this ensures parity across providers.
    """
    from jsonschema import validate

    validate(instance=data, schema=_schema())


# -----------------------------
# Structured output payloads
# -----------------------------

def _openai_style_response_format(*, name: str, strict: bool) -> dict[str, Any]:
    """
    OpenAI-compatible JSON Schema structured outputs format used by:
    - OpenAI
    - Cerebras
    - Groq
    - Fireworks (OpenAI-compatible API surface)

    Response characteristics:
    - choices[0].message.content is a STRING that is valid JSON
    - strict=True (when honored): schema violations are prevented / retried server-side
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": strict,
            "schema": _schema(),
        },
    }


def _together_response_format() -> dict[str, Any]:
    """
    Together.ai schema format:
    - response_format={"type":"json_schema","schema":{...}}
    - No strict flag in this shape
    """
    return {"type": "json_schema", "schema": _schema()}


# -----------------------------
# Provider callers
# -----------------------------

def call_openai_compatible(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    response_format: dict[str, Any] | None,
) -> str:
    """
    Generic OpenAI-compatible caller.

    API response shape:
    - resp.choices[0].message.content -> STRING
    """
    # pip install openai
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)

    kwargs: dict[str, Any] = {}
    if response_format is not None:
        kwargs["response_format"] = response_format

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a JSON-only API. Output strictly valid JSON."},
            {"role": "user", "content": prompt},
        ],
        **kwargs,
    )
    return (resp.choices[0].message.content or "").strip()


def call_cerebras(*, api_key: str, model: str, prompt: str, strict: bool) -> str:
    """
    Cerebras Inference (Cerebras SDK; OpenAI-compatible chat.completions).

    Response shape:
    - resp.choices[0].message.content is a STRING (JSON when structured outputs enabled)
    """
    # pip install cerebras_cloud_sdk
    from cerebras.cloud.sdk import Cerebras

    client = Cerebras(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a JSON-only API. Output strictly valid JSON."},
            {"role": "user", "content": prompt},
        ],
        response_format=_openai_style_response_format(name="answer_schema", strict=strict),
    )
    return (resp.choices[0].message.content or "").strip()


def call_openai(*, api_key: str, model: str, prompt: str, strict: bool) -> str:
    """
    OpenAI (OpenAI-compatible chat.completions).

    Response shape:
    - resp.choices[0].message.content is a STRING (JSON when structured outputs enabled)
    """
    return call_openai_compatible(
        base_url="https://api.openai.com/v1",
        api_key=api_key,
        model=model,
        prompt=prompt,
        response_format=_openai_style_response_format(name="answer_schema", strict=strict),
    )


def call_groq(*, api_key: str, model: str, prompt: str, strict: bool) -> str:
    """
    Groq (OpenAI-compatible chat.completions).

    Response shape:
    - resp.choices[0].message.content is a STRING (JSON when schema mode is honored)
    """
    return call_openai_compatible(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key,
        model=model,
        prompt=prompt,
        response_format=_openai_style_response_format(name="answer_schema", strict=strict),
    )


def call_fireworks(*, api_key: str, model: str, prompt: str) -> str:
    """
    Fireworks (OpenAI-compatible chat.completions).

    Response shape:
    - resp.choices[0].message.content is a STRING
    Caveat:
    - Some models may ignore strict; we keep strict=False and allow fallback parsing.
    """
    return call_openai_compatible(
        base_url="https://api.fireworks.ai/inference/v1",
        api_key=api_key,
        model=model,
        prompt=prompt,
        response_format=_openai_style_response_format(name="answer_schema", strict=False),
    )


def call_together(*, api_key: str, model: str, prompt: str, use_structured: bool) -> str:
    """
    Together.ai (Together SDK chat.completions).

    Response shape:
    - OpenAI-like: resp.choices[0].message.content is a STRING

    Structured outputs:
    - Supported via response_format={"type":"json_schema","schema":...}
    """
    # pip install together
    import together

    client = together.Together(api_key=api_key)

    kwargs: dict[str, Any] = {}
    if use_structured:
        kwargs["response_format"] = _together_response_format()

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a JSON-only API. Output strictly valid JSON."},
            {"role": "user", "content": prompt},
        ],
        **kwargs,
    )
    return (resp.choices[0].message.content or "").strip()


def call_anthropic(*, api_key: str, model: str, prompt: str, use_structured: bool) -> str:
    """
    Anthropic Messages API.

    Response shape:
    - resp.content is a LIST of blocks, often type="text"
    - We concatenate text blocks into a STRING

    Structured outputs (beta):
    - client.beta.messages.create(..., betas=[...], output_format={"type":"json_schema","schema":...})
    """
    # pip install anthropic
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)

    if use_structured:
        resp = client.beta.messages.create(
            model=model,
            max_tokens=512,
            betas=[ANTHROPIC_STRUCTURED_OUTPUTS_BETA],
            messages=[{"role": "user", "content": prompt}],
            output_format={"type": "json_schema", "schema": _anthropic_schema()},
        )
    else:
        # Baseline mode: plain text completion with instructions only
        resp = client.messages.create(
            model=model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
            system="You are a JSON-only API. Output strictly valid JSON.",
        )

    chunks: list[str] = []
    for block in getattr(resp, "content", []) or []:
        if getattr(block, "type", None) == "text":
            chunks.append(getattr(block, "text", "") or "")
    return "".join(chunks).strip()


def call_bedrock_prompt_json(*, model: str, prompt: str, region: str | None) -> str:
    """
    Bedrock Converse baseline mode (PROMPT_JSON):
    - We do NOT use toolConfig, just prompt the model to return JSON.
    - Response is text blocks; we concatenate to a STRING and parse/extract locally.
    """
    # pip install boto3
    import boto3

    region_name = region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if not region_name:
        raise SystemExit("Bedrock requires --region or AWS_REGION/AWS_DEFAULT_REGION env var")

    client = boto3.client("bedrock-runtime", region_name=region_name)
    resp = client.converse(
        modelId=model,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        system=[{"text": "You are a JSON-only API. Output strictly valid JSON."}],
    )

    msg = (resp.get("output") or {}).get("message") or {}
    content = msg.get("content") or []
    texts = [b.get("text", "") for b in content if isinstance(b, dict) and "text" in b]
    return "".join(texts).strip()


def call_bedrock_structured(*, model: str, prompt: str, region: str | None) -> dict[str, Any]:
    """
    Bedrock Converse structured mode (STRUCTURED_API):
    - Define a tool whose inputSchema is our target JSON Schema
    - Force the model to respond via toolChoice={"any":{}}
    - Extract JSON from toolUse.input (already parsed dict)
    """
    # pip install boto3
    import boto3

    region_name = region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if not region_name:
        raise SystemExit("Bedrock requires --region or AWS_REGION/AWS_DEFAULT_REGION env var")

    client = boto3.client("bedrock-runtime", region_name=region_name)

    tool_name = "emit_answer_json"
    resp = client.converse(
        modelId=model,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        system=[
            {"text": "You must respond by calling the tool exactly once. Do not produce any free-form text."}
        ],
        toolConfig={
            "tools": [
                {
                    "toolSpec": {
                        "name": tool_name,
                        "description": "Return the final answer as JSON matching the schema.",
                        "inputSchema": {"json": _schema()},
                    }
                }
            ],
            "toolChoice": {"any": {}},
        },
    )

    msg = (resp.get("output") or {}).get("message") or {}
    content = msg.get("content") or []

    for block in content:
        if isinstance(block, dict) and "toolUse" in block and isinstance(block["toolUse"], dict):
            tool_use = block["toolUse"]
            if tool_use.get("name") == tool_name and isinstance(tool_use.get("input"), dict):
                return tool_use["input"]

    raise ValueError("Bedrock structured response did not include expected toolUse block with JSON input")


# -----------------------------
# Run + Report
# -----------------------------

@dataclass
class RunResult:
    provider: str
    model: str
    mode: Mode
    ok: bool
    elapsed_ms: int
    parse_method: str
    raw_preview: str
    error: str | None = None
    data: dict[str, Any] | None = None


def _parse_and_validate(*, mode: Mode, raw_text: str | None, raw_dict: dict[str, Any] | None) -> tuple[dict[str, Any], str]:
    """
    Returns (data, parse_method).
    - If raw_dict is provided (e.g., Bedrock toolUse.input), no JSON parsing is needed.
    - Otherwise parse raw_text via json.loads first, fallback to extraction in prompt_json.
    """
    if raw_dict is not None:
        data = raw_dict
        validate_json(data)
        return data, "toolUse.input (dict)"

    assert raw_text is not None
    try:
        data = json.loads(raw_text)
        validate_json(data)
        return data, "json.loads"
    except Exception:
        # In STRUCTURED_API, a failure is noteworthy; still try extraction for diagnostics.
        if mode == "structured_api":
            data = _extract_first_json_object(raw_text)
            validate_json(data)
            return data, "extract_first_json_object (unexpected in structured_api)"
        # In PROMPT_JSON, extraction is expected sometimes.
        data = _extract_first_json_object(raw_text)
        validate_json(data)
        return data, "extract_first_json_object"


def run_once(*, provider: str, model: str, mode: Mode, question: str, region: str | None, strict: bool) -> RunResult:
    """
    Executes a single run for (provider, mode).
    Returns a RunResult with parse/validation status and small preview.
    """
    prompt = _prompt(question, mode=mode)

    t0 = time.time()
    raw_text: str | None = None
    raw_dict: dict[str, Any] | None = None

    try:
        if provider == "cerebras":
            if mode == "structured_api":
                raw_text = call_cerebras(api_key=_require_env("CEREBRAS_API_KEY"), model=model, prompt=prompt, strict=strict)
            else:
                # PROMPT_JSON: call via OpenAI-compatible client without response_format
                raw_text = call_openai_compatible(
                    base_url="https://api.cerebras.ai/v1",
                    api_key=_require_env("CEREBRAS_API_KEY"),
                    model=model,
                    prompt=prompt,
                    response_format=None,
                )

        elif provider == "openai":
            if mode == "structured_api":
                raw_text = call_openai(api_key=_require_env("OPENAI_API_KEY"), model=model, prompt=prompt, strict=strict)
            else:
                raw_text = call_openai_compatible(
                    base_url="https://api.openai.com/v1",
                    api_key=_require_env("OPENAI_API_KEY"),
                    model=model,
                    prompt=prompt,
                    response_format=None,
                )

        elif provider == "groq":
            if mode == "structured_api":
                raw_text = call_groq(api_key=_require_env("GROQ_API_KEY"), model=model, prompt=prompt, strict=strict)
            else:
                raw_text = call_openai_compatible(
                    base_url="https://api.groq.com/openai/v1",
                    api_key=_require_env("GROQ_API_KEY"),
                    model=model,
                    prompt=prompt,
                    response_format=None,
                )

        elif provider == "fireworks":
            if mode == "structured_api":
                raw_text = call_fireworks(api_key=_require_env("FIREWORKS_API_KEY"), model=model, prompt=prompt)
            else:
                raw_text = call_openai_compatible(
                    base_url="https://api.fireworks.ai/inference/v1",
                    api_key=_require_env("FIREWORKS_API_KEY"),
                    model=model,
                    prompt=prompt,
                    response_format=None,
                )

        elif provider == "together":
            raw_text = call_together(
                api_key=_require_env("TOGETHER_API_KEY"),
                model=model,
                prompt=prompt,
                use_structured=(mode == "structured_api"),
            )

        elif provider == "anthropic":
            raw_text = call_anthropic(
                api_key=_require_env("ANTHROPIC_API_KEY"),
                model=model,
                prompt=prompt,
                use_structured=(mode == "structured_api"),
            )

        elif provider == "bedrock":
            if mode == "structured_api":
                raw_dict = call_bedrock_structured(model=model, prompt=prompt, region=region)
            else:
                raw_text = call_bedrock_prompt_json(model=model, prompt=prompt, region=region)

        else:
            raise SystemExit(f"Unsupported provider: {provider}")

        elapsed_ms = int((time.time() - t0) * 1000)

        data, parse_method = _parse_and_validate(mode=mode, raw_text=raw_text, raw_dict=raw_dict)

        preview_src = raw_text if raw_text is not None else json.dumps(raw_dict or {})
        preview = (preview_src[:240] + "…") if len(preview_src) > 240 else preview_src

        return RunResult(
            provider=provider,
            model=model,
            mode=mode,
            ok=True,
            elapsed_ms=elapsed_ms,
            parse_method=parse_method,
            raw_preview=preview,
            data=data,
        )

    except Exception as e:
        elapsed_ms = int((time.time() - t0) * 1000)
        preview_src = raw_text if raw_text is not None else json.dumps(raw_dict or {})
        preview = (preview_src[:240] + "…") if len(preview_src) > 240 else preview_src
        return RunResult(
            provider=provider,
            model=model,
            mode=mode,
            ok=False,
            elapsed_ms=elapsed_ms,
            parse_method="(none)",
            raw_preview=preview,
            error=f"{type(e).__name__}: {e}",
            data=None,
        )


def print_result(r: RunResult) -> None:
    status = "PASS" if r.ok else "FAIL"
    print(f"\n[{status}] provider={r.provider} mode={r.mode} model={r.model} elapsed_ms={r.elapsed_ms} parse={r.parse_method}")
    if r.ok and r.data is not None:
        print(json.dumps(r.data, indent=2))
    else:
        print(f"error: {r.error}")
        if r.raw_preview:
            print(f"raw_preview: {r.raw_preview}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Structured outputs demo (prompt_json vs structured_api) for multiple providers.")
    parser.add_argument(
        "--provider",
        required=True,
        choices=["cerebras", "openai", "groq", "fireworks", "together", "anthropic", "bedrock"],
        help="Provider to call",
    )
    parser.add_argument("--model", required=True, help="Model name/id for the provider")
    parser.add_argument(
        "--mode",
        default="both",
        choices=["prompt_json", "structured_api", "both"],
        help="Which demo mode(s) to run",
    )
    parser.add_argument(
        "--question",
        default="List two reasons fast inference matters.",
        help="Question to answer in structured JSON",
    )
    parser.add_argument(
        "--region",
        default=None,
        help="AWS region (Bedrock only). If omitted uses AWS_REGION/AWS_DEFAULT_REGION.",
    )
    parser.add_argument(
        "--no_strict",
        action="store_true",
        help="Disable strict schema enforcement where supported (OpenAI/Cerebras/Groq).",
    )
    args = parser.parse_args()

    provider = args.provider.lower()
    strict = not args.no_strict

    if args.mode in ("prompt_json", "both"):
        r1 = run_once(provider=provider, model=args.model, mode="prompt_json", question=args.question, region=args.region, strict=strict)
        print_result(r1)

    if args.mode in ("structured_api", "both"):
        r2 = run_once(provider=provider, model=args.model, mode="structured_api", question=args.question, region=args.region, strict=strict)
        print_result(r2)


if __name__ == "__main__":
    main()
