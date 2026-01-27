"""
Tool Calling Demo (Multi-Provider)
=================================

This script compares tool calling across providers, including:

1) Strict mode tool calling (schema-constrained arguments where supported)
2) Multi-turn tool calling (model calls a tool, you return results, model continues)
3) Parallel tool calling (model requests multiple tool invocations in ONE turn)

IMPORTANT: Not every provider supports every feature
----------------------------------------------------
OpenAI-style tool calling providers (OpenAI, Cerebras, Groq, Fireworks, Together*)
  - Tool calling: ✅
  - Multi-turn: ✅ (client loop)
  - Parallel tool calls in one turn: ✅ (model-dependent + provider support)
  - Strict mode tool arguments: ✅ OpenAI + Cerebras (documented).
    Groq/Fireworks/Together may accept the fields but enforcement can vary by model/provider.

Anthropic (Claude Messages API)
  - Tool calling: ✅ (tool_use/tool_result blocks)
  - Multi-turn: ✅ (client loop)
  - Parallel tool calls: ✅ (model may emit multiple tool_use blocks)
  - Strict mode: ❌ (no OpenAI-style strict flag; rely on schema + model behavior)

AWS Bedrock Converse API
  - Tool calling: ✅ (toolUse/toolResult blocks)
  - Multi-turn: ✅ (client loop)
  - Parallel tool calls: ✅ (model may return multiple toolUse blocks)
  - Strict mode: ❌ (no universal strict flag; rely on schema + model behavior)

Key fix vs earlier versions
---------------------------
Do NOT set tool_choice="required" on every turn.

If you force tool_choice="required" for *every* model call, the model is not allowed to stop
calling tools, so it may never produce a final answer and your loop can hit max_turns.

Doc-aligned behavior:
- Optionally force tool usage on the FIRST turn only (to guarantee the demo triggers tool calling)
- After you observe tool calls once, switch tool_choice to "auto" (or omit it) so the model can finish.

This script implements that behavior for OpenAI-style providers (including Cerebras).

Usage (uv)
----------
uv run --env-file .env python examples/python/tool_calling.py --provider cerebras --model <MODEL> --parallel --strict
uv run --env-file .env python examples/python/tool_calling.py --provider openai --model <MODEL> --no-parallel --strict
uv run --env-file .env python examples/python/tool_calling.py --provider bedrock --model <MODEL> --region us-east-1
uv run --env-file .env python examples/python/tool_calling.py --provider groq --model <MODEL> --debug
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# -----------------------------
# Utilities
# -----------------------------

def _require_env(var_name: str) -> str:
    value = os.environ.get(var_name)
    if not value:
        raise SystemExit(f"Missing required environment variable: {var_name}")
    return value


def _debug_print(label: str, payload: Any, enabled: bool) -> None:
    if not enabled:
        return
    try:
        s = json.dumps(payload, indent=2, default=str)
    except Exception:
        s = str(payload)
    print(f"\n--- DEBUG {label} ---\n{s}\n--- END DEBUG {label} ---\n", file=sys.stderr)


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


# -----------------------------
# Local tools (your application layer)
# -----------------------------

def tool_get_weather(*, location: str, unit: str) -> Dict[str, Any]:
    """
    Dummy tool implementation.
    In real deployments, this would call a weather API.
    """
    temps_c = {
        "San Francisco": 16,
        "New York": 4,
        "Toronto": -2,
        "Montreal": -5,
        "London": 7,
    }
    base_c = temps_c.get(location, 10)
    if unit == "fahrenheit":
        temp = round(base_c * 9 / 5 + 32, 1)
        return {"location": location, "unit": unit, "temperature": temp}
    return {"location": location, "unit": unit, "temperature": float(base_c)}


def tool_get_time(*, city: str) -> Dict[str, Any]:
    """
    Dummy tool implementation.
    In real deployments, this might query a timezone DB or system clock.
    """
    fake_times = {
        "San Francisco": "08:15",
        "New York": "11:15",
        "Toronto": "11:15",
        "Montreal": "11:15",
        "London": "16:15",
    }
    return {"city": city, "local_time": fake_times.get(city, "12:00")}


TOOL_REGISTRY = {
    "get_weather": tool_get_weather,
    "get_time": tool_get_time,
}


# -----------------------------
# Tool schemas (per provider format)
# -----------------------------

OPENAI_STYLE_TOOLS_STRICT: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location.",
            # Strict mode is documented for OpenAI + Cerebras; others may ignore.
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name, e.g. 'San Francisco'"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location", "unit"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the local time for a city.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name, e.g. 'New York'"},
                },
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    },
]

OPENAI_STYLE_TOOLS_NON_STRICT: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string"},
                },
                "required": ["location", "unit"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the local time for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    },
]

# Anthropic tool format:
#   {"name": "...", "description": "...", "input_schema": {...}}
ANTHROPIC_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a location.",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location", "unit"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_time",
        "description": "Get the local time for a city.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    },
]

# Bedrock toolConfig schema
BEDROCK_TOOL_CONFIG: Dict[str, Any] = {
    "tools": [
        {
            "toolSpec": {
                "name": "get_weather",
                "description": "Get the current weather for a location.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                        },
                        "required": ["location", "unit"],
                        "additionalProperties": False,
                    }
                },
            }
        },
        {
            "toolSpec": {
                "name": "get_time",
                "description": "Get the local time for a city.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                        "additionalProperties": False,
                    }
                },
            }
        },
    ],
    "toolChoice": {"any": {}},
}


# -----------------------------
# Demo prompt
# -----------------------------

def _prompt_for_parallel_demo() -> str:
    """
    Designed to encourage parallel tool calling:
    - two independent lookups: weather (SF) and time (NY)
    """
    return (
        "You have access to tools.\n"
        "Task:\n"
        "1) Get the current weather in San Francisco in celsius.\n"
        "2) Get the local time in New York.\n"
        "Then write ONE short sentence comparing whether it's a good time to go for a walk.\n"
        "Use tools for the lookups.\n"
    )


# -----------------------------
# OpenAI-style tool call helpers
# -----------------------------

def _extract_openai_style_tool_calls(msg: Any) -> List[Dict[str, Any]]:
    """
    OpenAI-style tool calls:
      msg.tool_calls -> list
      each call has:
        call.id
        call.function.name
        call.function.arguments (JSON string)
    """
    calls: List[Dict[str, Any]] = []
    tool_calls = getattr(msg, "tool_calls", None) or []
    for c in tool_calls:
        fn = getattr(c, "function", None)
        calls.append(
            {
                "id": getattr(c, "id", None),
                "name": getattr(fn, "name", None),
                "arguments": getattr(fn, "arguments", None),
            }
        )
    return calls


def _openai_style_messages_append_tool_result(
    *,
    messages: List[Dict[str, Any]],
    tool_call_id: str,
    result_obj: Any,
) -> None:
    """
    OpenAI-style tool result message:
      {"role":"tool","content":"<json-string>","tool_call_id":"<id>"}
    """
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": _json(result_obj),
        }
    )


# -----------------------------
# Provider calls
# -----------------------------

def call_openai_compatible_chat(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Any] = None,
    parallel_tool_calls: Optional[bool] = None,
    debug: bool = False,
):
    """
    OpenAI SDK chat.completions.create call.

    Response format:
      resp.choices[0].message.content -> assistant text (may be empty if tool call)
      resp.choices[0].message.tool_calls -> list of tool call requests (if any)
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)

    kwargs: Dict[str, Any] = {}
    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    if parallel_tool_calls is not None:
        kwargs["parallel_tool_calls"] = parallel_tool_calls

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs,
    )

    try:
        dump = resp.choices[0].message.model_dump()
    except Exception:
        dump = str(resp)
    _debug_print("openai_compatible_response_message", dump, debug)

    return resp


def call_cerebras_chat(
    *,
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Any] = None,
    parallel_tool_calls: Optional[bool] = None,
    debug: bool = False,
):
    """
    Cerebras SDK chat.completions.create

    Response format (OpenAI-like):
      resp.choices[0].message.content
      resp.choices[0].message.tool_calls
    """
    from cerebras.cloud.sdk import Cerebras

    client = Cerebras(api_key=api_key)

    kwargs: Dict[str, Any] = {}
    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    if parallel_tool_calls is not None:
        kwargs["parallel_tool_calls"] = parallel_tool_calls

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs,
    )

    try:
        dump = resp.choices[0].message.model_dump()
    except Exception:
        dump = str(resp)
    _debug_print("cerebras_response_message", dump, debug)

    return resp


# -----------------------------
# Anthropic provider
# -----------------------------

def call_anthropic_messages(
    *,
    api_key: str,
    model: str,
    system: Optional[str],
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    tool_choice: Optional[Any],
    debug: bool = False,
):
    """
    Anthropic Messages API.

    Response format:
      resp.content -> list of blocks including:
        - text blocks
        - tool_use blocks: {"type":"tool_use","id":"...","name":"...","input":{...}}
    """
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)

    kwargs: Dict[str, Any] = {}
    if system is not None:
        kwargs["system"] = system
    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice

    resp = client.messages.create(
        model=model,
        max_tokens=512,
        messages=messages,
        **kwargs,
    )

    blocks = []
    for b in getattr(resp, "content", []) or []:
        try:
            blocks.append(getattr(b, "model_dump", lambda: {"type": getattr(b, "type", None)})())
        except Exception:
            blocks.append(str(b))
    _debug_print("anthropic_response_content_blocks", blocks, debug)

    return resp


def _anthropic_extract_tool_uses(resp: Any) -> List[Dict[str, Any]]:
    tool_uses: List[Dict[str, Any]] = []
    for b in getattr(resp, "content", []) or []:
        if getattr(b, "type", None) == "tool_use":
            tool_uses.append(
                {
                    "id": getattr(b, "id", None),
                    "name": getattr(b, "name", None),
                    "input": getattr(b, "input", None),
                }
            )
    return tool_uses


def _anthropic_extract_text(resp: Any) -> str:
    chunks: List[str] = []
    for b in getattr(resp, "content", []) or []:
        if getattr(b, "type", None) == "text":
            chunks.append(getattr(b, "text", "") or "")
    return "".join(chunks).strip()


# -----------------------------
# Bedrock provider
# -----------------------------

def call_bedrock_converse(
    *,
    model: str,
    region: Optional[str],
    messages: List[Dict[str, Any]],
    tool_config: Optional[Dict[str, Any]] = None,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    AWS Bedrock Runtime Converse API.

    Response format:
      resp["output"]["message"]["content"] -> list of blocks including:
        - {"text":"..."}
        - {"toolUse": {"toolUseId":"...","name":"...","input":{...}}}
    """
    import boto3

    region_name = region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if not region_name:
        raise SystemExit("Bedrock requires --region or AWS_REGION/AWS_DEFAULT_REGION env var")

    client = boto3.client("bedrock-runtime", region_name=region_name)

    kwargs: Dict[str, Any] = {}
    if tool_config is not None:
        kwargs["toolConfig"] = tool_config

    resp = client.converse(
        modelId=model,
        messages=messages,
        **kwargs,
    )

    _debug_print("bedrock_converse_response", resp, debug)
    return resp


def _bedrock_extract_tool_uses(resp: Dict[str, Any]) -> List[Dict[str, Any]]:
    tool_uses: List[Dict[str, Any]] = []
    msg = (resp.get("output") or {}).get("message") or {}
    content = msg.get("content") or []
    for b in content:
        if isinstance(b, dict) and "toolUse" in b and isinstance(b["toolUse"], dict):
            tu = b["toolUse"]
            tool_uses.append({"id": tu.get("toolUseId"), "name": tu.get("name"), "input": tu.get("input")})
    return tool_uses


def _bedrock_extract_text(resp: Dict[str, Any]) -> str:
    msg = (resp.get("output") or {}).get("message") or {}
    content = msg.get("content") or []
    texts = [b.get("text", "") for b in content if isinstance(b, dict) and "text" in b]
    return "".join(texts).strip()


# -----------------------------
# Multi-turn tool calling loops
# -----------------------------

def run_tool_loop_openai_style(
    *,
    provider_label: str,
    call_fn,
    model: str,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    force_tools_first_turn: bool,
    parallel_tool_calls: Optional[bool],
    debug: bool,
    max_turns: int = 8,
) -> str:
    """
    Multi-turn tool calling loop for OpenAI-like responses (tool_calls + tool role messages).

    CRITICAL BEHAVIOR (doc-aligned):
    - If force_tools_first_turn=True, we set tool_choice="required" ONLY on the first model call
      *until we see tool_calls at least once*.
    - After tool calls are observed, we switch tool_choice to "auto" so the model can produce a
      final answer. This avoids infinite tool loops.

    Parallel tool calling:
    - If the model returns multiple tool calls in msg.tool_calls, we execute ALL and append ALL
      tool results before asking the model to continue.
    """
    forced_tool_phase = force_tools_first_turn

    for _ in range(max_turns):
        effective_tool_choice = "required" if forced_tool_phase else "auto"

        resp = call_fn(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=effective_tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            debug=debug,
        )
        msg = resp.choices[0].message

        tool_calls = _extract_openai_style_tool_calls(msg)
        if tool_calls:
            # Once we see tool calls at least once, allow the model to finish on later turns.
            forced_tool_phase = False

            # Save assistant turn for multi-turn fidelity
            try:
                messages.append(msg.model_dump())
            except Exception:
                messages.append({"role": "assistant", "content": msg.content, "tool_calls": tool_calls})

            # Execute each tool call (parallel if multiple)
            for call in tool_calls:
                name = call.get("name")
                call_id = call.get("id")
                arg_str = call.get("arguments") or "{}"

                if not name or not call_id:
                    raise ValueError(f"{provider_label}: tool call missing name/id: {call}")
                if name not in TOOL_REGISTRY:
                    raise ValueError(f"{provider_label}: unknown tool requested: {name}")

                args_dict = json.loads(arg_str)
                result_obj = TOOL_REGISTRY[name](**args_dict)

                _debug_print(f"{provider_label}_tool_result_{name}", result_obj, debug)
                _openai_style_messages_append_tool_result(messages=messages, tool_call_id=call_id, result_obj=result_obj)

            continue

        # No tool calls => final answer
        return (msg.content or "").strip()

    raise RuntimeError(f"{provider_label}: exceeded max_turns={max_turns} without completing")


def run_tool_loop_anthropic(*, model: str, debug: bool, max_turns: int = 8) -> str:
    """
    Multi-turn tool calling loop for Anthropic.

    Notes:
    - No OpenAI-style strict flag.
    - Claude may emit multiple tool_use blocks in one response (parallel tool calling).
    """
    api_key = _require_env("ANTHROPIC_API_KEY")

    system = (
        "You are a helpful assistant that must use tools for lookups.\n"
        "After receiving tool results, produce a final natural language answer.\n"
    )
    messages: List[Dict[str, Any]] = [{"role": "user", "content": _prompt_for_parallel_demo()}]
    tool_choice = {"type": "auto"}

    for _ in range(max_turns):
        resp = call_anthropic_messages(
            api_key=api_key,
            model=model,
            system=system,
            messages=messages,
            tools=ANTHROPIC_TOOLS,
            tool_choice=tool_choice,
            debug=debug,
        )

        tool_uses = _anthropic_extract_tool_uses(resp)
        if tool_uses:
            tool_result_blocks: List[Dict[str, Any]] = []
            for tu in tool_uses:
                tu_id = tu.get("id")
                name = tu.get("name")
                tool_input = tu.get("input") or {}
                if not tu_id or not name:
                    raise ValueError(f"Anthropic tool_use missing id/name: {tu}")
                if name not in TOOL_REGISTRY:
                    raise ValueError(f"Anthropic requested unknown tool: {name}")

                result_obj = TOOL_REGISTRY[name](**tool_input)
                _debug_print(f"anthropic_tool_result_{name}", result_obj, debug)
                tool_result_blocks.append(
                    {"type": "tool_result", "tool_use_id": tu_id, "content": _json(result_obj)}
                )
            messages.append({"role": "user", "content": tool_result_blocks})
            continue

        return _anthropic_extract_text(resp)

    raise RuntimeError(f"Anthropic: exceeded max_turns={max_turns} without completing")


def run_tool_loop_bedrock(*, model: str, region: Optional[str], debug: bool, max_turns: int = 8) -> str:
    """
    Multi-turn tool calling loop for Bedrock Converse.

    Notes:
    - No universal strict flag.
    - Models may return multiple toolUse blocks in one response (parallel tool calling).
    """
    messages: List[Dict[str, Any]] = [{"role": "user", "content": [{"text": _prompt_for_parallel_demo()}]}]

    for _ in range(max_turns):
        resp = call_bedrock_converse(
            model=model,
            region=region,
            messages=messages,
            tool_config=BEDROCK_TOOL_CONFIG,
            debug=debug,
        )

        tool_uses = _bedrock_extract_tool_uses(resp)
        if tool_uses:
            tool_result_blocks: List[Dict[str, Any]] = []
            for tu in tool_uses:
                tu_id = tu.get("id")
                name = tu.get("name")
                tool_input = tu.get("input") or {}
                if not tu_id or not name:
                    raise ValueError(f"Bedrock toolUse missing id/name: {tu}")
                if name not in TOOL_REGISTRY:
                    raise ValueError(f"Bedrock requested unknown tool: {name}")

                result_obj = TOOL_REGISTRY[name](**tool_input)
                _debug_print(f"bedrock_tool_result_{name}", result_obj, debug)
                tool_result_blocks.append(
                    {"toolResult": {"toolUseId": tu_id, "content": [{"json": result_obj}]}}
                )
            messages.append({"role": "user", "content": tool_result_blocks})
            continue

        text = _bedrock_extract_text(resp)
        if text:
            return text

        messages.append({"role": "user", "content": [{"text": "Please continue."}]})

    raise RuntimeError(f"Bedrock: exceeded max_turns={max_turns} without completing")


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Tool calling demo (strict, multi-turn, parallel) across providers.")
    parser.add_argument(
        "--provider",
        required=True,
        choices=["cerebras", "openai", "groq", "fireworks", "together", "anthropic", "bedrock"],
        help="LLM provider",
    )
    parser.add_argument("--model", required=True, help="Model name/id for the provider")

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict tool calling where supported (OpenAI/Cerebras). Others may ignore.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Attempt to enable parallel tool calls where supported (OpenAI/Cerebras parameter).",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Attempt to disable parallel tool calls where supported (OpenAI/Cerebras parameter).",
    )
    parser.add_argument(
        "--force-tools-first-turn",
        action="store_true",
        help=(
            "Force tool usage on the first model turn (tool_choice='required' initially), then switch "
            "to auto so the model can finish. Recommended for demos."
        ),
    )
    parser.add_argument("--debug", action="store_true", help="Print raw tool call payloads to stderr.")
    parser.add_argument("--region", default=None, help="AWS region (Bedrock only).")

    args = parser.parse_args()
    provider = args.provider.lower()

    # OpenAI-style parallel parameter
    parallel_param: Optional[bool] = None
    if args.no_parallel:
        parallel_param = False
    elif args.parallel:
        parallel_param = True

    tools_openai_style = OPENAI_STYLE_TOOLS_STRICT if args.strict else OPENAI_STYLE_TOOLS_NON_STRICT

    # OpenAI-style message list
    messages_openai_style: List[Dict[str, Any]] = [
        {"role": "system", "content": "You are a helpful assistant that must use tools for lookups."},
        {"role": "user", "content": _prompt_for_parallel_demo()},
    ]

    if provider == "cerebras":
        api_key = _require_env("CEREBRAS_API_KEY")

        def _call_fn(**kwargs):
            return call_cerebras_chat(api_key=api_key, **kwargs)

        final = run_tool_loop_openai_style(
            provider_label="cerebras",
            call_fn=_call_fn,
            model=args.model,
            messages=messages_openai_style,
            tools=tools_openai_style,
            force_tools_first_turn=args.force_tools_first_turn,
            parallel_tool_calls=parallel_param,
            debug=args.debug,
        )
        print(final)

    elif provider == "openai":
        api_key = _require_env("OPENAI_API_KEY")

        def _call_fn(**kwargs):
            return call_openai_compatible_chat(
                base_url="https://api.openai.com/v1",
                api_key=api_key,
                **kwargs,
            )

        final = run_tool_loop_openai_style(
            provider_label="openai",
            call_fn=_call_fn,
            model=args.model,
            messages=messages_openai_style,
            tools=tools_openai_style,
            force_tools_first_turn=args.force_tools_first_turn,
            parallel_tool_calls=parallel_param,
            debug=args.debug,
        )
        print(final)

    elif provider == "groq":
        api_key = _require_env("GROQ_API_KEY")

        def _call_fn(**kwargs):
            return call_openai_compatible_chat(
                base_url="https://api.groq.com/openai/v1",
                api_key=api_key,
                **kwargs,
            )

        final = run_tool_loop_openai_style(
            provider_label="groq",
            call_fn=_call_fn,
            model=args.model,
            messages=messages_openai_style,
            tools=tools_openai_style,
            force_tools_first_turn=args.force_tools_first_turn,
            parallel_tool_calls=parallel_param,
            debug=args.debug,
        )
        print(final)

    elif provider == "fireworks":
        api_key = _require_env("FIREWORKS_API_KEY")

        def _call_fn(**kwargs):
            return call_openai_compatible_chat(
                base_url="https://api.fireworks.ai/inference/v1",
                api_key=api_key,
                **kwargs,
            )

        final = run_tool_loop_openai_style(
            provider_label="fireworks",
            call_fn=_call_fn,
            model=args.model,
            messages=messages_openai_style,
            tools=tools_openai_style,
            force_tools_first_turn=args.force_tools_first_turn,
            parallel_tool_calls=parallel_param,
            debug=args.debug,
        )
        print(final)

    elif provider == "together":
        api_key = _require_env("TOGETHER_API_KEY")

        def _call_fn(**kwargs):
            return call_openai_compatible_chat(
                base_url="https://api.together.xyz/v1",
                api_key=api_key,
                **kwargs,
            )

        final = run_tool_loop_openai_style(
            provider_label="together",
            call_fn=_call_fn,
            model=args.model,
            messages=messages_openai_style,
            tools=tools_openai_style,
            force_tools_first_turn=args.force_tools_first_turn,
            parallel_tool_calls=parallel_param,
            debug=args.debug,
        )
        print(final)

    elif provider == "anthropic":
        if args.strict:
            print(
                "Note: --strict requested, but Anthropic does not have an OpenAI-style strict flag. "
                "Proceeding with best-effort schema adherence.",
                file=sys.stderr,
            )
        if parallel_param is not None:
            print(
                "Note: --parallel/--no-parallel is an OpenAI-style parameter. Anthropic does not accept "
                "parallel_tool_calls; Claude may still emit multiple tool_use blocks.",
                file=sys.stderr,
            )

        final = run_tool_loop_anthropic(model=args.model, debug=args.debug)
        print(final)

    elif provider == "bedrock":
        if args.strict:
            print(
                "Note: --strict requested, but Bedrock does not have a universal strict flag. "
                "Proceeding with best-effort schema adherence.",
                file=sys.stderr,
            )
        if parallel_param is not None:
            print(
                "Note: --parallel/--no-parallel is an OpenAI-style parameter. Bedrock does not accept "
                "parallel_tool_calls; models may still return multiple toolUse blocks.",
                file=sys.stderr,
            )

        final = run_tool_loop_bedrock(model=args.model, region=args.region, debug=args.debug)
        print(final)

    else:
        raise SystemExit(f"Unsupported provider: {provider}")


if __name__ == "__main__":
    main()
