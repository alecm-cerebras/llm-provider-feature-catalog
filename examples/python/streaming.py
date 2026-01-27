"""
Streaming Chat Demo (Multi-Provider) — with API setup + response-format details

This script streams tokens from a chosen provider and prints them to stdout.

Why this exists
--------------
When customers ask "Is provider X compatible with OpenAI streaming?" the real answer is:
- Some providers are OpenAI-compatible at the HTTP/API shape level (chat.completions stream events).
- Others (Anthropic, Bedrock) have their own event formats and SDK abstractions.
This script highlights:
- how you instantiate each SDK/client
- what the streaming response objects look like
- where the token text actually lives in each chunk/event

Providers supported
-------------------
- cerebras   (Cerebras SDK; OpenAI-like chat.completions stream chunks)
- openai     (OpenAI SDK; OpenAI chat.completions stream chunks)
- groq       (OpenAI-compatible base_url; chat.completions stream chunks)
- fireworks  (OpenAI-compatible base_url; chat.completions stream chunks)
- together   (OpenAI-compatible base_url; chat.completions stream chunks)
- anthropic  (Anthropic SDK; messages.stream; yields text deltas via text_stream)
- bedrock    (AWS Bedrock Runtime; converse_stream; yields events with contentBlockDelta)

Output
------
Prints the streamed text tokens in real time.
Optionally you can enable --debug to print the raw event/chunk structure.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable, Iterator, List, Optional


# -----------------------------
# Shared utilities
# -----------------------------

def _require_env(var_name: str) -> str:
    value = os.environ.get(var_name)
    if not value:
        raise SystemExit(f"Missing required environment variable: {var_name}")
    return value


def _print_stream(iterator: Iterable[str]) -> None:
    """
    Consumer that prints streamed token text to stdout.

    Note: This assumes the upstream generator yields plain strings only.
    """
    for token in iterator:
        if token:
            print(token, end="")
            sys.stdout.flush()
    print()


def _debug_print(label: str, payload: Any, enabled: bool) -> None:
    """
    Print raw event/chunk payloads to stderr (so stdout stays clean for the model text).
    """
    if not enabled:
        return
    try:
        s = json.dumps(payload, indent=2, default=str)
    except Exception:
        s = str(payload)
    print(f"\n--- DEBUG {label} ---\n{s}\n--- END DEBUG {label} ---\n", file=sys.stderr)


# -----------------------------
# OpenAI-compatible streaming
# -----------------------------

def stream_openai_compatible(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    debug: bool = False,
) -> Iterator[str]:
    """
    OpenAI-compatible providers (OpenAI, Groq, Fireworks, Together via base_url).

    Client setup:
      from openai import OpenAI
      client = OpenAI(api_key=..., base_url=...)

    Streaming call:
      stream = client.chat.completions.create(..., stream=True)

    Stream chunk shape (OpenAI-style):
      chunk.choices[0].delta.content -> str | None

    Notes:
    - Some providers also populate chunk.choices[0].delta.role or tool/function call fields.
    - This generator only yields textual deltas from delta.content.
    """
    # pip install openai
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )

    for chunk in stream:
        # chunk is typically a ChatCompletionChunk (SDK object)
        # Most important path for streaming text:
        #   chunk.choices[0].delta.content
        _debug_print("openai_compatible_chunk", getattr(chunk, "model_dump", lambda: chunk)(), debug)

        try:
            delta = chunk.choices[0].delta
            token = getattr(delta, "content", None) or ""
        except Exception:
            token = ""

        if token:
            yield token


# -----------------------------
# Cerebras streaming
# -----------------------------

def stream_cerebras(
    *,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    debug: bool = False,
) -> Iterator[str]:
    """
    Cerebras Inference streaming.

    Client setup:
      from cerebras.cloud.sdk import Cerebras
      client = Cerebras(api_key=...)

    Streaming call (OpenAI-like):
      stream = client.chat.completions.create(..., stream=True)

    Stream chunk shape (OpenAI-style):
      chunk.choices[0].delta.content -> str | None

    Notes:
    - Cerebras SDK is designed to feel OpenAI-compatible at the object level.
    - This yields only the incremental text.
    """
    # pip install cerebras_cloud_sdk
    from cerebras.cloud.sdk import Cerebras

    client = Cerebras(api_key=api_key)
    stream = client.chat.completions.create(
        messages=messages,
        model=model,
        stream=True,
    )

    for chunk in stream:
        _debug_print("cerebras_chunk", getattr(chunk, "model_dump", lambda: chunk)(), debug)
        try:
            token = chunk.choices[0].delta.content or ""
        except Exception:
            token = ""
        if token:
            yield token


# -----------------------------
# Anthropic streaming
# -----------------------------

def stream_anthropic(
    *,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    debug: bool = False,
) -> Iterator[str]:
    """
    Anthropic streaming (Messages API).

    Client setup:
      from anthropic import Anthropic
      client = Anthropic(api_key=...)

    Streaming call:
      with client.messages.stream(model=..., messages=[...]) as stream:
          for text in stream.text_stream:
              yield text

    Response/event shape:
    - Anthropic exposes a high-level iterator `stream.text_stream` that yields text deltas directly.
    - Under the hood there are richer events (message_start, content_block_delta, etc),
      but the SDK abstracts those away via text_stream.

    Important limitation in this simple demo:
    - We collapse OpenAI-style multi-turn messages into a single user message, because
      Anthropic's messages API supports multi-turn but roles differ slightly and content blocks
      can be structured. For full fidelity, you'd map each turn and optionally include system.
    """
    # pip install anthropic
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)

    # Basic conversion: concatenate user turns for a simple demo.
    user_text = "\n".join(m.get("content", "") for m in messages if m.get("role") == "user") or ""

    with client.messages.stream(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": user_text}],
    ) as stream:
        # stream.text_stream yields incremental text only (already "just tokens")
        for text in stream.text_stream:
            if debug:
                # text_stream doesn't expose full event, just the incremental text.
                _debug_print("anthropic_text_delta", {"text": text}, debug)
            yield text


# -----------------------------
# AWS Bedrock streaming (Converse API)
# -----------------------------

def stream_bedrock(
    *,
    model: str,
    messages: List[Dict[str, str]],
    region: Optional[str],
    debug: bool = False,
) -> Iterator[str]:
    """
    AWS Bedrock Runtime streaming (Converse API).

    Client setup:
      import boto3
      client = boto3.client("bedrock-runtime", region_name=...)

    Streaming call:
      resp = client.converse_stream(modelId=..., messages=[...])

    Response shape:
      resp["stream"] is an iterable of events (dicts). Example event types include:
        - {"messageStart": {...}}
        - {"contentBlockStart": {...}}
        - {"contentBlockDelta": {"delta": {"text": "..."}, ...}}
        - {"contentBlockStop": {...}}
        - {"messageStop": {...}}
        - {"metadata": {...}}

    Where text lives (common case):
      event["contentBlockDelta"]["delta"]["text"]

    Notes:
    - Bedrock uses content blocks; messages contain lists of blocks like [{"text": "..."}].
    - This demo maps OpenAI roles:
        "assistant" -> "assistant"
        anything else -> "user"
    """
    # pip install boto3
    import boto3

    region_name = region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if not region_name:
        raise SystemExit("Bedrock requires --region or AWS_REGION/AWS_DEFAULT_REGION env var")

    client = boto3.client("bedrock-runtime", region_name=region_name)

    # Convert OpenAI-style messages to Bedrock Converse format.
    # Bedrock expects:
    #   [{"role":"user","content":[{"text":"..."}]}, {"role":"assistant","content":[{"text":"..."}]}]
    bedrock_messages: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role", "user")
        br_role = "assistant" if role == "assistant" else "user"
        bedrock_messages.append({"role": br_role, "content": [{"text": m.get("content", "")}]})

    resp = client.converse_stream(
        modelId=model,
        messages=bedrock_messages,
    )

    # resp["stream"] yields dict events
    for event in resp.get("stream", []):
        _debug_print("bedrock_event", event, debug)

        # Most common place for incremental text
        if "contentBlockDelta" in event:
            delta = event["contentBlockDelta"].get("delta", {})
            text = delta.get("text")
            if text:
                yield text


# -----------------------------
# CLI
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Stream a chat completion from a chosen provider (with response-format notes).")
    parser.add_argument(
        "--provider",
        required=True,
        choices=["cerebras", "bedrock", "groq", "fireworks", "together", "openai", "anthropic"],
        help="LLM provider",
    )
    parser.add_argument("--model", required=True, help="Model name/id for the provider")
    parser.add_argument(
        "--prompt",
        default="Why is fast inference important?",
        help="User prompt text",
    )
    parser.add_argument(
        "--region",
        default=None,
        help="AWS region (Bedrock only). If omitted uses AWS_REGION/AWS_DEFAULT_REGION.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print raw chunk/event structures to stderr (useful to understand response formats).",
    )

    args = parser.parse_args()

    # OpenAI-style messages used by most providers in this demo.
    # For richer demos, you can add system/assistant turns here.
    messages: List[Dict[str, str]] = [{"role": "user", "content": args.prompt}]
    provider = args.provider.lower()

    if provider == "cerebras":
        api_key = _require_env("CEREBRAS_API_KEY")
        iterator = stream_cerebras(api_key=api_key, model=args.model, messages=messages, debug=args.debug)

    elif provider == "anthropic":
        api_key = _require_env("ANTHROPIC_API_KEY")
        iterator = stream_anthropic(api_key=api_key, model=args.model, messages=messages, debug=args.debug)

    elif provider == "bedrock":
        # Bedrock uses standard AWS credentials; no single API key env var.
        iterator = stream_bedrock(model=args.model, messages=messages, region=args.region, debug=args.debug)

    elif provider == "groq":
        api_key = _require_env("GROQ_API_KEY")
        iterator = stream_openai_compatible(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
            model=args.model,
            messages=messages,
            debug=args.debug,
        )

    elif provider == "fireworks":
        api_key = _require_env("FIREWORKS_API_KEY")
        iterator = stream_openai_compatible(
            base_url="https://api.fireworks.ai/inference/v1",
            api_key=api_key,
            model=args.model,
            messages=messages,
            debug=args.debug,
        )

    elif provider == "together":
        api_key = _require_env("TOGETHER_API_KEY")
        iterator = stream_openai_compatible(
            base_url="https://api.together.xyz/v1",
            api_key=api_key,
            model=args.model,
            messages=messages,
            debug=args.debug,
        )

    elif provider == "openai":
        api_key = _require_env("OPENAI_API_KEY")
        iterator = stream_openai_compatible(
            base_url="https://api.openai.com/v1",
            api_key=api_key,
            model=args.model,
            messages=messages,
            debug=args.debug,
        )

    else:
        raise SystemExit(f"Unsupported provider: {provider}")

    _print_stream(iterator)


if __name__ == "__main__":
    main()
