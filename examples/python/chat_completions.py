"""
Chat Completions Demo (Multi-Provider) — with API setup + response-format details

This is a non-streaming version of a multi-provider chat example. It aims to be a
"field reference" for SAs answering:

- How do I authenticate to each provider?
- What SDK call do I make?
- What does the response object look like?
- Where do I read the assistant text from?

Providers supported
-------------------
- cerebras   (Cerebras SDK; OpenAI-like chat.completions)
- openai     (OpenAI SDK; chat.completions)
- groq       (OpenAI-compatible base_url; chat.completions)
- fireworks  (OpenAI-compatible base_url; chat.completions)
- together   (OpenAI-compatible base_url; chat.completions)
- anthropic  (Anthropic SDK; messages.create)
- bedrock    (AWS Bedrock Runtime; converse)

Response format cheat-sheet 
----------------------------------------------
OpenAI-style (OpenAI/Cerebras/Groq/Fireworks/Together):
  resp.choices[0].message.content  -> str

Anthropic Messages API:
  resp.content -> list of blocks, typically {"type":"text","text":"..."}
  so: "".join(block.text for block in resp.content if block.type=="text")

Bedrock Converse API:
  resp["output"]["message"]["content"] -> list of blocks, e.g. [{"text":"..."}]
  so: "".join(block["text"] for block in content if "text" in block)

This script:
------------
- Executes one request against the chosen provider.
- Prints BOTH a provider-normalized extracted text (so you can compare providers)
  and (optionally) the raw response for debugging.

"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Shared utilities
# -----------------------------

def _require_env(var_name: str) -> str:
    value = os.environ.get(var_name)
    if not value:
        raise SystemExit(f"Missing required environment variable: {var_name}")
    return value


def _safe_json(obj: Any) -> str:
    """
    Best-effort stringify for raw responses:
    - SDK objects often have model_dump() or dict()
    - fallback to str(obj)
    """
    try:
        if hasattr(obj, "model_dump"):
            return json.dumps(obj.model_dump(), indent=2, default=str)
        if isinstance(obj, dict):
            return json.dumps(obj, indent=2, default=str)
        return json.dumps(obj, indent=2, default=str)
    except Exception:
        return str(obj)


# -----------------------------
# Provider calls
# -----------------------------

def chat_cerebras(model: str, prompt: str):
    """
    Cerebras Inference (Cerebras SDK; OpenAI-compatible chat.completions).

    Setup:
      from cerebras.cloud.sdk import Cerebras
      client = Cerebras(api_key=...)

    Call:
      client.chat.completions.create(model=..., messages=[...])

    Response shape (OpenAI-like SDK object):
      resp.choices[0].message.content -> str
    """
    from cerebras.cloud.sdk import Cerebras

    client = Cerebras(api_key=_require_env("CEREBRAS_API_KEY"))
    return client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )


def chat_openai_compatible(*, base_url: str, api_key_env: str, model: str, prompt: str):
    """
    OpenAI-compatible providers (OpenAI/Groq/Fireworks/Together).

    Setup:
      from openai import OpenAI
      client = OpenAI(api_key=..., base_url=...)

    Call:
      client.chat.completions.create(model=..., messages=[...])

    Response shape (OpenAI SDK object):
      resp.choices[0].message.content -> str
    """
    from openai import OpenAI

    client = OpenAI(api_key=_require_env(api_key_env), base_url=base_url)
    return client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )


def chat_anthropic(model: str, prompt: str):
    """
    Anthropic Messages API.

    Setup:
      from anthropic import Anthropic
      client = Anthropic(api_key=...)

    Call:
      client.messages.create(model=..., max_tokens=..., messages=[...])

    Response shape (NOT OpenAI-like):
      resp.content is a list of blocks. Example:
        resp.content == [
          {"type":"text", "text":"Hello ..."}
        ]

    Where text lives:
      - Most commonly in text blocks:
        "".join(block.text for block in resp.content if block.type=="text")
    """
    # pip install anthropic
    import anthropic

    client = anthropic.Anthropic(api_key=_require_env("ANTHROPIC_API_KEY"))
    return client.messages.create(
        model=model,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )


def chat_bedrock(model: str, prompt: str, region: str | None):
    """
    AWS Bedrock Runtime Converse API.

    Setup:
      import boto3
      client = boto3.client("bedrock-runtime", region_name=...)

    Call:
      client.converse(modelId=..., messages=[...])

    Message format:
      - Bedrock uses content blocks (list of dicts), e.g. [{"text": "..."}]
      - So a single user message looks like:
        {"role":"user","content":[{"text": prompt}]}

    Response shape (dict):
      - resp["output"]["message"]["content"] is list of blocks, e.g. [{"text":"..."}]
      - Some models may return additional block types (toolUse, images, etc.)
    """
    # pip install boto3
    import boto3

    region_name = region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if not region_name:
        raise SystemExit("Bedrock requires --region or AWS_REGION/AWS_DEFAULT_REGION env var")

    client = boto3.client("bedrock-runtime", region_name=region_name)
    return client.converse(
        modelId=model,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
    )


def chat_fireworks(model: str, prompt: str):
    """
    Fireworks (OpenAI-compatible).
    """
    return chat_openai_compatible(
        base_url="https://api.fireworks.ai/inference/v1",
        api_key_env="FIREWORKS_API_KEY",
        model=model,
        prompt=prompt,
    )


def chat_groq(model: str, prompt: str):
    """
    Groq (OpenAI-compatible).
    """
    return chat_openai_compatible(
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        model=model,
        prompt=prompt,
    )


def chat_together(model: str, prompt: str):
    """
    Together (OpenAI-compatible).
    """
    return chat_openai_compatible(
        base_url="https://api.together.xyz/v1",
        api_key_env="TOGETHER_API_KEY",
        model=model,
        prompt=prompt,
    )


def chat_openai(model: str, prompt: str):
    """
    OpenAI (OpenAI-compatible).
    """
    return chat_openai_compatible(
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        model=model,
        prompt=prompt,
    )


# -----------------------------
# Normalization: extract assistant text
# -----------------------------

def extract_text(provider: str, resp: Any) -> str:
    """
    Provider-specific extraction for the assistant text.

    This is the most practical bit for SAs:
    - It tells you where the content lives so you can port code quickly.
    """
    provider = provider.lower()

    if provider in {"cerebras", "openai", "groq", "fireworks", "together"}:
        # OpenAI-like:
        # resp.choices[0].message.content -> str
        try:
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            return ""

    if provider == "anthropic":
        # Anthropic:
        # resp.content -> list of blocks (text blocks contain block.text)
        chunks: List[str] = []
        for block in getattr(resp, "content", []) or []:
            if getattr(block, "type", None) == "text":
                chunks.append(getattr(block, "text", "") or "")
            elif isinstance(block, dict) and block.get("type") == "text":
                chunks.append(block.get("text", "") or "")
        return "".join(chunks).strip()

    if provider == "bedrock":
        # Bedrock Converse:
        # resp["output"]["message"]["content"] -> list of blocks
        # Each block may have "text"
        try:
            msg = (resp.get("output") or {}).get("message") or {}
            content = msg.get("content") or []
            texts = [b.get("text", "") for b in content if isinstance(b, dict) and "text" in b]
            return "".join(texts).strip()
        except Exception:
            return ""

    return ""


# -----------------------------
# CLI
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Chat completions example for multiple providers (with response-format notes).")
    parser.add_argument(
        "--provider",
        required=True,
        choices=["cerebras", "openai", "anthropic", "bedrock", "groq", "fireworks", "together"],
        help="Provider to call",
    )
    parser.add_argument("--model", required=True, help="Model name/id for the provider")
    parser.add_argument(
        "--prompt",
        default="Why is fast inference important?",
        help="User prompt",
    )
    parser.add_argument(
        "--region",
        default=None,
        help="AWS region (Bedrock only). If omitted uses AWS_REGION/AWS_DEFAULT_REGION.",
    )
    parser.add_argument(
        "--print-raw",
        action="store_true",
        help="Print the raw SDK response object/dict (useful for debugging response formats).",
    )

    args = parser.parse_args()
    provider = args.provider.lower()

    # Execute provider call
    if provider == "cerebras":
        resp = chat_cerebras(args.model, args.prompt)
    elif provider == "openai":
        resp = chat_openai(args.model, args.prompt)
    elif provider == "anthropic":
        resp = chat_anthropic(args.model, args.prompt)
    elif provider == "bedrock":
        resp = chat_bedrock(args.model, args.prompt, args.region)
    elif provider == "groq":
        resp = chat_groq(args.model, args.prompt)
    elif provider == "fireworks":
        resp = chat_fireworks(args.model, args.prompt)
    elif provider == "together":
        resp = chat_together(args.model, args.prompt)
    else:
        raise SystemExit(f"Unsupported provider: {provider}")

    # Provider-normalized text output
    text = extract_text(provider, resp)
    print("\n=== EXTRACTED_TEXT ===")
    print(text if text else "(empty)")

    # Optional: print raw response shape
    if args.print_raw:
        print("\n=== RAW_RESPONSE ===")
        print(_safe_json(resp))


if __name__ == "__main__":
    main()
