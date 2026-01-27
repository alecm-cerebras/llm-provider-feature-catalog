"""
Prompt Caching Demo (Multi-Provider)

This script demonstrates *explicit / provider-native* prompt caching where available, and
prints *useful cache signals* (cache hits/reads/creates) when the provider returns them.

Providers implemented here
--------------------------
- openai      (Responses API prompt caching; server-managed; usage includes cache read/create tokens)
- anthropic   (Messages prompt caching; cache_control={"type":"ephemeral"} on prefix blocks)
- groq        (OpenAI-compatible; supports prompt caching; usage may include cache_* token fields)
- fireworks   (OpenAI-compatible; prompt caching supported; usage may include cache_* token fields)
- bedrock     (Bedrock prompt caching; implementation depends on AWS feature + model support)
- cerebras    (Cerebras prompt caching; see Cerebras docs for request shape)

Why two requests?
-----------------
We send two requests in a row:
1) Large stable prefix + Question A   -> expected cache CREATE (or MISS)
2) Same prefix + Question B          -> expected cache READ (or HIT)

Docs (primary)
--------------
- OpenAI:     https://platform.openai.com/docs/guides/prompt-caching
- Anthropic:  https://platform.claude.com/docs/en/build-with-claude/prompt-caching
- Groq:       https://console.groq.com/docs/prompt-caching
- Bedrock:    https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html
- Fireworks:  https://docs.fireworks.ai/guides/prompt-caching
- Cerebras:   https://inference-docs.cerebras.ai/capabilities/prompt-caching

OpenAI:
  uv run --env-file .env python examples/python/prompt_caching.py --provider openai --model gpt-4o-mini --prefix-bytes 50000

Anthropic:
  uv run --env-file .env python examples/python/prompt_caching.py --provider anthropic --model claude-3-5-sonnet-latest --prefix-bytes 50000

Groq:
  uv run --env-file .env python examples/python/prompt_caching.py --provider groq --model llama-3.1-70b-versatile --prefix-bytes 50000

Fireworks:
  uv run --env-file .env python examples/python/prompt_caching.py --provider fireworks --model accounts/fireworks/models/llama-v3p1-70b-instruct --prefix-bytes 50000

Bedrock:
  uv run --env-file .env python examples/python/prompt_caching.py --provider bedrock --model amazon.nova-lite-v1:0 --region us-east-1 --prefix-bytes 50000

Cerebras:
  uv run --env-file .env python examples/python/prompt_caching.py --provider cerebras --model zai-glm-4.7 --cache-id demo_cache_1 --prefix-bytes 50000 --debug

Notes / Caveats
---------------
- Prompt caching is not universally available for every model/tier.
- Cache controls and returned metrics differ by provider.
- This demo is designed to be informative even when a provider caches silently (no explicit fields).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple


# Anthropic: beta flag names can change. If your SDK errors, update/remove betas=[...].
ANTHROPIC_PROMPT_CACHING_BETA = "prompt-caching-2024-07-31"


# -----------------------------
# Utilities
# -----------------------------

def _require_env(var_name: str) -> str:
    value = os.environ.get(var_name)
    if not value:
        raise SystemExit(f"Missing required environment variable: {var_name}")
    return value


def _as_dict(resp: Any) -> dict:
    if hasattr(resp, "model_dump"):
        return resp.model_dump()
    if hasattr(resp, "dict"):
        return resp.dict()
    if isinstance(resp, dict):
        return resp
    return {"__raw__": str(resp)}


def _maybe_get(d: Any, path: str) -> Any:
    cur = d
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _print_provider_help(provider: str) -> None:
    if provider == "openai":
        print("Provider note (OpenAI): caching is automatic for eligible prompts. See:")
        print("  https://platform.openai.com/docs/guides/prompt-caching")
        print("Look for usage.input_tokens_details.cached_tokens (cached tokens on cache hits).\n")
    elif provider == "anthropic":
        print("Provider note (Anthropic): mark cacheable prefix blocks with cache_control=ephemeral. See:")
        print("  https://platform.claude.com/docs/en/build-with-claude/prompt-caching\n")
    elif provider == "groq":
        print("Provider note (Groq): prompt caching supported for eligible prompts. See:")
        print("  https://console.groq.com/docs/prompt-caching\n")
    elif provider == "fireworks":
        print("Provider note (Fireworks): prompt caching supported. See:")
        print("  https://docs.fireworks.ai/guides/prompt-caching\n")
    elif provider == "bedrock":
        print("Provider note (Bedrock): caching is model/region/feature dependent. See:")
        print("  https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html\n")
    elif provider == "cerebras":
        print("Provider note (Cerebras): prompt caching request shape is Cerebras-specific. See:")
        print("  https://inference-docs.cerebras.ai/capabilities/prompt-caching\n")


def _build_prefix(prefix_bytes: int) -> str:
    # Deterministic stable prefix to maximize cache reuse across request 1 & 2
    filler = ("Context: This is stable reference text. " * 5000).encode("utf-8")
    return (filler[:prefix_bytes]).decode("utf-8", errors="ignore")


def _extract_text(provider: str, resp: Any) -> str:
    if provider in ("openai", "groq", "fireworks", "cerebras"):
        # chat.completions: resp.choices[0].message.content
        try:
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            pass

        # responses API: resp.output_text
        try:
            return (resp.output_text or "").strip()
        except Exception:
            pass

    if provider == "anthropic":
        chunks: List[str] = []
        for block in getattr(resp, "content", []) or []:
            if getattr(block, "type", None) == "text":
                chunks.append(getattr(block, "text", "") or "")
        return "".join(chunks).strip()

    if provider == "bedrock":
        data = _as_dict(resp)
        msg = (data.get("output") or {}).get("message") or {}
        content = msg.get("content") or []
        texts = [b.get("text", "") for b in content if isinstance(b, dict) and "text" in b]
        return "".join(texts).strip()

    return str(resp)


def _print_usage_and_cache(provider: str, resp: Any) -> None:
    data = _as_dict(resp)

    usage = data.get("usage") if isinstance(data, dict) else None
    if usage:
        print("\nusage:")
        print(json.dumps(usage, indent=2, sort_keys=True))

    # Best-effort: common-ish cache signals
    candidates = {
        # OpenAI Responses API commonly reports cached tokens here:
        "cached_tokens": _maybe_get(data, "usage.input_tokens_details.cached_tokens"),
        # Some docs/SDKs expose these (keep for compatibility):
        "cache_read_input_tokens": _maybe_get(data, "usage.cache_read_input_tokens"),
        "cache_creation_input_tokens": _maybe_get(data, "usage.cache_creation_input_tokens"),
        # Other providers might use these:
        "cached_prompt_tokens": _maybe_get(data, "usage.cached_prompt_tokens"),
        "uncached_prompt_tokens": _maybe_get(data, "usage.uncached_prompt_tokens"),
        "prompt_cache": data.get("prompt_cache") if isinstance(data, dict) else None,
        "prompt_caching": data.get("prompt_caching") if isinstance(data, dict) else None,
        "cache": data.get("cache") if isinstance(data, dict) else None,
        "cache_info": data.get("cache_info") if isinstance(data, dict) else None,
        "time_info": data.get("time_info") if isinstance(data, dict) else None,
    }
    candidates = {k: v for k, v in candidates.items() if v is not None}

    if candidates:
        print("\ncache signals (best-effort):")
        for k, v in candidates.items():
            if isinstance(v, (dict, list)):
                print(f"  {k}: {json.dumps(v, indent=2, sort_keys=True)}")
            else:
                print(f"  {k}: {v}")
    else:
        print("\n(no explicit cache fields found; provider may still cache implicitly)")


# -----------------------------
# Provider calls
# -----------------------------

def call_openai(*, model: str, prefix: str, question: str) -> Any:
    """
    OpenAI: use the Responses API.

    Prompt caching is automatic for eligible prompts. Some accounts/SDK versions do not
    support an explicit cache_control parameter in the request. Detect caching via:
      usage.input_tokens_details.cached_tokens
    """
    # pip install openai
    from openai import OpenAI

    client = OpenAI(api_key=_require_env("OPENAI_API_KEY"))

    return client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prefix},
                    {"type": "input_text", "text": f"\n\nQuestion: {question}\n"},
                ],
            }
        ],
    )


def call_openai_compatible_chat_completions(
    *, base_url: str, api_key_env: str, model: str, prefix: str, question: str
) -> Any:
    """
    Groq / Fireworks: OpenAI-compatible Chat Completions endpoint.
    Prompt caching is provider-specific; if enabled for your account/model, it usually shows up in usage.
    """
    # pip install openai
    from openai import OpenAI

    client = OpenAI(api_key=_require_env(api_key_env), base_url=base_url)
    return client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": f"{prefix}\n\nQuestion: {question}"},
        ],
    )


def call_anthropic(*, model: str, prefix: str, question: str) -> Any:
    """
    Anthropic: mark the stable prefix as cacheable with cache_control={"type":"ephemeral"}.
    """
    # pip install anthropic
    import anthropic

    client = anthropic.Anthropic(api_key=_require_env("ANTHROPIC_API_KEY"))
    return client.messages.create(
        model=model,
        max_tokens=512,
        betas=[ANTHROPIC_PROMPT_CACHING_BETA],
        system="You are a concise assistant.",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prefix, "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": f"\n\nQuestion: {question}\n"},
                ],
            }
        ],
    )


def call_bedrock(*, model: str, prefix: str, question: str, region: Optional[str]) -> Any:
    """
    AWS Bedrock: prompt caching is model/region/feature dependent and the API surface can differ.
    This demo uses Converse. If your account/model supports caching, inspect usage/metrics fields.

    Auth: standard AWS credential chain; and AWS_REGION/AWS_DEFAULT_REGION or --region.
    """
    # pip install boto3
    import boto3

    region_name = region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if not region_name:
        raise SystemExit("Bedrock requires --region or AWS_REGION/AWS_DEFAULT_REGION")

    client = boto3.client("bedrock-runtime", region_name=region_name)
    prompt = f"{prefix}\n\nQuestion: {question}"

    return client.converse(
        modelId=model,
        system=[{"text": "You are a concise assistant."}],
        messages=[{"role": "user", "content": [{"text": prompt}]}],
    )


def call_cerebras(*, model: str, prefix: str, question: str, cache_id: Optional[str]) -> Any:
    """
    Cerebras: request shape for caching is Cerebras-specific.
    This keeps the caching fields isolated here for easy adjustment.

    Requires: CEREBRAS_API_KEY
    """
    from cerebras.cloud.sdk import Cerebras

    client = Cerebras(api_key=_require_env("CEREBRAS_API_KEY"))

    messages = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": f"{prefix}\n\nQuestion: {question}"},
    ]

    kwargs: Dict[str, Any] = {}
    if cache_id:
        # NOTE: Update these keys if the Cerebras doc/spec requires different names.
        kwargs["prompt_cache"] = {"id": cache_id, "prefix": prefix}

    return client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs,
    )


# -----------------------------
# CLI
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt caching demo (multi-provider).")
    parser.add_argument(
        "--provider",
        required=True,
        choices=["openai", "anthropic", "groq", "fireworks", "bedrock", "cerebras"],
        help="Provider to call",
    )
    parser.add_argument("--model", required=True, help="Model name/id for the provider")
    parser.add_argument(
        "--prefix-bytes",
        type=int,
        default=12000,
        help="Approx size of stable prefix (in bytes). Larger makes cache effects easier to see.",
    )
    parser.add_argument(
        "--cache-id",
        default="demo_cache_1",
        help="Cerebras-only stable cache identifier (reused across calls).",
    )
    parser.add_argument(
        "--region",
        default=None,
        help="AWS region (Bedrock only). If omitted uses AWS_REGION/AWS_DEFAULT_REGION.",
    )
    parser.add_argument("--debug", action="store_true", help="Print full raw responses")
    args = parser.parse_args()

    provider = args.provider.lower()
    _print_provider_help(provider)

    prefix = _build_prefix(args.prefix_bytes)

    questions = [
        "Summarize the key points from the context in 2 bullets.",
        "List 3 risks if this context is out of date.",
    ]

    for i, q in enumerate(questions, start=1):
        t0 = time.perf_counter()

        if provider == "openai":
            resp = call_openai(model=args.model, prefix=prefix, question=q)
        elif provider == "anthropic":
            resp = call_anthropic(model=args.model, prefix=prefix, question=q)
        elif provider == "groq":
            resp = call_openai_compatible_chat_completions(
                base_url="https://api.groq.com/openai/v1",
                api_key_env="GROQ_API_KEY",
                model=args.model,
                prefix=prefix,
                question=q,
            )
        elif provider == "fireworks":
            resp = call_openai_compatible_chat_completions(
                base_url="https://api.fireworks.ai/inference/v1",
                api_key_env="FIREWORKS_API_KEY",
                model=args.model,
                prefix=prefix,
                question=q,
            )
        elif provider == "bedrock":
            resp = call_bedrock(model=args.model, prefix=prefix, question=q, region=args.region)
        elif provider == "cerebras":
            resp = call_cerebras(model=args.model, prefix=prefix, question=q, cache_id=args.cache_id)
        else:
            raise SystemExit(f"Unsupported provider: {provider}")

        dt = time.perf_counter() - t0
        text = _extract_text(provider, resp)

        print(f"\n--- Request {i} ({provider}) ---")
        print(f"Latency: {dt:.3f}s")
        print(f"Question: {q}")
        print("\nAssistant:")
        print(text)

        _print_usage_and_cache(provider, resp)

        if args.debug:
            print("\nraw response:")
            print(json.dumps(_as_dict(resp), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()