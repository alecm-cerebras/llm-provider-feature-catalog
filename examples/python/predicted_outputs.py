"""
Predicted Outputs (aka "prediction") demo — multi-provider (minimal)

This script is intentionally small and focused on:
- how to CALL predicted outputs (when supported), and
- where to READ the returned text content.

Provider call differences
-------------------------
OpenAI (native)
- Use Chat Completions with `prediction=...` (matches the OpenAI example you provided)
    resp = client.chat.completions.create(..., prediction={"type":"content","content": ...})
- Read content:
    print(resp)
    print(resp.choices[0].message.content)

OpenAI-compatible Chat Completions (best-effort)
- Used by: Cerebras / Groq / Fireworks (OpenAI-compatible surfaces)
- Call: some servers may accept a `prediction` field, others may ignore/reject it
    resp = client.chat.completions.create(..., prediction={"type":"content","content": ...})
- Read content:
    print(resp)
    print(resp.choices[0].message.content)
- Only Cerebras support predicted outputs as of Feb 2026

Anthropic / Together
- No portable "predicted outputs" parameter shown here -> baseline only
- Read content:
    Anthropic: concatenate resp.content[*].text
    Together:  resp.choices[0].message.content

Run
---
uv run --env-file .env python examples/python/predicted_outputs.py --provider cerebras --model gpt-oss-120b --mode predicted
uv run --env-file .env python examples/python/predicted_outputs.py --provider openai --model gpt-4.1-mini --mode predicted
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Literal, Optional

Mode = Literal["baseline", "predicted", "both"]


# -----------------------------
# Example task (use this everywhere)
# -----------------------------

code = """
html {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
    scroll-behavior: smooth;
    font-size: 16px;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}
body {
    font-family: Georgia, serif;
    font-size: 14px;
    line-height: 1.8;
    background: #000000;
    margin: 0;
    padding: 0;
    color: #00FF00;
}
"""

instructions = "Change the color to blue. Respond only with code. Don't add comments."


def _require_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise SystemExit(f"Missing required environment variable: {name}")
    return v


# -----------------------------
# Provider callers
# -----------------------------

def call_openai_chat(*, api_key: str, model: str, prediction: Optional[str]) -> Any:
    """
    OpenAI (native): chat.completions + prediction (matches user's example).

    resp = client.chat.completions.create(..., prediction={"type":"content","content": code})
    """
    from openai import OpenAI  # pip install openai

    client = OpenAI(api_key=api_key)

    kwargs: dict[str, Any] = {}
    if prediction is not None:
        kwargs["prediction"] = {"type": "content", "content": prediction}

    return client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": instructions},
            {"role": "user", "content": code},
        ],
        **kwargs,
        # stream=True,  # Uncomment to enable streaming
    )


def call_openai_compatible_chat(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prediction: Optional[str],
) -> Any:
    """
    OpenAI-compatible chat.completions + best-effort prediction.

    Same call shape as OpenAI:
      resp = client.chat.completions.create(..., prediction={...})

    Caveat:
    - `prediction` is not standardized across all OpenAI-compatible servers.
      Some will ignore it; some may reject it.
    """
    from openai import OpenAI  # pip install openai

    client = OpenAI(api_key=api_key, base_url=base_url)

    kwargs: dict[str, Any] = {}
    if prediction is not None:
        kwargs["prediction"] = {"type": "content", "content": prediction}

    return client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": instructions},
            {"role": "user", "content": code},
        ],
        **kwargs,
        # stream=True,  # Uncomment to enable streaming
    )


def call_anthropic_baseline(*, api_key: str, model: str) -> Any:
    """
    Anthropic baseline only (no predicted outputs parameter in this demo).
    """
    from anthropic import Anthropic  # pip install anthropic

    client = Anthropic(api_key=api_key)
    return client.messages.create(
        model=model,
        max_tokens=400,
        system="Respond only with code. Don't add comments.",
        messages=[
            {"role": "user", "content": instructions},
            {"role": "user", "content": code},
        ],
    )


def call_together_baseline(*, api_key: str, model: str) -> Any:
    """
    Together baseline only (no predicted outputs parameter in this demo).
    """
    import together  # pip install together

    client = together.Together(api_key=api_key)
    return client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": instructions},
            {"role": "user", "content": code},
        ],
    )


# -----------------------------
# Print helpers (show full response + content)
# -----------------------------

def print_openai_compatible_chat(resp: Any) -> None:
    # Works for: OpenAI, Cerebras, Groq, Fireworks (chat.completions shape)
    print(resp)
    print(resp.choices[0].message.content)


def print_anthropic(resp: Any) -> None:
    print(resp)
    chunks: list[str] = []
    for block in getattr(resp, "content", []) or []:
        if getattr(block, "type", None) == "text":
            chunks.append(getattr(block, "text", "") or "")
    print("".join(chunks))


def print_together(resp: Any) -> None:
    print(resp)
    print(resp.choices[0].message.content)


# -----------------------------
# Main
# -----------------------------

def run(provider: str, model: str, mode: Literal["baseline", "predicted"]) -> None:
    prediction = code if mode == "predicted" else None

    if provider == "openai":
        resp = call_openai_chat(
            api_key=_require_env("OPENAI_API_KEY"),
            model=model,
            prediction=prediction,
        )
        print_openai_compatible_chat(resp)
        return

    if provider == "cerebras":
        resp = call_openai_compatible_chat(
            base_url="https://api.cerebras.ai/v1",
            api_key=_require_env("CEREBRAS_API_KEY"),
            model=model,
            prediction=prediction,
        )
        print_openai_compatible_chat(resp)
        return

    if provider == "groq":
        resp = call_openai_compatible_chat(
            base_url="https://api.groq.com/openai/v1",
            api_key=_require_env("GROQ_API_KEY"),
            model=model,
            prediction=prediction,
        )
        print_openai_compatible_chat(resp)
        return

    if provider == "fireworks":
        resp = call_openai_compatible_chat(
            base_url="https://api.fireworks.ai/inference/v1",
            api_key=_require_env("FIREWORKS_API_KEY"),
            model=model,
            prediction=prediction,
        )
        print_openai_compatible_chat(resp)
        return

    if provider == "anthropic":
        if mode == "predicted":
            raise SystemExit("anthropic: predicted outputs not shown in this demo; use --mode baseline")
        resp = call_anthropic_baseline(
            api_key=_require_env("ANTHROPIC_API_KEY"),
            model=model,
        )
        print_anthropic(resp)
        return

    if provider == "together":
        if mode == "predicted":
            raise SystemExit("together: predicted outputs not shown in this demo; use --mode baseline")
        resp = call_together_baseline(
            api_key=_require_env("TOGETHER_API_KEY"),
            model=model,
        )
        print_together(resp)
        return

    raise SystemExit(f"Unsupported provider: {provider}")


def main() -> None:
    p = argparse.ArgumentParser(description="Predicted outputs demo (prints full response + extracted content).")
    p.add_argument(
        "--provider",
        required=True,
        choices=["openai", "cerebras", "groq", "fireworks", "anthropic", "together"],
    )
    p.add_argument("--model", required=True)
    p.add_argument("--mode", default="both", choices=["baseline", "predicted", "both"])
    args = p.parse_args()

    if args.mode in ("baseline", "both"):
        run(args.provider, args.model, "baseline")

    if args.mode in ("predicted", "both"):
        run(args.provider, args.model, "predicted")


if __name__ == "__main__":
    main()