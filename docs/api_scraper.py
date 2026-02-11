#!/usr/bin/env python3
"""
tools/gen_openapi_md_bulk.py

Generate canonical OpenAPI 3.1 Markdown pages in a fixed schema layout.

Reads:  docs/<provider>/sources.json
Writes: docs/<provider>/<slug>.md

Run:
  uv run python tools/gen_openapi_md_bulk.py --docs-dir docs
  uv run python tools/gen_openapi_md_bulk.py --docs-dir docs --only cerebras,openai

Deps: stdlib only (no scraping; avoids 403 issues)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def canonical_base(info_title: str, description: str, server_url: str) -> Dict[str, Any]:
    return {
        "openapi": "3.1.0",
        "info": {
            "title": info_title,
            "description": description,
        },
        "servers": [{"url": server_url}],
        "paths": {},
        "components": {
            "securitySchemes": {
                "bearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "API key",
                }
            },
            "schemas": {},
        },
    }


def openai_compatible_chat_template(
    path: str,
    summary: str,
    operation_id: str,
    method: str = "post",
    include_queue_threshold: bool = False,
    include_version_patch_header: bool = False,
) -> Dict[str, Any]:
    params = []
    if include_queue_threshold:
        params.append(
            {
                "name": "queue_threshold",
                "in": "header",
                "required": False,
                "description": (
                    "Controls the queue time threshold for requests using the flex or auto service tiers. "
                    "Valid range: 50-20000 (milliseconds). Private Preview."
                ),
                "schema": {"type": "string"},
            }
        )

    # Keep this optional so you can turn it on only for Cerebras
    if include_version_patch_header:
        params.append(
            {
                "name": "X-Cerebras-Version-Patch",
                "in": "header",
                "required": False,
                "description": "Optional API version override header (e.g., \"2\").",
                "schema": {"type": "string", "pattern": "^[0-9]+$", "examples": ["2"]},
            }
        )

    op: Dict[str, Any] = {
        "summary": summary,
        "operationId": operation_id,
        "security": [{"bearerAuth": []}],
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/ChatCompletionRequest"}
                }
            },
        },
        "responses": {
            "200": {
                "description": "Successful response (non-streaming).",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/ChatCompletionResponse"}
                    }
                },
            },
            "400": {
                "description": "Bad Request (validation error).",
                "content": {
                    "application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}
                },
            },
            "401": {
                "description": "Unauthorized.",
                "content": {
                    "application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}
                },
            },
            "429": {
                "description": "Rate limit / capacity / queue threshold rejection.",
                "content": {
                    "application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}
                },
            },
            "500": {
                "description": "Server error.",
                "content": {
                    "application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}
                },
            },
        },
    }

    if params:
        op["parameters"] = params

    return {path: {(method or "post").lower(): op}}


def openai_compatible_schemas() -> Dict[str, Any]:
    return {
        "ChatCompletionRequest": {
            "type": "object",
            "required": ["model", "messages"],
            "properties": {
                "model": {
                    "type": "string",
                    "description": "Model ID. May be a public model name or an org endpoint id.",
                    "examples": ["gpt-oss-120b", "llama-3.3-70b", "my-org-llama-3.3-70b"],
                },
                "messages": {
                    "type": "array",
                    "minItems": 1,
                    "description": "A list of messages comprising the conversation so far.",
                    "items": {"$ref": "#/components/schemas/ChatMessage"},
                },
                "logprobs": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether to return log probabilities of the output tokens.",
                },
                "top_logprobs": {
                    "type": ["integer", "null"],
                    "minimum": 0,
                    "maximum": 20,
                    "description": "Number of most likely tokens to return per position. Requires logprobs=true.",
                },
                "max_completion_tokens": {
                    "type": ["integer", "null"],
                    "minimum": 0,
                    "description": "Maximum number of tokens generated in the completion.",
                },
                "stream": {
                    "type": ["boolean", "null"],
                    "default": False,
                    "description": "If true, partial message deltas will be sent as server-sent events.",
                },
                "temperature": {
                    "type": ["number", "null"],
                    "minimum": 0,
                    "maximum": 2,
                    "description": "Sampling temperature.",
                },
                "top_p": {
                    "type": ["number", "null"],
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Nucleus sampling parameter.",
                },
                "tools": {
                    "type": ["array", "null"],
                    "description": "A list of tools the model may call. Currently, only functions are supported.",
                    "items": {"$ref": "#/components/schemas/ToolDefinition"},
                },
                "tool_choice": {
                    "description": "Controls which (if any) tool is called by the model.",
                    "oneOf": [
                        {"type": "string", "enum": ["none", "auto", "required"]},
                        {
                            "type": "object",
                            "required": ["type", "function"],
                            "properties": {
                                "type": {"type": "string", "const": "function"},
                                "function": {
                                    "type": "object",
                                    "required": ["name"],
                                    "properties": {"name": {"type": "string"}},
                                    "additionalProperties": False,
                                },
                            },
                            "additionalProperties": False,
                        },
                    ],
                },
                "user": {
                    "type": ["string", "null"],
                    "description": "Unique identifier representing your end-user.",
                },
            },
            "additionalProperties": True,
        },
        "ChatMessage": {
            "type": "object",
            "required": ["role", "content"],
            "properties": {
                "role": {"type": "string", "enum": ["system", "user", "assistant"]},
                "content": {"type": "string", "description": "Message text content."},
            },
            "additionalProperties": False,
        },
        "ToolDefinition": {
            "type": "object",
            "required": ["type", "function"],
            "properties": {
                "type": {"type": "string", "const": "function"},
                "function": {
                    "type": "object",
                    "required": ["name"],
                    "properties": {
                        "name": {
                            "type": "string",
                            "maxLength": 64,
                            "pattern": "^[A-Za-z0-9_-]+$",
                            "description": "Function name to be called.",
                        },
                        "description": {"type": "string"},
                        "parameters": {
                            "type": "object",
                            "description": "JSON Schema object describing tool arguments.",
                        },
                    },
                    "additionalProperties": True,
                },
            },
            "additionalProperties": False,
        },
        "ChatCompletionResponse": {
            "type": "object",
            "required": ["id", "choices", "created", "model", "object"],
            "properties": {
                "id": {"type": "string", "description": "Unique identifier for the chat completion."},
                "object": {"type": "string"},
                "created": {"type": "integer", "description": "Unix timestamp (seconds)."},
                "model": {"type": "string"},
                "choices": {"type": "array", "minItems": 1, "items": {"type": "object"}},
                "usage": {"type": "object"},
            },
            "additionalProperties": True,
        },
        "ErrorResponse": {
            "type": "object",
            "description": "Generic error response schema (shape may vary).",
            "properties": {
                "error": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"},
                        "type": {"type": "string"},
                        "code": {"type": ["string", "null"]},
                        "param": {"type": ["string", "null"]},
                    },
                    "additionalProperties": True,
                }
            },
            "additionalProperties": True,
        },
    }


# ----------------------------
# Anthropic template
# ----------------------------

def anthropic_messages_template(path: str, summary: str, operation_id: str, method: str = "post") -> Dict[str, Any]:
    return {
        path: {
            (method or "post").lower(): {
                "summary": summary,
                "operationId": operation_id,
                "security": [{"bearerAuth": []}],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/MessageCreateRequest"}
                        }
                    },
                },
                "responses": {
                    "200": {
                        "description": "Successful response (non-streaming).",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/MessageResponse"}
                            }
                        },
                    },
                    "400": {
                        "description": "Bad Request (validation error).",
                        "content": {
                            "application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}
                        },
                    },
                    "401": {
                        "description": "Unauthorized.",
                        "content": {
                            "application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}
                        },
                    },
                    "429": {
                        "description": "Rate limit / capacity rejection.",
                        "content": {
                            "application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}
                        },
                    },
                    "500": {
                        "description": "Server error.",
                        "content": {
                            "application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}
                        },
                    },
                },
            }
        }
    }


def anthropic_schemas() -> Dict[str, Any]:
    return {
        "MessageCreateRequest": {
            "type": "object",
            "required": ["model", "messages", "max_tokens"],
            "properties": {
                "model": {"type": "string"},
                "system": {"type": ["string", "null"], "description": "System prompt (top-level)."},
                "messages": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"$ref": "#/components/schemas/AnthropicMessage"},
                },
                "max_tokens": {"type": "integer"},
                "tools": {"type": ["array", "null"], "items": {"$ref": "#/components/schemas/AnthropicTool"}},
            },
            "additionalProperties": True,
        },
        "AnthropicMessage": {
            "type": "object",
            "required": ["role", "content"],
            "properties": {
                "role": {"type": "string", "enum": ["user", "assistant"]},
                "content": {"type": "array", "items": {"$ref": "#/components/schemas/ContentBlock"}},
            },
            "additionalProperties": True,
        },
        "ContentBlock": {
            "oneOf": [
                {
                    "type": "object",
                    "required": ["type", "text"],
                    "properties": {"type": {"const": "text"}, "text": {"type": "string"}},
                    "additionalProperties": True,
                },
                {
                    "type": "object",
                    "required": ["type", "thinking"],
                    "properties": {"type": {"const": "thinking"}, "thinking": {"type": "string"}},
                    "additionalProperties": True,
                },
                {
                    "type": "object",
                    "required": ["type", "name", "input"],
                    "properties": {"type": {"const": "tool_use"}, "name": {"type": "string"}, "input": {"type": "object"}},
                    "additionalProperties": True,
                },
            ]
        },
        "AnthropicTool": {
            "type": "object",
            "required": ["name", "input_schema"],
            "properties": {
                "name": {"type": "string"},
                "description": {"type": ["string", "null"]},
                "input_schema": {"type": "object"},
            },
            "additionalProperties": True,
        },
        "MessageResponse": {"type": "object", "additionalProperties": True},
        "ErrorResponse": {"type": "object", "additionalProperties": True},
    }


def render_md(info_title: str, method: str, path: str, spec: Dict[str, Any]) -> str:
    return (
        f"# {info_title} (OpenAPI 3.1)\n\n"
        f"This page contains the OpenAPI 3.1 specification for `{method.upper()} {path}`.\n\n"
        "## OpenAPI JSON\n\n"
        "```json\n"
        f"{json.dumps(spec, indent=2, ensure_ascii=False)}\n"
        "```\n"
    )


def read_sources_json(p: Path) -> List[Dict[str, Any]]:
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs-dir", default="docs", help="docs root (feature_catalog/docs)")
    ap.add_argument("--only", default=None, help="Comma-separated provider folders to run (e.g. openai,anthropic)")
    args = ap.parse_args()

    docs_dir = Path(args.docs_dir)
    only = {x.strip().lower() for x in args.only.split(",")} if args.only else None

    for provider_dir in sorted([p for p in docs_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
        provider = provider_dir.name.lower()
        if only and provider not in only:
            continue

        src = provider_dir / "sources.json"
        if not src.exists():
            continue

        entries = read_sources_json(src)
        for e in entries:
            slug = e["slug"]
            doc_url = e.get("doc_url", "")
            method = e.get("method", "post").lower()

            path = e["path"]  # "/chat/completions" or "/completions" or "/messages"
            server_url = e["server_url"]

            summary = e.get("summary", "Create")
            operation_id = e.get("operationId", slug)

            include_queue_threshold = bool(e.get("include_queue_threshold", False))
            include_version_patch_header = bool(e.get("include_version_patch_header", False))

            if provider == "anthropic":
                info_title = e.get("info_title", "Anthropic Inference API - Messages")
                spec = canonical_base(
                    info_title=info_title,
                    description=f"Machine-verifiable OpenAPI 3.1 contract for {method.upper()} {path}.",
                    server_url=server_url,
                )
                spec["paths"] = anthropic_messages_template(path, summary, operation_id, method=method)
                spec["components"]["schemas"] = anthropic_schemas()
            else:
                # Treat everything else as OpenAI-compatible by default
                info_title = e.get("info_title", f"{provider.capitalize()} Inference API - Chat Completions")
                spec = canonical_base(
                    info_title=info_title,
                    description=f"Machine-verifiable OpenAPI 3.1 contract for {method.upper()} {path}.",
                    server_url=server_url,
                )
                spec["paths"] = openai_compatible_chat_template(
                    path,
                    summary,
                    operation_id,
                    method=method,
                    include_queue_threshold=include_queue_threshold,
                    include_version_patch_header=include_version_patch_header,
                )
                spec["components"]["schemas"] = openai_compatible_schemas()

            out_md = provider_dir / f"{slug}.md"
            out_md.write_text(render_md(spec["info"]["title"], method, path, spec), encoding="utf-8")
            print(f"Wrote {out_md} (from {doc_url})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
