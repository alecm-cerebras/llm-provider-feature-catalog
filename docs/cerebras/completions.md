# Cerebras Inference API — Completions (OpenAPI 3.1)

This page contains the OpenAPI 3.1 specification for `POST /v1/completions`.

## OpenAPI JSON

```json
{
  "openapi": "3.1.0",
  "info": {
    "title": "Cerebras Inference API - Completions",
    "version": "2026-01",
    "description": "Machine-verifiable OpenAPI 3.1 contract for POST /v1/completions."
  },
  "servers": [
    {
      "url": "https://api.cerebras.ai/v1"
    }
  ],
  "paths": {
    "/completions": {
      "post": {
        "summary": "Completions",
        "operationId": "createCompletion",
        "security": [
          {
            "bearerAuth": []
          }
        ],
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/CompletionRequest"
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Successful response (non-streaming).",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/CompletionResponse"
                }
              }
            }
          },
          "400": {
            "description": "Bad Request (validation error).",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ErrorResponse"
                }
              }
            }
          },
          "401": {
            "description": "Unauthorized.",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ErrorResponse"
                }
              }
            }
          },
          "429": {
            "description": "Rate limit / capacity / queue rejection.",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ErrorResponse"
                }
              }
            }
          },
          "500": {
            "description": "Server error.",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ErrorResponse"
                }
              }
            }
          }
        }
      }
    }
  },
  "components": {
    "securitySchemes": {
      "bearerAuth": {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "API key"
      }
    },
    "schemas": {
      "CompletionRequest": {
        "type": "object",
        "required": ["model"],
        "properties": {
          "model": {
            "type": "string",
            "description": "Model ID.",
            "examples": ["llama3.1-8b", "qwen-3-235b-a22b-instruct-2507"]
          },
          "prompt": {
            "description": "The prompt(s) to generate completions for. Can be a string, array of strings, array of tokens, or array of token arrays.",
            "default": "",
            "oneOf": [
              { "type": "string" },
              { "type": "array", "items": { "type": "string" } },
              { "type": "array", "items": { "type": "integer" } },
              { "type": "array", "items": { "type": "array", "items": { "type": "integer" } } }
            ]
          },
          "stream": {
            "type": ["boolean", "null"],
            "default": false,
            "description": "If true, partial deltas are sent as data-only server-sent events (SSE), terminated by `data: [DONE]`."
          },
          "return_raw_tokens": {
            "type": ["boolean", "null"],
            "default": false,
            "description": "Return raw tokens instead of text."
          },
          "max_tokens": {
            "type": ["integer", "null"],
            "minimum": 0,
            "description": "Maximum number of tokens to generate. Input+output limited by model context length."
          },
          "min_tokens": {
            "type": ["integer", "null"],
            "description": "Minimum number of tokens to generate. If -1, sets to max sequence length."
          },
          "grammar_root": {
            "type": ["string", "null"],
            "description": "Grammar root used for structured output generation.",
            "enum": [
              "root",
              "fcall",
              "nofcall",
              "insidevalue",
              "value",
              "object",
              "array",
              "string",
              "number",
              "funcarray",
              "func",
              "ws",
              null
            ]
          },
          "seed": {
            "type": ["integer", "null"],
            "description": "Best-effort deterministic sampling; determinism not guaranteed."
          },
          "stop": {
            "description": "Up to 4 sequences where generation stops. Returned text will not contain the stop sequence.",
            "oneOf": [
              { "type": "string" },
              { "type": "array", "items": { "type": "string" }, "maxItems": 4 },
              { "type": "null" }
            ]
          },
          "temperature": {
            "type": ["number", "null"],
            "minimum": 0,
            "maximum": 1.5,
            "default": 1.0,
            "description": "Sampling temperature."
          },
          "top_p": {
            "type": ["number", "null"],
            "minimum": 0,
            "maximum": 1,
            "default": 1.0,
            "description": "Nucleus sampling parameter."
          },
          "echo": {
            "type": "boolean",
            "default": false,
            "description": "Echo back the prompt in addition to the completion. Incompatible with return_raw_tokens=true."
          },
          "user": {
            "type": ["string", "null"],
            "description": "End-user identifier for abuse monitoring."
          },
          "logprobs": {
            "type": ["integer", "null"],
            "minimum": 0,
            "maximum": 20,
            "description": "If set, return log probabilities and top tokens (up to 20). null disables; 0 enables but omits top_logprobs."
          }
        },
        "additionalProperties": true
      },
      "CompletionResponse": {
        "type": "object",
        "required": ["id", "object", "created", "model", "system_fingerprint", "choices", "usage", "time_info"],
        "properties": {
          "id": { "type": "string", "description": "Unique identifier for the completion." },
          "object": { "type": "string", "const": "text_completion" },
          "created": { "type": "integer", "description": "Unix timestamp (seconds) when the completion was created." },
          "model": { "type": "string", "description": "Model used for completion." },
          "system_fingerprint": {
            "type": "string",
            "description": "Backend configuration fingerprint; useful with seed to understand determinism changes."
          },
          "choices": {
            "type": "array",
            "minItems": 1,
            "description": "List of completion choices.",
            "items": { "$ref": "#/components/schemas/CompletionChoice" }
          },
          "usage": { "$ref": "#/components/schemas/Usage" },
          "time_info": { "$ref": "#/components/schemas/TimeInfo" }
        },
        "additionalProperties": true
      },
      "CompletionChoice": {
        "type": "object",
        "required": ["finish_reason", "index", "text"],
        "properties": {
          "finish_reason": {
            "type": ["string", "null"],
            "description": "Why generation stopped.",
            "enum": ["stop", "length", "content_filter", null]
          },
          "index": { "type": "integer" },
          "text": { "type": "string" },
          "logprobs": {
            "type": ["object", "null"],
            "description": "Log probability details if requested.",
            "properties": {
              "text_offset": {
                "type": "array",
                "items": { "type": "integer" },
                "description": "Character offsets since the prompt."
              },
              "token_logprobs": {
                "type": "array",
                "items": { "type": "number" },
                "description": "Logprob per token."
              },
              "tokens": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Tokens."
              },
              "top_logprobs": {
                "type": "array",
                "items": {
                  "type": "object",
                  "additionalProperties": { "type": "number" }
                },
                "description": "Most likely tokens and their logprobs for each position."
              }
            },
            "additionalProperties": true
          }
        },
        "additionalProperties": true
      },
      "Usage": {
        "type": "object",
        "required": ["prompt_tokens", "completion_tokens", "total_tokens"],
        "properties": {
          "prompt_tokens": { "type": "integer" },
          "completion_tokens": { "type": "integer" },
          "total_tokens": { "type": "integer" }
        },
        "additionalProperties": true
      },
      "TimeInfo": {
        "type": "object",
        "required": ["queue_time", "prompt_time", "completion_time", "total_time", "created"],
        "properties": {
          "queue_time": { "type": "number" },
          "prompt_time": { "type": "number" },
          "completion_time": { "type": "number" },
          "total_time": { "type": "number" },
          "created": { "type": "number", "description": "Unix timestamp (seconds) when time_info recorded." }
        },
        "additionalProperties": true
      },
      "ErrorResponse": {
        "type": "object",
        "description": "Generic error response schema (shape may vary).",
        "properties": {
          "error": {
            "type": "object",
            "properties": {
              "message": { "type": "string" },
              "type": { "type": "string" },
              "code": { "type": ["string", "null"] },
              "param": { "type": ["string", "null"] }
            },
            "additionalProperties": true
          }
        },
        "additionalProperties": true
      }
    }
  }
}
