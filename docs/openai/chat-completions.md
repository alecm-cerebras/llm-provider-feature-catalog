# OpenAI Inference API - Chat Completions (OpenAPI 3.1)

This page contains the OpenAPI 3.1 specification for `POST /v1/chat/completions`.

## OpenAPI JSON

```json
{
  "openapi": "3.1.0",
  "info": {
    "title": "OpenAI Inference API - Chat Completions",
    "description": "Machine-verifiable OpenAPI 3.1 contract for POST /v1/chat/completions."
  },
  "servers": [
    {
      "url": "https://api.openai.com/v1"
    }
  ],
  "paths": {
    "/chat/completions": {
      "post": {
        "summary": "Chat Completions",
        "operationId": "createChatCompletion",
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
                "$ref": "#/components/schemas/ChatCompletionRequest"
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
                  "$ref": "#/components/schemas/ChatCompletionResponse"
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
            "description": "Rate limit / capacity / queue threshold rejection.",
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
      "ChatCompletionRequest": {
        "type": "object",
        "required": [
          "model",
          "messages"
        ],
        "properties": {
          "model": {
            "type": "string",
            "description": "Model ID. May be a public model name or an org endpoint id.",
            "examples": [
              "gpt-oss-120b",
              "llama-3.3-70b",
              "my-org-llama-3.3-70b"
            ]
          },
          "messages": {
            "type": "array",
            "minItems": 1,
            "description": "A list of messages comprising the conversation so far.",
            "items": {
              "$ref": "#/components/schemas/ChatMessage"
            }
          },
          "logprobs": {
            "type": "boolean",
            "default": false,
            "description": "Whether to return log probabilities of the output tokens."
          },
          "top_logprobs": {
            "type": [
              "integer",
              "null"
            ],
            "minimum": 0,
            "maximum": 20,
            "description": "Number of most likely tokens to return per position. Requires logprobs=true."
          },
          "max_completion_tokens": {
            "type": [
              "integer",
              "null"
            ],
            "minimum": 0,
            "description": "Maximum number of tokens generated in the completion."
          },
          "stream": {
            "type": [
              "boolean",
              "null"
            ],
            "default": false,
            "description": "If true, partial message deltas will be sent as server-sent events."
          },
          "temperature": {
            "type": [
              "number",
              "null"
            ],
            "minimum": 0,
            "maximum": 2,
            "description": "Sampling temperature."
          },
          "top_p": {
            "type": [
              "number",
              "null"
            ],
            "minimum": 0,
            "maximum": 1,
            "description": "Nucleus sampling parameter."
          },
          "tools": {
            "type": [
              "array",
              "null"
            ],
            "description": "A list of tools the model may call. Currently, only functions are supported.",
            "items": {
              "$ref": "#/components/schemas/ToolDefinition"
            }
          },
          "tool_choice": {
            "description": "Controls which (if any) tool is called by the model.",
            "oneOf": [
              {
                "type": "string",
                "enum": [
                  "none",
                  "auto",
                  "required"
                ]
              },
              {
                "type": "object",
                "required": [
                  "type",
                  "function"
                ],
                "properties": {
                  "type": {
                    "type": "string",
                    "const": "function"
                  },
                  "function": {
                    "type": "object",
                    "required": [
                      "name"
                    ],
                    "properties": {
                      "name": {
                        "type": "string"
                      }
                    },
                    "additionalProperties": false
                  }
                },
                "additionalProperties": false
              }
            ]
          },
          "user": {
            "type": [
              "string",
              "null"
            ],
            "description": "Unique identifier representing your end-user."
          }
        },
        "additionalProperties": true
      },
      "ChatMessage": {
        "type": "object",
        "required": [
          "role",
          "content"
        ],
        "properties": {
          "role": {
            "type": "string",
            "enum": [
              "system",
              "user",
              "assistant"
            ]
          },
          "content": {
            "type": "string",
            "description": "Message text content."
          }
        },
        "additionalProperties": false
      },
      "ToolDefinition": {
        "type": "object",
        "required": [
          "type",
          "function"
        ],
        "properties": {
          "type": {
            "type": "string",
            "const": "function"
          },
          "function": {
            "type": "object",
            "required": [
              "name"
            ],
            "properties": {
              "name": {
                "type": "string",
                "maxLength": 64,
                "pattern": "^[A-Za-z0-9_-]+$",
                "description": "Function name to be called."
              },
              "description": {
                "type": "string"
              },
              "parameters": {
                "type": "object",
                "description": "JSON Schema object describing tool arguments."
              }
            },
            "additionalProperties": true
          }
        },
        "additionalProperties": false
      },
      "ChatCompletionResponse": {
        "type": "object",
        "required": [
          "id",
          "choices",
          "created",
          "model",
          "object"
        ],
        "properties": {
          "id": {
            "type": "string",
            "description": "Unique identifier for the chat completion."
          },
          "object": {
            "type": "string"
          },
          "created": {
            "type": "integer",
            "description": "Unix timestamp (seconds)."
          },
          "model": {
            "type": "string"
          },
          "choices": {
            "type": "array",
            "minItems": 1,
            "items": {
              "type": "object"
            }
          },
          "usage": {
            "type": "object"
          }
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
              "message": {
                "type": "string"
              },
              "type": {
                "type": "string"
              },
              "code": {
                "type": [
                  "string",
                  "null"
                ]
              },
              "param": {
                "type": [
                  "string",
                  "null"
                ]
              }
            },
            "additionalProperties": true
          }
        },
        "additionalProperties": true
      }
    }
  }
}
```
