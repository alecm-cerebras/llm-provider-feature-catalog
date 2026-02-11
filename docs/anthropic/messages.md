# Anthropic Inference API - Messages (OpenAPI 3.1)

This page contains the OpenAPI 3.1 specification for `POST /messages`.

## OpenAPI JSON

```json
{
  "openapi": "3.1.0",
  "info": {
    "title": "Anthropic Inference API - Messages",
    "description": "Machine-verifiable OpenAPI 3.1 contract for POST /messages."
  },
  "servers": [
    {
      "url": "https://api.anthropic.com"
    }
  ],
  "paths": {
    "/messages": {
      "post": {
        "summary": "Create a message",
        "operationId": "createMessage",
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
                "$ref": "#/components/schemas/MessageCreateRequest"
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
                  "$ref": "#/components/schemas/MessageResponse"
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
            "description": "Rate limit / capacity rejection.",
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
      "MessageCreateRequest": {
        "type": "object",
        "required": [
          "model",
          "messages",
          "max_tokens"
        ],
        "properties": {
          "model": {
            "type": "string"
          },
          "system": {
            "type": [
              "string",
              "null"
            ],
            "description": "System prompt (top-level)."
          },
          "messages": {
            "type": "array",
            "minItems": 1,
            "items": {
              "$ref": "#/components/schemas/AnthropicMessage"
            }
          },
          "max_tokens": {
            "type": "integer"
          },
          "tools": {
            "type": [
              "array",
              "null"
            ],
            "items": {
              "$ref": "#/components/schemas/AnthropicTool"
            }
          }
        },
        "additionalProperties": true
      },
      "AnthropicMessage": {
        "type": "object",
        "required": [
          "role",
          "content"
        ],
        "properties": {
          "role": {
            "type": "string",
            "enum": [
              "user",
              "assistant"
            ]
          },
          "content": {
            "type": "array",
            "items": {
              "$ref": "#/components/schemas/ContentBlock"
            }
          }
        },
        "additionalProperties": true
      },
      "ContentBlock": {
        "oneOf": [
          {
            "type": "object",
            "required": [
              "type",
              "text"
            ],
            "properties": {
              "type": {
                "const": "text"
              },
              "text": {
                "type": "string"
              }
            },
            "additionalProperties": true
          },
          {
            "type": "object",
            "required": [
              "type",
              "thinking"
            ],
            "properties": {
              "type": {
                "const": "thinking"
              },
              "thinking": {
                "type": "string"
              }
            },
            "additionalProperties": true
          },
          {
            "type": "object",
            "required": [
              "type",
              "name",
              "input"
            ],
            "properties": {
              "type": {
                "const": "tool_use"
              },
              "name": {
                "type": "string"
              },
              "input": {
                "type": "object"
              }
            },
            "additionalProperties": true
          }
        ]
      },
      "AnthropicTool": {
        "type": "object",
        "required": [
          "name",
          "input_schema"
        ],
        "properties": {
          "name": {
            "type": "string"
          },
          "description": {
            "type": [
              "string",
              "null"
            ]
          },
          "input_schema": {
            "type": "object"
          }
        },
        "additionalProperties": true
      },
      "MessageResponse": {
        "type": "object",
        "additionalProperties": true
      },
      "ErrorResponse": {
        "type": "object",
        "additionalProperties": true
      }
    }
  }
}
```
