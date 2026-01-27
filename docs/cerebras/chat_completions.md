{
  "openapi": "3.1.0",
  "info": {
    "title": "Cerebras Inference API - Chat Completions",
    "version": "2026-01",
    "description": "Machine-verifiable OpenAPI 3.1 contract for POST /v1/chat/completions."
  },
  "servers": [
    {
      "url": "https://api.cerebras.ai/v1"
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
        "parameters": [
          {
            "name": "queue_threshold",
            "in": "header",
            "required": false,
            "description": "Controls the queue time threshold for requests using the flex or auto service tiers. Valid range: 50-20000 (milliseconds). Private Preview.",
            "schema": {
              "type": "string"
            }
          },
          {
            "name": "X-Cerebras-Version-Patch",
            "in": "header",
            "required": false,
            "description": "Optional API version override header (e.g., \"2\").",
            "schema": {
              "type": "string",
              "pattern": "^[0-9]+$",
              "examples": ["2"]
            }
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
        "required": ["model", "messages"],
        "properties": {
          "model": {
            "type": "string",
            "description": "Model ID. May be a public model name or an org endpoint id.",
            "examples": ["gpt-oss-120b", "llama-3.3-70b", "my-org-llama-3.3-70b"]
          },
          "messages": {
            "type": "array",
            "minItems": 1,
            "description": "A list of messages comprising the conversation so far.",
            "items": {
              "$ref": "#/components/schemas/ChatMessage"
            }
          },

          "clear_thinking": {
            "type": ["boolean", "null"],
            "description": "Controls whether thinking content from previous turns is included in prompt context. Supported only on zai-glm-4.7."
          },

          "logprobs": {
            "type": "boolean",
            "default": false,
            "description": "Whether to return log probabilities of the output tokens."
          },
          "top_logprobs": {
            "type": ["integer", "null"],
            "minimum": 0,
            "maximum": 20,
            "description": "Number of most likely tokens to return per position. Requires logprobs=true."
          },

          "max_completion_tokens": {
            "type": ["integer", "null"],
            "minimum": 0,
            "description": "Maximum number of tokens generated in the completion, including reasoning tokens."
          },

          "parallel_tool_calls": {
            "type": ["boolean", "null"],
            "default": true,
            "description": "Whether to enable parallel function calling during tool use. If enabled, the model can request multiple tool calls in a single response."
          },

          "prediction": {
            "type": ["object", "null"],
            "description": "Predicted Output configuration.",
            "properties": {
              "type": {
                "type": "string",
                "const": "content",
                "description": "Type of the predicted content; currently always 'content'."
              },
              "content": {
                "description": "Predicted content to match. String or array of typed parts.",
                "oneOf": [
                  { "type": "string" },
                  {
                    "type": "array",
                    "items": {
                      "type": "object",
                      "required": ["type", "text"],
                      "properties": {
                        "type": { "type": "string", "const": "text" },
                        "text": { "type": "string" }
                      },
                      "additionalProperties": false
                    }
                  }
                ]
              }
            },
            "additionalProperties": true
          },

          "reasoning_effort": {
            "type": ["string", "null"],
            "enum": ["low", "medium", "high"],
            "description": "Controls the amount of reasoning the model performs. Only available for gpt-oss-120b."
          },

          "response_format": {
            "type": ["object", "null"],
            "description": "Controls the output format of the model response.",
            "oneOf": [
              { "$ref": "#/components/schemas/ResponseFormatText" },
              { "$ref": "#/components/schemas/ResponseFormatJsonSchema" },
              { "$ref": "#/components/schemas/ResponseFormatJsonObject" }
            ]
          },

          "seed": {
            "type": ["integer", "null"],
            "description": "Best-effort deterministic sampling (not guaranteed)."
          },

          "service_tier": {
            "type": ["string", "null"],
            "enum": ["priority", "default", "auto", "flex"],
            "default": "default",
            "description": "Controls request prioritization. Private Preview."
          },

          "stop": {
            "type": ["string", "null"],
            "description": "Up to 4 sequences where the API will stop generating further tokens."
          },

          "stream": {
            "type": ["boolean", "null"],
            "default": false,
            "description": "If true, partial message deltas will be sent as server-sent events."
          },

          "temperature": {
            "type": ["number", "null"],
            "minimum": 0,
            "maximum": 1.5,
            "description": "Sampling temperature."
          },

          "top_p": {
            "type": ["number", "null"],
            "minimum": 0,
            "maximum": 1,
            "description": "Nucleus sampling parameter."
          },

          "tool_choice": {
            "description": "Controls which (if any) tool is called by the model.",
            "oneOf": [
              {
                "type": "string",
                "enum": ["none", "auto", "required"]
              },
              {
                "type": "object",
                "required": ["type", "function"],
                "properties": {
                  "type": { "type": "string", "const": "function" },
                  "function": {
                    "type": "object",
                    "required": ["name"],
                    "properties": {
                      "name": { "type": "string" }
                    },
                    "additionalProperties": false
                  }
                },
                "additionalProperties": false
              }
            ]
          },

          "tools": {
            "type": ["array", "null"],
            "description": "A list of tools the model may call. Currently, only functions are supported.",
            "items": {
              "$ref": "#/components/schemas/ToolDefinition"
            }
          },

          "user": {
            "type": ["string", "null"],
            "description": "Unique identifier representing your end-user."
          }
        },
        "additionalProperties": true
      },

      "ChatMessage": {
        "type": "object",
        "required": ["role", "content"],
        "properties": {
          "role": {
            "type": "string",
            "enum": ["system", "user", "assistant"]
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
        "required": ["type", "function"],
        "properties": {
          "type": {
            "type": "string",
            "const": "function"
          },
          "function": {
            "type": "object",
            "required": ["name"],
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

      "ResponseFormatText": {
        "type": "object",
        "required": ["type"],
        "properties": {
          "type": { "type": "string", "const": "text" }
        },
        "additionalProperties": false
      },

      "ResponseFormatJsonObject": {
        "type": "object",
        "required": ["type"],
        "properties": {
          "type": { "type": "string", "const": "json_object" }
        },
        "additionalProperties": false
      },

      "ResponseFormatJsonSchema": {
        "type": "object",
        "required": ["type", "json_schema"],
        "properties": {
          "type": { "type": "string", "const": "json_schema" },
          "json_schema": {
            "type": "object",
            "required": ["name", "schema"],
            "properties": {
              "name": { "type": "string" },
              "description": { "type": "string" },
              "schema": {
                "type": "object",
                "description": "JSON Schema defining the structured output."
              },
              "strict": {
                "type": "boolean",
                "default": false
              }
            },
            "additionalProperties": false
          }
        },
        "additionalProperties": false
      },

      "ChatCompletionResponse": {
        "type": "object",
        "required": ["id", "choices", "created", "model", "object", "usage", "time_info", "system_fingerprint"],
        "properties": {
          "id": {
            "type": "string",
            "description": "Unique identifier for the chat completion."
          },
          "object": {
            "type": "string",
            "const": "chat.completion"
          },
          "created": {
            "type": "integer",
            "description": "Unix timestamp (seconds) of when the completion was created."
          },
          "model": {
            "type": "string",
            "description": "Model used."
          },
          "system_fingerprint": {
            "type": "string"
          },
          "choices": {
            "type": "array",
            "minItems": 1,
            "items": {
              "$ref": "#/components/schemas/ChatCompletionChoice"
            }
          },
          "service_tier_used": {
            "type": "string",
            "enum": ["priority", "default", "flex"],
            "description": "Only present when service_tier was set to auto in the request."
          },
          "usage": {
            "$ref": "#/components/schemas/Usage"
          },
          "time_info": {
            "$ref": "#/components/schemas/TimeInfo"
          }
        },
        "additionalProperties": true
      },

      "ChatCompletionChoice": {
        "type": "object",
        "required": ["finish_reason", "index", "message"],
        "properties": {
          "finish_reason": {
            "type": "string",
            "enum": ["stop", "length", "content_filter", "tool_calls"]
          },
          "index": {
            "type": "integer"
          },
          "message": {
            "$ref": "#/components/schemas/AssistantMessage"
          }
        },
        "additionalProperties": false
      },

      "AssistantMessage": {
        "type": "object",
        "required": ["role", "content"],
        "properties": {
          "role": {
            "type": "string",
            "const": "assistant"
          },
          "content": {
            "type": "string"
          },
          "reasoning": {
            "type": "string",
            "description": "Reasoning content when using reasoning models/settings."
          }
        },
        "additionalProperties": true
      },

      "Usage": {
        "type": "object",
        "required": ["prompt_tokens", "completion_tokens", "total_tokens", "prompt_tokens_details", "completion_tokens_details"],
        "properties": {
          "prompt_tokens": { "type": "integer" },
          "completion_tokens": { "type": "integer" },
          "total_tokens": { "type": "integer" },
          "prompt_tokens_details": {
            "type": "object",
            "required": ["cached_tokens"],
            "properties": {
              "cached_tokens": { "type": "integer" }
            },
            "additionalProperties": false
          },
          "completion_tokens_details": {
            "type": "object",
            "required": ["accepted_prediction_tokens", "rejected_prediction_tokens"],
            "properties": {
              "accepted_prediction_tokens": { "type": "integer" },
              "rejected_prediction_tokens": { "type": "integer" }
            },
            "additionalProperties": false
          }
        },
        "additionalProperties": false
      },

      "TimeInfo": {
        "type": "object",
        "required": ["queue_time", "prompt_time", "completion_time", "total_time", "created"],
        "properties": {
          "queue_time": { "type": "number", "description": "Seconds spent in queue." },
          "prompt_time": { "type": "number", "description": "Seconds spent processing prompt tokens." },
          "completion_time": { "type": "number", "description": "Seconds spent generating completion tokens." },
          "total_time": { "type": "number", "description": "Total end-to-end request time in seconds." },
          "created": { "type": "number", "description": "Unix timestamp (seconds) when time_info recorded." }
        },
        "additionalProperties": false
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
