# Chat Template Format Guide

This document explains how to create and customize chat templates for TensorRT Edge-LLM models.

## Overview

Chat templates define how conversational messages are formatted before being processed by the language model. Each model typically has its own chat template format with specific tokens and structures. The chat template ensures that system prompts, user messages, and assistant responses are properly formatted according to the model's training format.

### Implementation Philosophy

Our chat template implementation follows closely with HuggingFace's `apply_chat_template` API:

```python
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True  # For models that support thinking mode
)
```

To keep TensorRT Edge-LLM lightweight and free from Jinja dependencies, we use a simple JSON-based chat template format. In practice, the dynamic nature of Jinja templates is only useful when dealing with variables that change the behavior of chat template application, or to handle many different types of messages in one logic block. By restricting messages to follow a specified order and format, and supported uses cases, we can condense the chat template into a list of prefix/suffix pairs and formats for a supported list of multimodalities, achieving the same result with greater simplicity. 

### Key Concepts

- **Roles**: Different message types (system, user, assistant)
- **Prefixes/Suffixes**: Special tokens that wrap each message
- **Content Types**: Formatting for multimodal content (text, image, video)
- **Generation Prompt**: Token sequence that prompts the model to generate a response (may include model-specific markers to disable thinking for models that support it)
- **Generation Prompt (Thinking)**: Alternative generation prompt to enable thinking mode (optional, only present for models with thinking support)
- **Default System Prompt**: Fallback system instruction when none is provided

---

## Chat Template File Format

The chat template is defined in a JSON file. When exported with the model, this file is renamed to `processed_chat_template.json` and placed in the model's engine directory.

### File Structure

```json
{
  "model_path": "string (optional)",
  "roles": {
    "system": {
      "prefix": "string",
      "suffix": "string"
    },
    "user": {
      "prefix": "string",
      "suffix": "string"
    },
    "assistant": {
      "prefix": "string",
      "suffix": "string"
    }
  },
  "content_types": {
    "image": {
      "format": "string"
    },
    "video": {
      "format": "string"
    }
  },
  "generation_prompt": "string",
  "generation_prompt_thinking": "string (optional)",
  "default_system_prompt": "string"
}
```

### Field Descriptions

#### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `roles` | object | Maps role names to their prefix/suffix formatting |
| `roles.system` | object | Formatting for system messages |
| `roles.user` | object | Formatting for user messages |
| `roles.assistant` | object | Formatting for assistant messages |
| `roles.<role>.prefix` | string | Token(s) placed before the message content (can be empty string) |
| `roles.<role>.suffix` | string | Token(s) placed after the message content (can be empty string) |

#### Optional Fields

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `model_path` | string | Path or identifier for the model | "" |
| `content_types` | object | Formatting for multimodal content (images, videos) | {} |
| `content_types.image.format` | string | Token sequence that replaces image content in the formatted prompt | "" |
| `content_types.video.format` | string | Token sequence that replaces video content in the formatted prompt | "" |
| `generation_prompt` | string | Token sequence to prompt model generation (may include model-specific markers to disable thinking for models that support it) | "" |
| `generation_prompt_thinking` | string | Token sequence to prompt model generation to enable thinking (optional, model-specific, only present for models with thinking support) | "" |
| `default_system_prompt` | string | Default system instruction when none provided | "" |

---

## Thinking Mode Support

Some models (like Qwen3) support a "thinking mode" where the model can optionally generate its reasoning process before providing the final answer. This is controlled by the `enable_thinking` parameter in the input JSON.

**Important:** Thinking mode is only effective for models that are trained with this capability. During model export, the system automatically detects whether the model supports thinking mode by examining the tokenizer's chat template. If supported, both generation prompts are extracted and saved.

### How It Works

When a model supports thinking mode, the chat template can define two different generation prompts:

- **`generation_prompt`**: Generation prompt that disables thinking mode
  - For Qwen3 example: `<|im_start|>assistant\n<think>\n\n</think>\n\n`
  - The specific mechanism varies by model. In Qwen3's case, empty `<think>` tags signal that thinking has already been done, so the model should provide the answer directly
  
- **`generation_prompt_thinking`**: Generation prompt that enables thinking mode
  - For Qwen3 example: `<|im_start|>assistant\n`
  - Without predefined thinking markers, the model is free to generate its own reasoning process

During model export, if the tokenizer supports thinking mode, both generation prompts are automatically extracted and saved to the chat template. If the model doesn't support thinking mode, only `generation_prompt` is present, and the `enable_thinking` parameter will have no effect.

**Note:** The `enable_thinking` parameter naming follows HuggingFace's implementation for models that support thinking. When set to `False` (default), the model may use the prompt with empty `<think>` tags to disable thinking. When set to `True`, it uses the prompt without thinking tags, allowing the model to generate its reasoning.

### Example Output Format

For Qwen3, when thinking mode is disabled (default, `enable_thinking=False`), the formatted prompt includes empty thinking tags:

```
<|im_start|>user
Give me a short introduction to large language model.<|im_end|>
<|im_start|>assistant
<think>

</think>

```

In Qwen3's case, the empty `<think>` tags tell the model not to generate additional reasoning and to provide the answer directly. Other models may use different mechanisms to control thinking behavior.

### Usage

In your input JSON, set the `enable_thinking` parameter to control which generation prompt is used:

```json
{
  "enable_thinking": true,
  "requests": [
    {
      "messages": [
        {"role": "user", "content": "Solve this complex problem..."}
      ]
    }
  ]
}
```

**Behavior:**

- When `enable_thinking` is **not specified** (default `false`): Uses `generation_prompt` (thinking mode disabled, if model supports thinking)
- When `enable_thinking` is `false`: Uses `generation_prompt` (thinking mode disabled, if model supports thinking)
- When `enable_thinking` is `true`: Uses `generation_prompt_thinking` (thinking mode enabled, if model supports thinking)
- When model doesn't support thinking mode: The `enable_thinking` parameter has no effect; always uses `generation_prompt`

**Note:** The specific mechanism for enabling/disabling thinking varies by model and is determined during model export when the chat template is extracted from the tokenizer.

For more details on the `enable_thinking` parameter, see [INPUT_FORMAT.md](../../../examples/llm/INPUT_FORMAT.md).

---

## Using Chat Templates During Export

By default, when you export a model, the tool:

1. Tries to detect and extract a chat template from the model's tokenizer.
2. If the model is known to have an incompatible or missing template, automatically falls back to a pre-built template (see [Pre-built Templates](#pre-built-templates)).


### Providing a Custom Template (Optional)

You can override the default behavior by supplying your own chat template with the `--chat_template` flag:

```bash
tensorrt-edgellm-export-llm \
    --model_dir /path/to/model \
    --output_dir /path/to/output \
    --chat_template /path/to/my_custom_template.json
```

Provide a custom template when:

1. **Automatic extraction fails**: The export tool cannot reliably detect or convert the model's chat template.
2. **Model is not covered by pre-built templates** but still has issues with its tokenizer template.
3. **You want different formatting** than the model's default (for example, different system prompts, role tokens, or multimodal placeholders).

---

## Pre-built Templates

Some models have known issues with their tokenizer's chat template definitions (e.g., missing templates or incompatible formats). For these models, TensorRT Edge-LLM includes pre-built templates that are used automatically.

**Location:** `tensorrt_edgellm/chat_templates/templates/`

| Model | Template File | Reason |
|-------|--------------|--------|
| Phi-4-Multimodal | `phi4mm.json` | Tokenizer lacks proper multimodal chat template definition |

**Automatic Fallback:**
When you export a model that is on the incompatible list (like Phi-4-Multimodal), the system automatically uses the pre-built template. You don't need to manually specify it unless you want to override it.

---

## Examples

### Basic Text-Only Model

```json
{
  "model_path": "/path/to/Qwen2-7B",
  "roles": {
    "system": {
      "prefix": "<|im_start|>system\n",
      "suffix": "<|im_end|>\n"
    },
    "user": {
      "prefix": "<|im_start|>user\n",
      "suffix": "<|im_end|>\n"
    },
    "assistant": {
      "prefix": "<|im_start|>assistant\n",
      "suffix": "<|im_end|>\n"
    }
  },
  "content_types": {},
  "generation_prompt": "<|im_start|>assistant\n",
  "default_system_prompt": "You are a helpful assistant"
}
```

### Multimodal Model with Content Types

For vision-language models, the `format` field specifies the token sequence that replaces each image or video in the formatted prompt:

```json
{
  "model_path": "/path/to/Qwen2-VL-7B",
  "roles": {
    "system": {
      "prefix": "<|im_start|>system\n",
      "suffix": "<|im_end|>\n"
    },
    "user": {
      "prefix": "<|im_start|>user\n",
      "suffix": "<|im_end|>\n"
    },
    "assistant": {
      "prefix": "<|im_start|>assistant\n",
      "suffix": "<|im_end|>\n"
    }
  },
  "content_types": {
    "image": {
      "format": "<|vision_start|><|image_pad|><|vision_end|>"
    },
    "video": {
      "format": "<|vision_start|><|video_pad|><|vision_end|>"
    }
  },
  "generation_prompt": "<|im_start|>assistant\n",
  "default_system_prompt": "You are a helpful assistant."
}
```

### Model with Thinking Mode Support

For models that support thinking mode (like Qwen3), the chat template includes both generation prompts:

```json
{
  "model_path": "/path/to/Qwen3-0.6B",
  "roles": {
    "system": {
      "prefix": "<|im_start|>system\n",
      "suffix": "<|im_end|>\n"
    },
    "user": {
      "prefix": "<|im_start|>user\n",
      "suffix": "<|im_end|>\n"
    },
    "assistant": {
      "prefix": "<|im_start|>assistant\n",
      "suffix": "<|im_end|>\n"
    }
  },
  "content_types": {},
  "generation_prompt": "<|im_start|>assistant\n<think>\n\n</think>\n\n",
  "generation_prompt_thinking": "<|im_start|>assistant\n",
  "default_system_prompt": "You are a helpful assistant"
}
```

The `generation_prompt` field may include model-specific markers to disable thinking (e.g., empty `<think>` tags for Qwen3). The `generation_prompt_thinking` field enables thinking by omitting such markers (used when `enable_thinking: true` is set in the input JSON). The specific format is automatically determined during model export based on the model's tokenizer configuration.

### Multi-turn Conversation Example

**Template:**
```json
{
  "roles": {
    "system": {
      "prefix": "<|im_start|>system\n",
      "suffix": "<|im_end|>\n"
    },
    "user": {
      "prefix": "<|im_start|>user\n",
      "suffix": "<|im_end|>\n"
    },
    "assistant": {
      "prefix": "<|im_start|>assistant\n",
      "suffix": "<|im_end|>\n"
    }
  },
  "generation_prompt": "<|im_start|>assistant\n",
  "default_system_prompt": "You are a helpful assistant"
}
```

And this input:
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a math tutor."
    },
    {
      "role": "user",
      "content": "What is 2+2?"
    },
    {
      "role": "assistant",
      "content": "2+2 equals 4."
    },
    {
      "role": "user",
      "content": "What about 3+3?"
    }
  ]
}
```

The formatted output will be:
```
<|im_start|>system
You are a math tutor.<|im_end|>
<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
2+2 equals 4.<|im_end|>
<|im_start|>user
What about 3+3?<|im_end|>
<|im_start|>assistant

```
---

## System Prompt Priority

The system prompt is determined by the following priority order:

1. **Explicit system message in request** (highest priority)
2. **`default_system_prompt` from chat template** (lowest priority)

If neither is provided, no system prompt is added to the conversation.

**Example scenarios:**

**Scenario 1: Explicit system message (highest priority)**

```json
{
  "requests": [
    {
      "messages": [
        {"role": "system", "content": "You are a math tutor."},
        {"role": "user", "content": "What is 2+2?"}
      ]
    }
  ]
}
```

Result: Uses "You are a math tutor." regardless of the chat template's default.

**Scenario 2: Fallback to chat template default**

```json
{
  "requests": [
    {
      "messages": [
        {"role": "user", "content": "Hello!"}
      ]
    }
  ]
}
```

Result: Since no explicit system message exists, uses the `default_system_prompt` from `processed_chat_template.json` (if defined).

For more details on the input JSON format, see [INPUT_FORMAT.md](../../../examples/llm/INPUT_FORMAT.md).