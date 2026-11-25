"""Direct Anthropic API client for Claude models, used when ANTHROPIC_API_KEY is set.

It returns the same result shape as openrouter.chat, plus provider="anthropic". OpenRouter model IDs such as
"anthropic/claude-fable-5.2" are mapped to Anthropic IDs ("claude-fable-5-2").
"""
import os

# USD per 1M tokens (input, output). Cache reads are billed lower; this estimate ignores caching.
PRICES = {
    "claude-fable-5-2": (8, 40), "claude-fable-5-1": (10, 50), "claude-opus-5-5": (4, 20), "claude-opus-5": (5, 25),
    "claude-opus-4-8": (5, 25), "claude-sonnet-5-5": (2, 10), "claude-sonnet-5": (2, 10), "claude-haiku-5-5": (0.1, 0.5),
}
FALLBACK_BETA = "server-side-fallback-2026-07-01"
_client = None


def available():
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def handles(model_id):
    return model_id.startswith("anthropic/") and available()


def native_id(model_id):
    return model_id.removeprefix("anthropic/").replace(".", "-")


def _get_client():
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic(timeout=180.0, max_retries=2)
    return _client


def chat(role: str, model_id: str, system: str, messages: list[dict]) -> dict:
    import anthropic

    model = native_id(model_id)
    result = dict(ok=False, text="", model=model_id, prompt_tokens=0, completion_tokens=0, cost_usd=0.0,
                  error=None, finish_reason=None, provider="anthropic")
    try:
        # Thinking is always on for these models; effort is the control. A safety decline is re-run on the
        # server's default fallback model inside the same call.
        response = _get_client().beta.messages.create(
            model=model, max_tokens=6000, system=system, messages=messages,
            output_config={"effort": "low"}, betas=[FALLBACK_BETA], fallbacks="default",
        )
    except anthropic.AuthenticationError:
        result["error"] = "Anthropic: invalid API key"
        return result
    except anthropic.RateLimitError:
        result["error"] = "Anthropic: rate limited"
        return result
    except anthropic.APIStatusError as exc:
        result["error"] = f"Anthropic: HTTP {exc.status_code}: {str(exc.message)[:400]}"
        return result
    except anthropic.APIConnectionError as exc:
        result["error"] = f"Anthropic: connection error: {exc}"[:400]
        return result

    usage = response.usage
    prompt = (usage.input_tokens or 0) + (getattr(usage, "cache_read_input_tokens", 0) or 0) \
        + (getattr(usage, "cache_creation_input_tokens", 0) or 0)
    completion = usage.output_tokens or 0
    in_price, out_price = PRICES.get(response.model, PRICES.get(model, (8, 40)))
    text = "".join(block.text for block in response.content if block.type == "text")
    result.update(prompt_tokens=prompt, completion_tokens=completion, text=text, finish_reason=response.stop_reason,
                  cost_usd=(prompt * in_price + completion * out_price) / 1_000_000)
    if response.model != model:
        result["served_by"] = response.model
    if response.stop_reason == "refusal":
        category = getattr(getattr(response, "stop_details", None), "category", None)
        result["error"] = "refusal" + (f" ({category})" if category else "")
    elif not text.strip():
        result["error"] = "empty text"
    else:
        result["ok"] = True
    return result
