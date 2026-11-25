"""Model client for the assist roles: OpenRouter, or the Anthropic API for Claude models when its key is set."""

import math
import os

import httpx

from . import anthropic_api, mock

DEFAULT_ROLES = {
    "translate": "anthropic/claude-fable-5.2",
    "critic": "openai/gpt-6.2-sol",
    "alternatives": "google/gemini-4-neon",
}
PRICES = {
    "anthropic/claude-fable-5.2": (8, 40),
    "openai/gpt-6.2-sol": (2, 10),
    "google/gemini-4-neon": (3, 15),
}
URL = "https://openrouter.ai/api/v1/chat/completions"


def chat(role: str, model_id: str, system: str, messages: list[dict]) -> dict:
    result = dict(ok=False, text="", model=model_id, prompt_tokens=0, completion_tokens=0,
                  cost_usd=0.0, error=None, finish_reason=None)
    if os.getenv("WSB_LLM") == "mock":
        result.update(ok=True, text=mock.reply(role, messages), finish_reason="stop")
        return result
    if anthropic_api.handles(model_id):        # Claude models go to the Anthropic API when its key is set
        return anthropic_api.chat(role, model_id, system, messages)
    result["provider"] = "openrouter"
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        result["error"] = "no-key"
        return result
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "X-Title": "wsbench"}
    body = {
        "model": model_id,
        "messages": [{"role": "system", "content": system}, *messages],
        "max_tokens": 6000, "reasoning": {"effort": "low", "exclude": True},
        "usage": {"include": True},
    }
    try:
        response = httpx.post(URL, headers=headers, json=body, timeout=180.0)
        if response.status_code != 200:
            detail = response.text.replace(key, "[redacted]")[:500]
            result["error"] = f"HTTP {response.status_code}: {detail}"
            return result
        payload = response.json()
        usage = payload.get("usage") or {}
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        cost = usage.get("cost")
        if cost is None:
            in_price, out_price = PRICES.get(model_id, (0, 0))
            cost = (prompt * in_price + completion * out_price) / 1_000_000
        cost = float(cost)
        if not math.isfinite(cost):
            raise ValueError("non-finite usage cost")
        result.update(prompt_tokens=prompt, completion_tokens=completion, cost_usd=cost)
        choice = payload["choices"][0]
        content = choice["message"].get("content") or ""
        if not isinstance(content, str):
            raise ValueError("non-text message content")
        reason = choice.get("finish_reason")
        result.update(text=content, finish_reason=reason)
        if reason == "content_filter":
            result["error"] = "content_filter"
        elif not content.strip():
            result["error"] = "empty text"
        else:
            result["ok"] = True
    except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError, AttributeError,
            RecursionError, OverflowError) as exc:
        result["error"] = f"OpenRouter: {exc}".replace(key, "[redacted]")
    return result
