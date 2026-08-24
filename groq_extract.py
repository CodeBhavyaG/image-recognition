"""Groq vision LLM -> structured schema extraction (capstone Phase 2).

Turns an image into a Pydantic-validated ``ImageUnderstanding`` (subject,
category, attributes, caption, confidence) using a Groq vision model through
Groq's OpenAI-compatible endpoint. Every response is validated at the boundary
(``schemas.ImageUnderstanding``); malformed output is retried, never trusted.

Setup
-----
1. Get a free key at https://console.groq.com  (no credit card).
2. Export it or put it in a ``.env`` file (see ``.env.example``):

       GROQ_API_KEY=your_key_here
       GROQ_VISION_MODEL=qwen/qwen3.6-27b

CLI
---
    python groq_extract.py path/to/image.jpg [more.jpg ...]
"""
import base64
import json
import os
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # python-dotenv is optional
    pass

from schemas import ImageUnderstanding

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# Groq's current multimodal (vision) model. Check with `--list-models`.
DEFAULT_VISION_MODEL = os.environ.get("GROQ_VISION_MODEL", "qwen/qwen3.6-27b")

SYSTEM_PROMPT = (
    "You are an image understanding engine. Return ONLY a JSON object with "
    'exactly these keys: "subject" (a short string naming the main subject), '
    '"category" (a high-level category such as "animal"), '
    '"attributes" (an array of short descriptive strings), '
    '"caption" (one sentence describing the image), '
    '"confidence" (a number between 0 and 1). '
    "Output valid JSON only — no markdown, no code fences, no extra text."
)


def get_client(api_key: Optional[str] = None):
    """Build an OpenAI-compatible client pointed at Groq's API."""
    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Add it to a .env file (see .env.example) "
            "or export it in your shell."
        )
    from openai import OpenAI  # lazy import so the module can be tested w/o the SDK
    return OpenAI(base_url=GROQ_BASE_URL, api_key=key)


def encode_image(image_path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _extract_json(text: str) -> dict:
    """Parse a model response into a dict, tolerating code fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


def parse_and_validate(text: str) -> ImageUnderstanding:
    """Parse + validate a raw model response. Raises on malformed output."""
    return ImageUnderstanding.model_validate(_extract_json(text))


def extract_understanding(
    image_path,
    model: str = DEFAULT_VISION_MODEL,
    max_tries: int = 3,
    client=None,
) -> ImageUnderstanding:
    """Send an image to a Groq vision model and return a validated schema.

    Uses Groq JSON mode first; if the model doesn't support it, falls back to a
    plain prompt and still validates the parsed JSON with Pydantic.
    """
    client = client or get_client()
    data_url = "data:image/jpeg;base64," + encode_image(image_path)
    content = [
        {"type": "text", "text": SYSTEM_PROMPT},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]

    last_error: Optional[Exception] = None
    for attempt in range(1, max_tries + 1):
        try:
            try:
                # Try Groq's JSON output mode first.
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": content}],
                    response_format={"type": "json_object"},
                )
            except Exception:
                # Fallback: plain prompt (model may not support JSON mode).
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": content}],
                )
            return parse_and_validate(resp.choices[0].message.content)
        except Exception as e:
            last_error = e
            print(f"[attempt {attempt}/{max_tries}] failed: {e}")

    raise RuntimeError(f"Schema extraction failed after {max_tries} tries: {last_error}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Groq vision LLM -> structured schema extraction."
    )
    ap.add_argument("images", nargs="*", help="Image file(s) to analyze.")
    ap.add_argument("--model", default=DEFAULT_VISION_MODEL, help="Groq vision model ID.")
    ap.add_argument("--list-models", action="store_true",
                    help="List the models available to your GROQ_API_KEY.")
    args = ap.parse_args()

    client = get_client()

    if args.list_models:
        for m in client.models.list():
            print(m.id)
        raise SystemExit(0)

    if not args.images:
        ap.print_help()
        raise SystemExit(1)

    for p in args.images:
        if not Path(p).exists():
            print(f"!! file not found: {p}", file=sys.stderr)
            continue
        u = extract_understanding(p, model=args.model, client=client)
        print(json.dumps(u.model_dump(), indent=2))