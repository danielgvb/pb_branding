"""Thin wrapper around the google-genai SDK."""
from __future__ import annotations

import io
import json
import time

import re

from google import genai
from google.genai import errors, types
from PIL import Image

from . import config


def client() -> genai.Client:
    key = config.api_key()
    if not key:
        raise RuntimeError("Missing GEMINI_API_KEY (env var or .streamlit/secrets.toml).")
    return genai.Client(api_key=key)


def to_part(img: Image.Image | bytes, quality: int = 90) -> types.Part:
    if isinstance(img, Image.Image):
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=quality)
        img = buf.getvalue()
    return types.Part.from_bytes(data=img, mime_type="image/jpeg")


RETRYABLE = {429, 500, 502, 503, 504}


def _retry(fn, attempts: int = 5):
    """Retry only transient errors. 429 (rate limit) honours the server's suggested delay."""
    for i in range(attempts):
        try:
            return fn()
        except errors.APIError as e:
            if e.code not in RETRYABLE or i == attempts - 1:
                raise
            m = re.search(r"retry in ([\d.]+)s|retryDelay['\"]?: ?['\"]?(\d+)", str(e))
            wait = float(next(g for g in m.groups() if g)) + 1 if m else min(60, 5 * 2 ** i)
            time.sleep(wait)


def explain(e: Exception) -> str:
    """Human-readable explanation of a Gemini API error (Streamlit Cloud redacts the original)."""
    if not isinstance(e, errors.APIError):
        return f"{type(e).__name__}: {e}"
    msg = (e.message or str(e))[:600]
    hints = {
        400: "Bad request: usually the selected model does not support this call. Pick another model in the sidebar.",
        401: "API key invalid. Check GEMINI_API_KEY in Streamlit Secrets.",
        403: "The key has no access to this model (the image models need billing enabled on the Google AI Studio project).",
        404: "Model not found: the model name changed or is not available to your key. Pick another in the sidebar.",
        429: "Quota / rate limit exceeded. Free-tier keys have low per-minute limits and NO access to the Pro image "
             "model. Wait a minute and retry, enable billing in Google AI Studio, or choose a Flash image model.",
    }
    return f"Gemini API error {e.code} ({e.status}): {msg}\n\n**What to do:** {hints.get(e.code, 'See message above.')}"


def ask_json(c: genai.Client, contents: list, schema, model: str | None = None):
    """Vision/text call that returns a validated pydantic object."""
    resp = _retry(lambda: c.models.generate_content(
        model=model or config.VISION_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.1,
        ),
    ))
    if getattr(resp, "parsed", None) is not None:
        return resp.parsed
    return schema.model_validate(json.loads(resp.text))


def ask_text(c: genai.Client, contents: list, model: str | None = None) -> str:
    resp = _retry(lambda: c.models.generate_content(
        model=model or config.VISION_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(temperature=0.2),
    ))
    return resp.text or ""


def generate_image(c: genai.Client, contents: list, aspect_ratio: str,
                   model: str | None = None, size: str | None = None) -> Image.Image:
    model = model or config.RENDER_MODEL
    img_cfg = {"aspect_ratio": aspect_ratio}
    if not model.startswith("gemini-2."):  # image_size is not supported on 2.x image models
        img_cfg["image_size"] = size or config.OUTPUT_SIZE
    resp = _retry(lambda: c.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(**img_cfg),
        ),
    ))
    for cand in resp.candidates or []:
        for part in (cand.content.parts if cand.content else []) or []:
            if part.inline_data and part.inline_data.data:
                return Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")
    reason = resp.candidates[0].finish_reason if resp.candidates else "no candidates"
    raise RuntimeError(f"Model returned no image (finish_reason={reason}).")


def embed_image(c: genai.Client, img: Image.Image | bytes) -> list[float]:
    res = _retry(lambda: c.models.embed_content(
        model=config.EMBED_MODEL,
        contents=[to_part(img)],
        config=types.EmbedContentConfig(output_dimensionality=768),
    ))
    return list(res.embeddings[0].values)


def embed_text(c: genai.Client, text: str) -> list[float]:
    res = _retry(lambda: c.models.embed_content(
        model=config.EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=768),
    ))
    return list(res.embeddings[0].values)


EXCLUDE = ("live", "tts", "audio", "embedding", "aqa", "imagen", "veo", "robotics", "computer-use", "learnlm", "gemma")


def list_models(c: genai.Client) -> list[str]:
    """Models usable with generateContent, minus live/audio/tts/etc. variants."""
    try:
        out = []
        for m in c.models.list():
            name = m.name.removeprefix("models/")
            actions = m.supported_actions or []
            if "generateContent" in actions and name.startswith("gemini") and not any(x in name for x in EXCLUDE):
                out.append(name)
        return sorted(out)
    except Exception:
        return []
