"""Thin wrapper around the google-genai SDK."""
from __future__ import annotations

import io
import json
import time

from google import genai
from google.genai import types
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


def _retry(fn, attempts: int = 3):
    for i in range(attempts):
        try:
            return fn()
        except Exception:
            if i == attempts - 1:
                raise
            time.sleep(2 ** i * 2)


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


def list_models(c: genai.Client) -> list[str]:
    try:
        return sorted(m.name.removeprefix("models/") for m in c.models.list())
    except Exception:
        return []
