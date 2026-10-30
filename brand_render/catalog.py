"""Brand database indexing + retrieval.

Indexing runs ONCE (scripts/build_index.py or the app's "Rebuild index" button):
  1. every JPG in brand_db/<category>/ is described by the vision model (what it is, materials, colors)
  2. every JPG gets a multimodal embedding
  3. a written "brand brief" is distilled from a sample of all images
Result is saved to brand_db/index.json and committed to the repo, so rendering never re-reads the whole DB.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import BaseModel

from . import config, gemini

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


class ImageTags(BaseModel):
    subject: str            # e.g. "oak display table", "warm 3000K track lighting"
    applies_to: list[str]   # scene elements it could style: floor, wall, ceiling, shelving, table, rack, counter, fitting_room, lighting, signage...
    materials: list[str]
    colors: list[str]       # names or hex
    description: str        # one sentence


@dataclass
class Entry:
    path: str               # relative to BRAND_DB_DIR
    category: str
    subject: str
    applies_to: list[str]
    materials: list[str]
    colors: list[str]
    description: str
    embedding: list[float]


TAG_PROMPT = (
    "This image belongs to a clothing retailer's store-design brand library, category '{cat}'. "
    "Describe it as a design reference: what it shows, which store elements it could style "
    "(use short snake_case element names), materials, colors. Be literal and concise."
)

BRIEF_PROMPT = (
    "These images are the brand library for a clothing retailer's physical stores. "
    "Write a precise BRAND RENDERING BRIEF (max 250 words) that a 3D visualizer would follow: "
    "color palette (with approximate hex), floor/wall/ceiling materials, furniture finishes, "
    "metal finishes, lighting (color temperature, type, mood), signage style, overall mood. "
    "Only state what is visible in the images. Bullet points, no preamble."
)


def _images_in(folder: Path):
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXT)


def build_index(progress=None) -> dict:
    c = gemini.client()
    root = config.BRAND_DB_DIR
    cats = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    files = [(cat, p) for cat in cats for p in _images_in(root / cat)]
    if not files:
        raise RuntimeError(f"No images found under {root}/<category>/")

    entries = []
    for i, (cat, p) in enumerate(files):
        img = _load(p, max_side=1024)
        tags = gemini.ask_json(c, [gemini.to_part(img), TAG_PROMPT.format(cat=cat)], ImageTags)
        emb = gemini.embed_image(c, img)
        entries.append(Entry(str(p.relative_to(root)), cat, tags.subject, tags.applies_to,
                             tags.materials, tags.colors, tags.description, emb))
        if progress:
            progress((i + 1) / len(files), f"{cat}/{p.name}")

    # Brand brief from a balanced sample (max 14 images per call)
    sample = []
    for cat in cats:
        cat_files = [p for cc, p in files if cc == cat]
        sample += random.Random(0).sample(cat_files, min(3, len(cat_files)))
    sample = sample[:14]
    brief = gemini.ask_text(c, [gemini.to_part(_load(p, 768)) for p in sample] + [BRIEF_PROMPT])

    index = {"brand_brief": brief, "categories": cats, "entries": [asdict(e) for e in entries]}
    config.INDEX_PATH.write_text(json.dumps(index, ensure_ascii=False))
    return index


def load_index() -> dict | None:
    if config.INDEX_PATH.exists():
        return json.loads(config.INDEX_PATH.read_text())
    return None


def _load(path: Path, max_side: int = 1024) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))
    return img


def load_ref(entry: dict, max_side: int = 1024) -> Image.Image:
    return _load(config.BRAND_DB_DIR / entry["path"], max_side)


def retrieve(index: dict, query_vec: list[float], element: str | None = None,
             categories: list[str] | None = None, k: int = 2, exclude: set[str] = frozenset()) -> list[dict]:
    """Cosine-similarity search, boosted when the entry's `applies_to` matches the element."""
    q = np.asarray(query_vec, dtype=np.float32)
    q /= np.linalg.norm(q) + 1e-9
    scored = []
    for e in index["entries"]:
        if e["path"] in exclude or (categories and e["category"] not in categories):
            continue
        v = np.asarray(e["embedding"], dtype=np.float32)
        s = float(q @ (v / (np.linalg.norm(v) + 1e-9)))
        if element and any(element.lower() in a.lower() or a.lower() in element.lower()
                           for a in e["applies_to"]):
            s += 0.15
        scored.append((s, e))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [e for _, e in scored[:k]]
