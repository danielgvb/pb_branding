"""Brand material library: indexing + retrieval.

Layout supported:
  brand_db/PISO GENERAL.jpg            loose files -> the FILE NAME is the material label (authoritative)
  brand_db/<category>/<name>.jpg       optional sub-folders -> category = folder name

The file name tells the model WHERE the material goes ("PISO GENERAL" = general floor,
"TEXTURA COLUMNAS" = columns). Indexing adds a short visual description per image and a
brand brief, saved to brand_db/index.json. Re-run it whenever images are added/renamed.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import BaseModel

from . import config, gemini

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
ROOT_CATEGORY = "materiales"


class ImageTags(BaseModel):
    applies_to: list[str]   # store elements this material is for, e.g. ["floor"], ["walls"], ["columns"]
    material: str           # e.g. "walnut wood veneer", "terracotta brick-shaped tiles"
    colors: list[str]       # names + approximate hex
    finish: str             # matte / satin / gloss / textured...
    description: str        # one sentence, literal


@dataclass
class Entry:
    path: str               # relative to BRAND_DB_DIR
    label: str              # from the file name, e.g. "PISO GENERAL"
    category: str
    applies_to: list[str]
    material: str
    colors: list[str]
    finish: str
    description: str
    embedding: list[float] | None


TAG_PROMPT = (
    "This image is a material swatch from an architect's specification for a clothing store. "
    "Its official label (from the architect, in Spanish) is: '{label}'. The label is authoritative "
    "about WHERE the material is used (e.g. PISO = floor, PAREDES = walls, COLUMNAS = columns, "
    "FACHADA = facade, MADERA = wood for furniture/fixtures, METAL = metal fixtures). "
    "Describe the material literally: type, colors (approximate hex), finish, scale of pattern."
)

BRIEF_PROMPT = (
    "These are the architect's material swatches for a clothing store, with their labels:\n{labels}\n"
    "Write a precise MATERIAL & STYLE BRIEF (max 200 words, bullet points, English) a 3D visualizer "
    "would follow: one bullet per label saying which surfaces it covers and what it looks like "
    "(color hex, finish, pattern scale), then 2-3 bullets on the overall palette and mood. "
    "Only state what is visible or in the labels. No preamble."
)


def label_from_path(p: Path) -> str:
    return " ".join(p.stem.replace("_", " ").split()).upper()


def scan() -> list[tuple[str, Path]]:
    """(category, path) for every image in the library."""
    root = config.BRAND_DB_DIR
    out = [(ROOT_CATEGORY, p) for p in sorted(root.iterdir()) if p.is_file() and p.suffix.lower() in IMG_EXT]
    for d in sorted(x for x in root.iterdir() if x.is_dir() and not x.name.startswith(".")):
        out += [(d.name, p) for p in sorted(d.rglob("*")) if p.suffix.lower() in IMG_EXT]
    return out


def is_stale(index: dict | None) -> bool:
    if not index:
        return True
    on_disk = {str(p.relative_to(config.BRAND_DB_DIR)) for _, p in scan()}
    return on_disk != {e["path"] for e in index["entries"]}


def build_index(progress=None) -> dict:
    c = gemini.client()
    files = scan()
    if not files:
        raise RuntimeError(f"No images found in {config.BRAND_DB_DIR}")

    entries = []
    for i, (cat, p) in enumerate(files):
        label = label_from_path(p)
        img = load_path(p, 1024)
        tags = gemini.ask_json(c, [gemini.to_part(img), TAG_PROMPT.format(label=label)], ImageTags)
        try:
            emb = gemini.embed_image(c, img)
        except Exception:  # embeddings only matter for large libraries
            emb = None
        entries.append(Entry(str(p.relative_to(config.BRAND_DB_DIR)), label, cat, tags.applies_to,
                             tags.material, tags.colors, tags.finish, tags.description, emb))
        if progress:
            progress((i + 1) / len(files), label)

    sample = entries[:14]
    labels = "\n".join(f"- Image {i + 1}: {e.label}" for i, e in enumerate(sample))
    brief = gemini.ask_text(c, [gemini.to_part(load_path(config.BRAND_DB_DIR / e.path, 768)) for e in sample]
                            + [BRIEF_PROMPT.format(labels=labels)])

    index = {"brand_brief": brief, "categories": sorted({e.category for e in entries}),
             "entries": [asdict(e) for e in entries]}
    config.INDEX_PATH.write_text(json.dumps(index, ensure_ascii=False, indent=1))
    return index


def load_index() -> dict | None:
    if config.INDEX_PATH.exists():
        return json.loads(config.INDEX_PATH.read_text())
    return None


def load_path(path: Path, max_side: int = 1024) -> Image.Image:
    """Load a swatch. Very elongated swatches (e.g. a 1024x6144 facade strip) are cropped to at most
    1:2 so the pattern is not shrunk to a sliver; the swatch repeats, so a crop keeps all information."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if h > 2 * w:
        img = img.crop((0, 0, w, 2 * w))
    elif w > 2 * h:
        img = img.crop((0, 0, 2 * h, h))
    img.thumbnail((max_side, max_side))
    return img


FAMILIES = {"PISO": "floor", "PARED": "wall", "COLUMNA": "column", "FACHADA": "facade",
            "MADERA": "wood", "METAL": "metal", "TECHO": "ceiling"}


def family(label: str) -> str | None:
    for key, fam in FAMILIES.items():
        if key in label:
            return fam
    return None


def lightness(entry: dict) -> float:
    """Mean perceived lightness 0 (black) .. 1 (white), computed locally from the swatch."""
    a = np.asarray(load_ref(entry, 256).convert("L"), dtype=np.float32) / 255.0
    return float(a.mean())


def tone_notes(entries: list[dict]) -> dict[str, str]:
    """For families with several swatches (e.g. two wall textures), rank them light -> dark.
    Floors are excluded: they are chosen by zone (GENERAL vs ESPECIAL), not by tone."""
    groups: dict[str, list[tuple[float, str]]] = {}
    for e in entries:
        fam = family(e["label"])
        if fam and fam != "floor":
            groups.setdefault(fam, []).append((lightness(e), e["label"]))
    notes = {}
    for fam, items in groups.items():
        if len(items) < 2:
            continue
        items.sort(reverse=True)
        for rank, (l, lab) in enumerate(items):
            pos = "LIGHTEST" if rank == 0 else "DARKEST" if rank == len(items) - 1 else "MID-TONE"
            notes[lab] = f"{pos} of the {len(items)} {fam} options (lightness {l:.2f})"
    return notes


def load_ref(entry: dict, max_side: int = 1024) -> Image.Image:
    return load_path(config.BRAND_DB_DIR / entry["path"], max_side)


def retrieve(index: dict, query_vec: list[float], element: str | None = None,
             categories: list[str] | None = None, k: int = 2, exclude: set[str] = frozenset()) -> list[dict]:
    """Cosine-similarity search (only used when the library is larger than the reference budget)."""
    q = np.asarray(query_vec, dtype=np.float32)
    q /= np.linalg.norm(q) + 1e-9
    scored = []
    for e in index["entries"]:
        if e["path"] in exclude or (categories and e["category"] not in categories):
            continue
        s = 0.0
        if e.get("embedding"):
            v = np.asarray(e["embedding"], dtype=np.float32)
            s = float(q @ (v / (np.linalg.norm(v) + 1e-9)))
        if element and any(element.lower() in a.lower() or a.lower() in element.lower()
                           for a in e["applies_to"] + [e["label"]]):
            s += 0.15
        scored.append((s, e))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [e for _, e in scored[:k]]
