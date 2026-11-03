"""Central settings. Model IDs are overridable via env vars or the Streamlit sidebar,
because Gemini model names change often (preview -> GA)."""
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BRAND_DB_DIR = Path(os.getenv("BRAND_DB_DIR", ROOT / "brand_db"))
INDEX_PATH = BRAND_DB_DIR / "index.json"

# Preferred models, best first. The app picks the first one your API key actually has
# (Gemini names change: preview -> stable -> retired), so defaults never point at a dead model.
RENDER_PREFS = ["gemini-3-pro-image", "gemini-3-pro-image-preview",          # Nano Banana Pro
                "gemini-3.1-flash-image", "gemini-3.1-flash-image-preview",  # Nano Banana 2
                "gemini-2.5-flash-image"]
VISION_PREFS = ["gemini-flash-latest"]   # Google's auto-updated alias; else the newest *-flash on your key
QA_PREFS = ["gemini-pro-latest"]         # else the newest *-pro on your key

# Env vars override everything.
RENDER_MODEL = os.getenv("RENDER_MODEL", RENDER_PREFS[0])
VISION_MODEL = os.getenv("VISION_MODEL", "gemini-flash-latest")
QA_MODEL = os.getenv("QA_MODEL", "gemini-pro-latest")


def _version(name: str) -> float:
    m = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
    return float(m.group(1)) if m else 0.0


def pick(prefs: list[str], available: list[str], env: str | None = None, family: str = "flash") -> str:
    """Env override > first preferred model the key has > newest available model of `family`."""
    if env and os.getenv(env):
        return os.environ[env]
    for m in prefs:
        if m in available:
            return m
    fam = [m for m in available if family in m and "lite" not in m]
    if fam:
        return max(fam, key=lambda m: (_version(m), "preview" not in m))
    return available[0] if available else prefs[0]


# Multimodal embedding model used to pick the most relevant references.
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-2-preview")

# Categories = sub-folders of brand_db/. Add more folders and they are picked up automatically.
DEFAULT_CATEGORIES = ["colors", "textures", "furniture", "lighting", "stores"]

MAX_REFERENCES = 10          # sketch + refs must stay under the model's limit (14 for 3 Pro Image)
MAX_RETRIES = 2              # repair edits of the best render when QA fails
N_CANDIDATES = 2             # independent renders on the first run (in parallel); best one is kept
QA_MIN_SCORE = 7             # QA score needed to pass (plus no MAJOR issues and edge fidelity)
EDGE_FIDELITY_MIN = 0.75     # min structural overlap (0-1); a 5% camera shift scored ~0.71 in tests. Tune on real sketches.
OUTPUT_SIZE = os.getenv("OUTPUT_SIZE", "2K")  # 1K / 2K / 4K

SUPPORTED_RATIOS = ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]


def api_key() -> str | None:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("GEMINI_API_KEY")
    except Exception:
        return None
