"""Central settings. Model IDs are overridable via env vars or the Streamlit sidebar,
because Gemini model names change often (preview -> GA)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BRAND_DB_DIR = Path(os.getenv("BRAND_DB_DIR", ROOT / "brand_db"))
INDEX_PATH = BRAND_DB_DIR / "index.json"

# Image model that edits the SketchUp screenshot (needs multi-reference support).
RENDER_MODEL = os.getenv("RENDER_MODEL", "gemini-3-pro-image-preview")
# Text/vision model used for scene analysis, brand brief, and QA.
VISION_MODEL = os.getenv("VISION_MODEL", "gemini-2.5-flash")
# Multimodal embedding model used to pick the most relevant references.
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-2-preview")

# Categories = sub-folders of brand_db/. Add more folders and they are picked up automatically.
DEFAULT_CATEGORIES = ["colors", "textures", "furniture", "lighting", "stores"]

MAX_REFERENCES = 10          # sketch + refs must stay under the model's limit (14 for 3 Pro Image)
MAX_RETRIES = 2              # re-render attempts when QA fails
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
