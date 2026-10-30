"""Index the brand database once: python scripts/build_index.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from brand_render import catalog, config  # noqa: E402

if __name__ == "__main__":
    idx = catalog.build_index(progress=lambda p, m: print(f"[{p:5.0%}] {m}"))
    print(f"\nIndexed {len(idx['entries'])} images -> {config.INDEX_PATH}\n\nBRAND BRIEF:\n{idx['brand_brief']}")
