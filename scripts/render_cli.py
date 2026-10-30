"""Batch / CLI rendering: python scripts/render_cli.py input.jpg [more.jpg ...] --out renders/"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from brand_render import agent, catalog  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", default="renders")
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    index = catalog.load_index() or sys.exit("Build the index first: python scripts/build_index.py")
    out = Path(a.out); out.mkdir(exist_ok=True)
    for f in a.inputs:
        res = agent.run(Image.open(f), index, extra_instructions=a.notes)
        stem = Path(f).stem
        res.image.save(out / f"{stem}_render.jpg", quality=95)
        (out / f"{stem}_report.json").write_text(json.dumps(
            {"passed": res.passed, "attempts": [{k: v for k, v in x.items() if k != "image"} for x in res.attempts],
             "references": [e["path"] for e in res.references]}, indent=2))
        print(f"{f}: {'PASS' if res.passed else 'REVIEW'} -> {out / (stem + '_render.jpg')}")
