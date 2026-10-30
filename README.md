# Brand Render — SketchUp → branded photoreal render

Upload a SketchUp screenshot and get back a photorealistic render that uses the brand's
**materials, colors, furniture finishes and lighting**, while keeping the design (camera, walls, objects)
the same.

## How it works

```
brand_db/ (JPGs by category) ──► build_index ──► index.json  (tags + embeddings + brand brief)
                                                     │
sketch.jpg ─► 1 ANALYZE ─► 2 RETRIEVE refs ─► 3 RENDER ─► 4 VERIFY ─┬─ pass ─► render.jpg
              (Gemini)      (embeddings)      (Gemini      (edges +   │
                                               image)       Gemini QA)└─ fail ─► 5 REPAIR (feedback → re-render)
```

| Step | What it does | Model |
|---|---|---|
| Index (once) | Tags every brand image (subject, which elements it styles, materials, colors), embeds it, writes a brand brief | `VISION_MODEL`, `EMBED_MODEL` |
| Analyze | Lists every surface/object in the sketch | `VISION_MODEL` |
| Retrieve | Picks ≤13 references: 2 store photos + lighting + 1 per element | `EMBED_MODEL` |
| Render | Edits the sketch, sketch = image 1, explicit element→reference mapping, strict no-add rules | `RENDER_MODEL` |
| Verify | (a) edge-overlap score between sketch and render, no model involved (b) QA audit listing added/moved objects | `VISION_MODEL` |
| Repair | Re-renders with the concrete violations fed back; keeps the best attempt | `RENDER_MODEL` |

## Setup

```bash
git clone <your-repo> && cd brand-render
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # add your GEMINI_API_KEY
```

1. Put the brand JPGs in `brand_db/<category>/`. Default categories: `colors`, `textures`, `furniture`,
   `lighting`, `stores`. Any extra folder becomes a category.
2. Build the index: `python scripts/build_index.py`. Commit `brand_db/index.json`.
3. Run: `streamlit run app.py`
4. Batch: `python scripts/render_cli.py shots/*.jpg --out renders/`

## Deploy (Streamlit Community Cloud)
Push to GitHub → share.streamlit.io → New app → pick `app.py` → paste `GEMINI_API_KEY` into Secrets.
For a private brand DB, keep the repo private, or set `BRAND_DB_DIR` to a mounted path.

## Configuration (env vars)
| Var | Default | Note |
|---|---|---|
| `RENDER_MODEL` | `gemini-3-pro-image-preview` | Needs multi-reference image editing (up to 14 images) |
| `VISION_MODEL` | `gemini-2.5-flash` | Any Gemini text+vision model |
| `EMBED_MODEL` | `gemini-embedding-2-preview` | Multimodal embeddings |
| `OUTPUT_SIZE` | `2K` | `1K` / `2K` / `4K` (Gemini 3 image models) |

The sidebar lists the models your key can see, so you can swap models without code changes.

## Tips for good results
- **Better SketchUp exports give better renders**: hide axes and guides, use the "Shaded" style with
  visible edges, and model every object you want in the render. Anything left out of the model stays out of the render.
- **Name your brand images clearly** (e.g. `furniture/oak_center_table.jpg`): it helps the tagger.
- Use **close-up material photos** in `textures/` and **wide store photos** in `stores/`.
- Tune `EDGE_FIDELITY_MIN` in `config.py` on a few real sketches. Check the "Fidelity overlay" tab.
