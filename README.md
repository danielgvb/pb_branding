# Brand Render — SketchUp → branded photoreal render

Upload a SketchUp screenshot and get back a photorealistic render that uses the brand's
**materials, colors, furniture finishes and lighting**, while keeping the design (camera, walls, objects)
the same.

## How it works

```
brand_db/*.jpg  (architect's material swatches, FILE NAME = where it goes)
      │ build index (once): describe each swatch + style brief -> brand_db/index.json
      ▼
sketch.jpg ─► 1 ANALYZE ─► 2 SELECT ─► 3 ASSIGN ─► 4 RENDER ─► 5 VERIFY ─┬─ pass ─► render.jpg
              elements     swatches    element→     Gemini      edges +   └─ fail ─► 6 REPAIR
                                       swatch       image       Gemini QA            (feedback → re-render)
```

| Step | What it does |
|---|---|
| Index (once) | Describes each swatch (material, colors, finish). The **file name is the label** and is authoritative. |
| Analyze | Lists every surface/object in the SketchUp screenshot, incl. distinct floor/wall zones |
| Select | All swatches if they fit (≤ `MAX_REFERENCES`), otherwise retrieval by embeddings |
| Assign | Maps each element → one swatch label, or `KEEP_AS_MODELED` (glass, lamps, anything without a swatch) |
| Render | Sketch = image 1 + **only the assigned swatches**, explicit element→image mapping, strict no-add rules |
| Verify | (a) model-free edge-overlap score (b) QA: geometry + "does each element show its material?" |
| Repair | Re-renders with the concrete violations; keeps the best attempt |

## Brand library (`brand_db/`)
Drop swatch images directly in `brand_db/` with descriptive Spanish names. Current set:

| File | Used for |
|---|---|
| `PISO GENERAL` | main floor |
| `PISO ESPECIAL` | a visibly distinct floor zone in the model (only if one exists) |
| `TEXTURA PAREDES`, `TEXTURA PAREDES 2` | walls |
| `TEXTURA COLUMNAS` | columns |
| `MADERA NOGAL` | wood furniture / panels |
| `METAL BLANCO MICROTEXTURIZADO` | metal fixtures, racks, frames |
| `FACHADA` | exterior facade (exterior shots only) |

Naming keywords the assigner understands: PISO, PAREDES, COLUMNAS, FACHADA, MADERA, METAL, TECHO (ceiling).
Sub-folders also work (`brand_db/<category>/...`). **After adding/renaming images, rebuild the index**
(sidebar button or `python scripts/build_index.py`); the app warns when the index is stale.

## Setup

```bash
git clone <your-repo> && cd brand-render
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # add your GEMINI_API_KEY
```

1. Put the swatches in `brand_db/` (see above).
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
- **Model zones in SketchUp with different colors** (e.g. the special-floor area). That is how the assigner tells PISO ESPECIAL from PISO GENERAL, or the two wall textures apart.
- To tell the two wall textures apart, rename them for their use (e.g. `TEXTURA PAREDES FONDO.jpg`, `TEXTURA PAREDES LATERALES.jpg`).
- Tune `EDGE_FIDELITY_MIN` in `config.py` on a few real sketches. Check the "Fidelity overlay" tab.
