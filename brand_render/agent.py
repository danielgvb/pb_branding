"""The agentic render workflow.

  sketch.jpg ──► 1. ANALYZE   vision model lists every visible element (walls, floor, racks, tables...)
                 2. RETRIEVE  for each element, pick the best brand references (embeddings + tags)
                 3. RENDER    image model edits the sketch using refs + brand brief, under strict rules
                 4. VERIFY    a) edge-fidelity score (deterministic)  b) vision-model QA vs the sketch
                 5. REPAIR    if verification fails, re-render with the specific issues fed back
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from PIL import Image
from pydantic import BaseModel

from . import catalog, config, fidelity, gemini


# ---------- schemas ----------
class SceneElement(BaseModel):
    name: str          # snake_case, e.g. "back_wall", "floor", "center_table", "wall_shelving"
    kind: str          # floor | wall | ceiling | furniture | fixture | lighting | signage | other
    count: int
    location: str      # where in the frame


class SceneAnalysis(BaseModel):
    elements: list[SceneElement]
    camera: str        # viewpoint/lens description
    has_people: bool
    notes: str


class QAReport(BaseModel):
    geometry_preserved: bool       # same walls, openings, camera, proportions
    objects_match: bool            # same objects, same count, same positions; nothing added/removed
    branding_applied: bool         # materials/colors/lighting follow the references
    added_or_changed: list[str]    # concrete list of violations
    score: int                     # 0-10 overall


@dataclass
class RunResult:
    image: Image.Image | None = None
    attempts: list[dict] = field(default_factory=list)
    analysis: SceneAnalysis | None = None
    references: list[dict] = field(default_factory=list)
    passed: bool = False


# ---------- prompts ----------
ANALYZE_PROMPT = (
    "This is a SketchUp screenshot of a retail clothing store interior. List EVERY visible "
    "architectural surface and object (group identical ones with a count). Use snake_case names. "
    "Describe the camera. Be exhaustive and literal; do not interpret or suggest anything."
)

RENDER_RULES = """You are a photorealistic architectural visualizer. IMAGE 1 is a SketchUp model screenshot.
The other images are the brand's official design references.

TASK: Produce a photorealistic render of IMAGE 1 with the brand's materials, colors, finishes and lighting applied.

ABSOLUTE RULES (violating any of them is a failure):
1. Keep the EXACT same camera position, angle, focal length and framing as IMAGE 1.
2. Keep every wall, opening, column, ceiling and floor boundary exactly where it is. Same proportions.
3. Keep EXACTLY the objects in IMAGE 1: same number, same positions, same sizes, same shapes.
4. Do NOT add anything: no people, no clothing, no mannequins, no plants, no decor, no logos, no text,
   no extra lights or furniture — unless it is already modeled in IMAGE 1.
5. Do NOT remove or merge anything.
6. Only change surface appearance: materials, textures, colors, finishes, and the lighting/mood.

MATERIAL ASSIGNMENT (element -> reference images to take its look from):
{assignments}

BRAND BRIEF:
{brief}
{feedback}
Output only the final rendered image, same aspect ratio as IMAGE 1."""

QA_PROMPT = (
    "IMAGE A is a SketchUp screenshot. IMAGE B is supposed to be the SAME scene rendered "
    "photorealistically with brand materials. Strictly audit B against A: camera, walls, openings, "
    "every object (count, position, size, shape). List every object/feature in B that was ADDED, "
    "REMOVED, MOVED or RESHAPED compared with A. Material/color/lighting changes are EXPECTED and are "
    "not violations. Then judge whether the branding looks consistent with the brief: {brief}"
)


# ---------- helpers ----------
def nearest_ratio(img: Image.Image) -> str:
    r = img.width / img.height
    def val(s):
        a, b = s.split(":")
        return int(a) / int(b)
    return min(config.SUPPORTED_RATIOS, key=lambda s: abs(val(s) - r))


KIND_TO_CATEGORIES = {
    "floor": ["textures", "colors", "stores"],
    "wall": ["textures", "colors", "stores"],
    "ceiling": ["textures", "lighting", "stores"],
    "furniture": ["furniture", "textures"],
    "fixture": ["furniture", "textures"],
    "lighting": ["lighting"],
    "signage": ["colors", "stores"],
}


def plan_references(c, index: dict, analysis: SceneAnalysis, sketch: Image.Image, budget: int):
    """Assign refs per element. Returns (ordered unique ref entries, assignment text)."""
    chosen: list[dict] = []
    used: set[str] = set()
    lines = []
    cats = set(index["categories"])

    def take(entries):
        out = []
        for e in entries:
            if e["path"] not in used and len(chosen) < budget:
                used.add(e["path"]); chosen.append(e)
            if e["path"] in used:
                out.append(next(i for i, x in enumerate(chosen) if x["path"] == e["path"]) + 2)
        return out

    # 1) Global look: store photos most similar to the whole sketch + lighting.
    sketch_vec = gemini.embed_image(c, sketch)
    if "stores" in cats:
        take(catalog.retrieve(index, sketch_vec, categories=["stores"], k=2))
    if "lighting" in cats:
        ids = take(catalog.retrieve(index, sketch_vec, categories=["lighting"], k=1))
        lines.append(f"- overall lighting & mood -> images {ids}")

    # 2) Per element.
    for el in analysis.elements:
        if len(chosen) >= budget:
            break
        el_cats = [x for x in KIND_TO_CATEGORIES.get(el.kind, list(cats)) if x in cats] or None
        q = gemini.embed_text(c, f"{el.name.replace('_', ' ')} ({el.kind}) in a clothing store")
        ids = take(catalog.retrieve(index, q, element=el.name, categories=el_cats, k=1, exclude=set()))
        if ids:
            lines.append(f"- {el.name} (x{el.count}, {el.location}) -> image {ids[0]}")

    store_ids = [i + 2 for i, e in enumerate(chosen) if e["category"] == "stores"]
    if store_ids:
        lines.insert(0, f"- overall style reference (do NOT copy their layout) -> images {store_ids}")
    return chosen, "\n".join(lines)


# ---------- main entry ----------
def run(sketch: Image.Image, index: dict, *, render_model: str | None = None,
        vision_model: str | None = None, extra_instructions: str = "",
        max_refs: int = config.MAX_REFERENCES, max_retries: int = config.MAX_RETRIES,
        log: Callable[[str], None] = print) -> RunResult:
    c = gemini.client()
    res = RunResult()
    sketch = sketch.convert("RGB")

    log("1/4 Analyzing the SketchUp scene…")
    res.analysis = gemini.ask_json(c, [gemini.to_part(sketch), ANALYZE_PROMPT], SceneAnalysis, vision_model)
    log(f"   found {len(res.analysis.elements)} element types")

    log("2/4 Selecting brand references…")
    refs, assignments = plan_references(c, index, res.analysis, sketch, max_refs)
    res.references = refs
    log(f"   {len(refs)} references selected")

    ratio = nearest_ratio(sketch)
    brief = index.get("brand_brief", "")
    if extra_instructions:
        brief += f"\nADDITIONAL INSTRUCTIONS FROM USER: {extra_instructions}"
    feedback = ""

    for attempt in range(max_retries + 1):
        log(f"3/4 Rendering (attempt {attempt + 1})…")
        prompt = RENDER_RULES.format(assignments=assignments or "- use the references as you see fit",
                                     brief=brief, feedback=feedback)
        contents = [gemini.to_part(sketch, 95)] + [gemini.to_part(catalog.load_ref(e)) for e in refs] + [prompt]
        img = gemini.generate_image(c, contents, ratio, model=render_model)

        log("4/4 Verifying fidelity…")
        edge = fidelity.edge_fidelity(sketch, img)
        qa = gemini.ask_json(c, [gemini.to_part(sketch), gemini.to_part(img), QA_PROMPT.format(brief=brief[:1500])],
                             QAReport, vision_model)
        passed = (edge >= config.EDGE_FIDELITY_MIN and qa.geometry_preserved
                  and qa.objects_match and not qa.added_or_changed)
        res.attempts.append({"image": img, "edge_fidelity": round(edge, 3), "qa": qa.model_dump(), "passed": passed})
        log(f"   edge fidelity {edge:.2f} | QA score {qa.score}/10 | {'PASS' if passed else 'FAIL'}")

        if passed:
            break
        issues = qa.added_or_changed or ["geometry drifted from the sketch"]
        if edge < config.EDGE_FIDELITY_MIN:
            issues.append(f"lines/geometry do not match the sketch (edge score {edge:.2f})")
        feedback = ("\nA PREVIOUS ATTEMPT FAILED REVIEW. Fix these problems:\n- " + "\n- ".join(issues) + "\n")

    best = max(res.attempts, key=lambda a: (a["passed"], a["qa"]["score"] + 10 * a["edge_fidelity"]))
    res.image, res.passed = best["image"], best["passed"]
    return res
