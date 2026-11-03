"""The agentic render workflow.

  sketch ─► 1. ANALYZE   vision model lists every visible surface/object in the SketchUp screenshot
            2. SELECT    pick material swatches (all of them if they fit the budget, else retrieval)
            3. ASSIGN    vision model maps each element -> one swatch label (or KEEP_AS_MODELED)
            4. RENDER    image model re-skins the sketch with that exact mapping, under strict rules
            5. VERIFY    a) edge-fidelity score (deterministic)  b) vision QA: geometry + materials
            6. REPAIR    if verification fails, re-render with the concrete issues fed back
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

from PIL import Image
from pydantic import BaseModel

from . import catalog, config, fidelity, gemini

KEEP = "KEEP_AS_MODELED"


# ---------- schemas ----------
class SceneElement(BaseModel):
    name: str          # snake_case, e.g. "back_wall", "floor_main_area", "column_left", "display_table"
    kind: str          # floor | wall | ceiling | column | furniture | fixture | facade | lighting | signage | other
    count: int
    location: str      # where in the frame
    modeled_color: str # color/material as drawn in SketchUp
    relative_tone: str # light | medium | dark, compared with OTHER elements of the same kind


class SceneAnalysis(BaseModel):
    elements: list[SceneElement]
    camera: str
    is_exterior: bool
    notes: str


class Assignment(BaseModel):
    element: str       # name from the analysis
    material_label: str  # exactly one of the provided labels, or KEEP_AS_MODELED
    reason: str


class AssignmentPlan(BaseModel):
    assignments: list[Assignment]


class QAIssue(BaseModel):
    issue: str         # concrete description
    kind: str          # geometry | object | material | artifact
    severity: str      # major | minor


class QAReport(BaseModel):
    issues: list[QAIssue]
    score: int         # 0-10 overall fidelity + material correctness


@dataclass
class RunResult:
    image: Image.Image | None = None
    attempts: list[dict] = field(default_factory=list)
    analysis: SceneAnalysis | None = None
    plan: AssignmentPlan | None = None
    references: list[dict] = field(default_factory=list)
    passed: bool = False


# ---------- prompts ----------
ANALYZE_PROMPT = (
    "This is a SketchUp screenshot of a clothing store (interior or facade). List EVERY visible "
    "architectural surface and object, grouping identical ones with a count. Split surfaces that are "
    "visibly different zones: e.g. walls modeled in a lighter vs a darker color -> separate elements; "
    "a floor area modeled in a different color/pattern from the rest -> separate element. For each element "
    "give its relative tone (light/medium/dark) compared with the other elements of the same kind. "
    "Use snake_case names. Describe the camera. Be exhaustive and literal; do not interpret or suggest."
)

ASSIGN_PROMPT = """IMAGE 1 is a SketchUp screenshot of a clothing store. The following images are the architect's
material swatches. Their labels (Spanish, authoritative about where each material goes) are:
{legend}

Scene elements found in IMAGE 1:
{elements}

For EVERY scene element choose exactly one material label from the list above, or {keep} if no swatch
applies (e.g. glass, mirrors, lamps, ceilings with no ceiling swatch). Rules:
- PISO = floor, PAREDES = walls, COLUMNAS = columns, FACHADA = exterior facade, MADERA = wood
  furniture/fixtures/panels, METAL = metal fixtures, racks, frames, structure, TECHO = ceiling.
- WALLS: there are several wall textures that differ in tone (see LIGHTEST/DARKEST in the list). The
  SketchUp model encodes which is which by color: walls modeled DARKER get the DARKEST wall texture,
  walls modeled LIGHTER get the LIGHTEST wall texture. Compare wall tones within IMAGE 1, not absolutely.
  If every wall has the same modeled tone, use the LIGHTEST wall texture for all of them.
- FLOORS: PISO GENERAL is the default for all floor. Use PISO ESPECIAL ONLY for a floor zone that is
  visibly distinct in IMAGE 1 (modeled in a different color/pattern, raised, or bounded area).
  If no such zone exists, do not use PISO ESPECIAL at all.
- Use the color/material shown in the SketchUp model as a hint (e.g. a brown-modeled table -> wood).
- FACHADA only applies if an exterior facade is visible.
- If unsure, use {keep}. Never invent elements."""

RENDER_RULES = """You are a photorealistic architectural visualizer. IMAGE 1 is a SketchUp model screenshot.
The other images are the architect's material swatches:
{legend}

TASK: Produce a photorealistic render of IMAGE 1 with the materials applied exactly as assigned below.

ABSOLUTE RULES (violating any of them is a failure):
1. Keep the EXACT same camera position, angle, focal length and framing as IMAGE 1.
2. Keep every wall, opening, column, ceiling and floor boundary exactly where it is. Same proportions.
3. Keep EXACTLY the objects in IMAGE 1: same number, positions, sizes and shapes.
4. Do NOT add anything: no people, no clothing, no mannequins, no plants, no decor, no logos, no text,
   no extra lights or furniture, unless it is already modeled in IMAGE 1.
5. Do NOT remove or merge anything.
6. Only change surface appearance (materials, textures, finishes) and make the lighting realistic.
7. IMAGE 1 is a SketchUp viewport: do NOT reproduce any SketchUp drawing artifacts. Remove the
   red/green/blue axis lines, dashed guide lines, section planes, selection highlights, the black edge
   outlines and any colored construction lines. Edges must appear as real physical material boundaries,
   never as drawn lines. The output must look like a photograph, with no lines painted over surfaces.
8. Apply each swatch at realistic real-world scale (tile/plank size, grain size), following the surface's
   perspective. Elements marked {keep} keep the color shown in IMAGE 1, rendered realistically.

MATERIAL ASSIGNMENT (material -> elements that use it):
{assignments}

STYLE BRIEF:
{brief}
{feedback}
Output only the final rendered image, same aspect ratio as IMAGE 1."""

QA_PROMPT = """IMAGE A is a SketchUp screenshot. IMAGE B is supposed to be the SAME scene rendered photorealistically.
Audit B against A and list concrete issues, each with a kind and a severity.

MAJOR (would make the render unusable):
- camera/viewpoint/framing changed; a wall, opening, column or floor boundary moved or changed shape
- an object ADDED that is not in A, or an object from A REMOVED, or clearly MOVED/RESIZED
- a large surface (floor, a wall, columns, main furniture) with the WRONG material vs the assignment
- drawn lines visible in B (axis lines, outline strokes copied from SketchUp)
MINOR (acceptable): small decor details, slight shape/proportion differences, lighting/shading, texture
scale or tone variations, small objects with slightly different materials.
Material/lighting changes are expected and are not geometry issues. Be fair: do not report MAJOR unless sure.

Required material assignment:
{assignments}
Material descriptions:
{materials}

Score 0-10: 10 = same design with the right materials, 7 = only minor issues, <5 = major issues."""

REPAIR_PROMPT = """IMAGE 1 is a SketchUp model screenshot (the ground truth for geometry and objects).
The material swatches are:
{legend}
The LAST image is a photorealistic render of IMAGE 1 that is ALMOST correct.

TASK: Edit the LAST image to fix ONLY these problems:
{issues}

Keep everything else in the LAST image exactly as it is (camera, lighting, materials, all other objects).
Do not add or remove anything else. No drawn lines. Material assignment for reference:
{assignments}
Output only the corrected image, same aspect ratio."""


# ---------- helpers ----------
def nearest_ratio(img: Image.Image) -> str:
    def val(s):
        a, b = s.split(":")
        return int(a) / int(b)
    r = img.width / img.height
    return min(config.SUPPORTED_RATIOS, key=lambda s: abs(val(s) - r))


def _describe(e: dict, notes: dict | None = None) -> str:
    d = f"{e['label']}: {e.get('material', '')}, {e.get('finish', '')}, colors {', '.join(e.get('colors', []))}"
    if notes and e["label"] in notes:
        d += f" [{notes[e['label']]}]"
    return d


def select_references(c, index: dict, analysis: SceneAnalysis, budget: int) -> list[dict]:
    entries = index["entries"]
    if len(entries) <= budget:
        return list(entries)
    chosen, used = [], set()
    for el in analysis.elements:
        q = gemini.embed_text(c, f"{el.name.replace('_', ' ')} ({el.kind}) material in a clothing store")
        for e in catalog.retrieve(index, q, element=el.kind, k=1, exclude=used):
            if len(chosen) < budget:
                chosen.append(e); used.add(e["path"])
    return chosen


# ---------- main entry ----------
def run(sketch: Image.Image, index: dict, *, render_model: str | None = None,
        vision_model: str | None = None, extra_instructions: str = "",
        max_refs: int = config.MAX_REFERENCES, max_retries: int = config.MAX_RETRIES,
        n_candidates: int = config.N_CANDIDATES,
        log: Callable[[str], None] = print) -> RunResult:
    c = gemini.client()
    res = RunResult()
    sketch = sketch.convert("RGB")
    if max(sketch.size) > 2048:  # keep requests small; output resolution is set separately
        sketch = sketch.copy(); sketch.thumbnail((2048, 2048))
    sketch_part = gemini.to_part(sketch, 95)

    log("1/5 Analyzing the SketchUp scene…")
    res.analysis = gemini.ask_json(c, [sketch_part, ANALYZE_PROMPT], SceneAnalysis, vision_model)
    log(f"   found {len(res.analysis.elements)} elements")

    log("2/5 Selecting material swatches…")
    refs = select_references(c, index, res.analysis, max_refs)
    res.references = refs
    ref_parts = [gemini.to_part(catalog.load_ref(e)) for e in refs]
    notes = catalog.tone_notes(refs)
    legend = "\n".join(f"- IMAGE {i + 2} = {_describe(e, notes)}" for i, e in enumerate(refs))
    log(f"   {len(refs)} swatches: {', '.join(e['label'] for e in refs)}")

    log("3/5 Assigning materials to elements…")
    elements_txt = "\n".join(f"- {el.name} ({el.kind}, x{el.count}, {el.location}, modeled as {el.modeled_color}, "
                             f"tone {el.relative_tone})"
                             for el in res.analysis.elements)
    res.plan = gemini.ask_json(
        c, [sketch_part, *ref_parts,
            ASSIGN_PROMPT.format(legend=legend, elements=elements_txt, keep=KEEP)], AssignmentPlan, vision_model)
    # Keep only the swatches that were actually assigned, renumbered, so unused ones can't leak in.
    known = {e["label"]: e for e in refs}
    pairs = [(a.element, a.material_label.strip().upper()) for a in res.plan.assignments]
    used = [e for e in refs if any(lab == e["label"] for _, lab in pairs)]
    res.references = used
    ref_parts = [gemini.to_part(catalog.load_ref(e)) for e in used]
    label_to_img = {e["label"]: i + 2 for i, e in enumerate(used)}
    legend = "\n".join(f"- IMAGE {i + 2} = {_describe(e, notes)}" for i, e in enumerate(used)) or "- (none)"
    groups: dict[str, list[str]] = {}
    for el, lab in pairs:
        groups.setdefault(lab if lab in known else KEEP, []).append(el.replace("_", " "))
    lines = [f"- {lab} (IMAGE {label_to_img[lab]}): {', '.join(els)}" for lab, els in groups.items() if lab != KEEP]
    if KEEP in groups:
        lines.append(f"- {KEEP} (keep the color shown in IMAGE 1, rendered realistically): {', '.join(groups[KEEP])}")
    assignments = "\n".join(lines)
    log("   " + assignments.replace("\n", "\n   "))

    ratio = nearest_ratio(sketch)
    brief = index.get("brand_brief", "")
    if extra_instructions:
        brief += f"\nADDITIONAL INSTRUCTIONS FROM USER: {extra_instructions}"
    materials_txt = "\n".join(f"   - {_describe(e, notes)}" for e in used)
    swatch_parts = list(ref_parts)

    def evaluate(img: Image.Image, label: str) -> dict:
        edge = fidelity.edge_fidelity(sketch, img)
        qa = gemini.ask_json(c, [sketch_part, gemini.to_part(img),
                                 QA_PROMPT.format(assignments=assignments, materials=materials_txt)],
                             QAReport, vision_model)
        major = [i for i in qa.issues if i.severity.lower() == "major"]
        passed = edge >= config.EDGE_FIDELITY_MIN and not major and qa.score >= config.QA_MIN_SCORE
        return {"image": img, "label": label, "edge_fidelity": round(edge, 3), "qa": qa.model_dump(),
                "major": [i.issue for i in major], "passed": passed}

    def rank(a: dict):
        return (a["passed"], -len(a["major"]), a["qa"]["score"] + 10 * a["edge_fidelity"])

    def report(a: dict):
        log(f"   {a['label']}: edge {a['edge_fidelity']:.2f} | QA {a['qa']['score']}/10 | "
            f"{len(a['major'])} major issue(s) | {'PASS' if a['passed'] else 'FAIL'}")

    # 4. First run: N independent candidates in parallel, keep the best (best-of-N beats sequential retries).
    prompt = RENDER_RULES.format(legend=legend, assignments=assignments, brief=brief, feedback="", keep=KEEP)
    log(f"4/5 Rendering {n_candidates} candidate(s) in parallel…")

    def candidate(i: int) -> dict:
        img = gemini.generate_image(c, [sketch_part, *swatch_parts, prompt], ratio, model=render_model)
        return evaluate(img, f"candidate {i + 1}")

    with ThreadPoolExecutor(max_workers=n_candidates) as ex:
        futures = [ex.submit(candidate, i) for i in range(n_candidates)]
        errors_ = []
        for f in futures:
            try:
                res.attempts.append(f.result())
            except Exception as e:  # one failed candidate should not kill the run
                errors_.append(e)
    if not res.attempts:
        raise errors_[0]
    log("5/5 Verified:")
    for a in res.attempts:
        report(a)
    best = max(res.attempts, key=rank)

    # 6. Repair: EDIT the best render to fix only its major issues; keep the edit only if it ranks higher.
    for r in range(max_retries):
        if best["passed"]:
            break
        issues = best["major"] or [i["issue"] for i in best["qa"]["issues"]][:5]
        if best["edge_fidelity"] < config.EDGE_FIDELITY_MIN:
            issues.append("geometry/camera drifted from IMAGE 1: realign walls, openings and objects to IMAGE 1")
        log(f"↺ Repair {r + 1}: fixing {len(issues)} issue(s) on the best render…")
        rp = REPAIR_PROMPT.format(legend=legend, assignments=assignments,
                                  issues="\n".join(f"- {x}" for x in issues))
        try:
            img = gemini.generate_image(c, [sketch_part, *swatch_parts, gemini.to_part(best["image"], 95), rp],
                                        ratio, model=render_model)
        except Exception as e:
            log(f"   repair failed: {gemini.explain(e)[:200]}")
            break
        a = evaluate(img, f"repair {r + 1}")
        res.attempts.append(a)
        report(a)
        if rank(a) > rank(best):
            best = a
        else:
            log("   repair not better, keeping previous best")

    res.image, res.passed = best["image"], best["passed"]
    return res
