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
- If unsure, use {keep}. Never invent elements.
PROJECT RULES:
{context}"""

EDIT_HEADER = """IMAGE EDITING TASK, not a redesign. The first image is a SketchUp screenshot of an approved design.
Edit it into a photorealistic image by changing ONLY the surface materials listed below.
The geometry, camera and every object must stay exactly as in the screenshot."""

EDIT_RULES = """WHAT YOU MAY CHANGE (and nothing else):
A. The surface material of the elements listed in the MATERIAL MAP, using the matching swatch above,
   at realistic real-world scale, following each surface's perspective.
B. Photorealism: realistic shading, soft shadows, reflections and material sheen from the light sources
   that already exist in the screenshot.
C. Remove SketchUp viewport artifacts: axis lines (red/green/blue), guides, black outline strokes.

MATERIAL MAP (swatch -> elements):
{assignments}

EVERYTHING ELSE IS FROZEN:
- Same camera, viewpoint, focal length, framing and aspect ratio. Do not zoom, crop, rotate or extend.
- Same walls, openings, columns, ceiling and floor layout, in exactly the same pixel positions.
- Same objects: same count, position, size, shape and modeled color. Nothing added, removed, moved,
  merged, restyled or "improved". No people, products, clothing, plants, decor, logos or text added.
- Elements not in the MATERIAL MAP keep their modeled color and shape; only make them look photoreal.
- When in doubt about anything, leave it exactly as it is in the screenshot.

PROJECT RULES:
{context}
{feedback}
FINAL CHECK before answering: the output must overlay the screenshot line-for-line. Output only the edited image."""

QA_PROMPT = """IMAGE A is a SketchUp screenshot. IMAGE B is supposed to be the SAME scene rendered photorealistically.
Audit B against A and list concrete issues, each with a kind and a severity.

MAJOR (would make the render unusable):
- camera/viewpoint/framing changed; a wall, opening, column or floor boundary moved or changed shape
- an object ADDED that is not in A, or an object from A REMOVED, or clearly MOVED/RESIZED
- a large surface (floor, a wall, columns, main furniture) with the WRONG material vs the assignment
- drawn lines visible in B (axis lines, outline strokes copied from SketchUp)
- an element that is NOT in the material assignment changed color, material or style noticeably
- any violation of the PROJECT RULES below
MINOR (acceptable): small decor details, slight shape/proportion differences, lighting/shading, texture
scale or tone variations, small objects with slightly different materials.
Material/lighting changes are expected and are not geometry issues. Be fair: do not report MAJOR unless sure.

Required material assignment:
{assignments}
Material descriptions:
{materials}
PROJECT RULES:
{context}

Score 0-10: 10 = same design with the right materials, 7 = only minor issues, <5 = major issues."""

REPAIR_RULES = """The LAST image is an edit of the SketchUp screenshot that is ALMOST correct.
Edit the LAST image again to fix ONLY these problems, using the SketchUp screenshot as ground truth:
{issues}

Everything else in the LAST image stays exactly as it is (camera, materials, lighting, all other objects).
Do not add, remove or move anything else. No drawn lines.
MATERIAL MAP for reference:
{assignments}
PROJECT RULES:
{context}
Output only the corrected image, same framing."""


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
        vision_model: str | None = None, qa_model: str | None = None, extra_instructions: str = "",
        max_refs: int = config.MAX_REFERENCES, max_retries: int = config.MAX_RETRIES,
        n_candidates: int = config.N_CANDIDATES, context: str | None = None,
        log: Callable[[str], None] = print) -> RunResult:
    c = gemini.client()
    res = RunResult()
    context = (context if context is not None else config.load_context()) or "- (none)"
    if extra_instructions:
        context += f"\n- Notes for this image: {extra_instructions}"
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
            ASSIGN_PROMPT.format(legend=legend, elements=elements_txt, keep=KEEP, context=context)], AssignmentPlan, vision_model)
    # Keep only the swatches that were actually assigned, renumbered, so unused ones can't leak in.
    known = {e["label"]: e for e in refs}
    pairs = [(a.element, a.material_label.strip().upper()) for a in res.plan.assignments]
    used = [e for e in refs if any(lab == e["label"] for _, lab in pairs)]
    res.references = used
    ref_parts = [gemini.to_part(catalog.load_ref(e)) for e in used]
    legend = "\n".join(f"- IMAGE {i + 2} = {_describe(e, notes)}" for i, e in enumerate(used)) or "- (none)"
    groups: dict[str, list[str]] = {}
    for el, lab in pairs:
        groups.setdefault(lab if lab in known else KEEP, []).append(el.replace("_", " "))
    lines = [f"- SWATCH {lab}: {', '.join(els)}" for lab, els in groups.items() if lab != KEEP]
    if KEEP in groups:
        lines.append(f"- NOT CHANGED (keep modeled color/shape, only photoreal): {', '.join(groups[KEEP])}")
    assignments = "\n".join(lines)
    log("   " + assignments.replace("\n", "\n   "))

    ratio = nearest_ratio(sketch)
    materials_txt = "\n".join(f"   - {_describe(e, notes)}" for e in used)
    # Interleave a text label right before each image: more reliable than "IMAGE n" references.
    labeled = ["SCREENSHOT TO EDIT (ground truth for camera, geometry and objects):", sketch_part]
    for e, part in zip(used, ref_parts):
        labeled += [f"SWATCH {e['label']}: {_describe(e, notes).split(': ', 1)[-1]}", part]

    def evaluate(img: Image.Image, label: str) -> dict:
        edge = fidelity.edge_fidelity(sketch, img)
        qa = gemini.ask_json(c, [sketch_part, gemini.to_part(img),
                                 QA_PROMPT.format(assignments=assignments, materials=materials_txt, context=context)],
                             QAReport, qa_model or vision_model)
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
    prompt = EDIT_RULES.format(assignments=assignments, context=context, feedback="")
    log(f"4/5 Rendering {n_candidates} candidate(s) in parallel…")

    def candidate(i: int) -> dict:
        img = gemini.generate_image(c, [EDIT_HEADER, *labeled, prompt], ratio, model=render_model)
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
            issues.append("geometry/camera drifted from the screenshot: realign walls, openings and objects to it")
        log(f"↺ Repair {r + 1}: fixing {len(issues)} issue(s) on the best render…")
        rp = REPAIR_RULES.format(assignments=assignments, context=context,
                                 issues="\n".join(f"- {x}" for x in issues))
        try:
            img = gemini.generate_image(c, [*labeled, "LAST IMAGE (edit to correct):", gemini.to_part(best["image"], 95), rp],
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
