import hmac
import io

import streamlit as st
from PIL import Image

from brand_render import agent, catalog, config, fidelity, gemini

st.set_page_config(page_title="Brand Render", page_icon="🏬", layout="wide")

# ---------- password gate (set APP_PASSWORD in Streamlit Secrets; protects your Gemini key) ----------
def _password_ok() -> bool:
    expected = st.secrets.get("APP_PASSWORD") if hasattr(st, "secrets") else None
    if not expected:
        return True  # no password configured -> open (fine for local use)
    if st.session_state.get("auth_ok"):
        return True
    pw = st.text_input("Password", type="password")
    if pw and hmac.compare_digest(pw, str(expected)):
        st.session_state["auth_ok"] = True
        st.rerun()
    elif pw:
        st.error("Wrong password")
    return False


try:
    _gate = _password_ok()
except Exception:  # no secrets.toml at all (local run)
    _gate = True
if not _gate:
    st.stop()

st.title("🏬 SketchUp → Brand Render")
st.caption("Applies the brand's materials, colors and lighting to a SketchUp screenshot — without changing the design.")

# ---------- sidebar ----------
with st.sidebar:
    st.header("Settings")
    if not config.api_key():
        st.error("Set GEMINI_API_KEY in .streamlit/secrets.toml or as an env var.")
        st.stop()

    @st.cache_data(ttl=3600)
    def _models():
        return gemini.list_models(gemini.client())

    models = _models()
    # Defaults always first, so a stray alphabetical pick never becomes the default.
    img_models = [config.RENDER_MODEL] + [m for m in models if "image" in m and m != config.RENDER_MODEL]
    txt_models = [config.VISION_MODEL] + [m for m in models if "image" not in m and m != config.VISION_MODEL]
    render_model = st.selectbox("Render model", img_models,
                                index=0)
    vision_model = st.selectbox("Analysis / QA model", txt_models,
                                index=0)
    max_refs = st.slider("Max reference images", 3, 13, config.MAX_REFERENCES)
    max_retries = st.slider("Max repair attempts", 0, 3, config.MAX_RETRIES)

    st.divider()
    st.subheader("Brand database")
    index = catalog.load_index()
    if catalog.is_stale(index):  # missing, or images added/renamed -> build automatically
        bar = st.progress(0.0, "Indexing brand swatches (first run)…")
        try:
            index = catalog.build_index(progress=lambda p, msg: bar.progress(p, msg))
        except Exception as e:
            st.error(gemini.explain(e))
            st.stop()
        bar.empty()
    if index:
        st.success(f"{len(index['entries'])} swatches indexed")
        with st.expander("Swatches"):
            tones = catalog.tone_notes(index["entries"])
            for e in index["entries"]:
                note = f" · _{tones[e['label']]}_" if e["label"] in tones else ""
                st.markdown(f"**{e['label']}** — {e['material']} ({e['finish']}){note}")
        with st.expander("Style brief"):
            st.markdown(index["brand_brief"])
    if st.button("🔄 Rebuild index", use_container_width=True):
        bar = st.progress(0.0, "Indexing…")
        index = catalog.build_index(progress=lambda p, msg: bar.progress(p, msg))
        st.rerun()

if not index:
    st.stop()

# ---------- main ----------
up = st.file_uploader("SketchUp screenshot (JPG/PNG)", type=["jpg", "jpeg", "png"])
extra = st.text_input("Optional notes (e.g. 'evening lighting', 'use the oak table finish')", "")

if up:
    sketch = Image.open(up).convert("RGB")
    if st.session_state.get("upload_name") != up.name:   # new file -> clear old result
        st.session_state.pop("res", None)
        st.session_state["upload_name"] = up.name
    if "res" not in st.session_state:
        st.image(sketch, caption="Input (SketchUp)", width=520)

    if st.button("✨ Render with brand style", type="primary"):
        logbox = st.status("Running…", expanded=True)
        try:
            res = agent.run(sketch, index, render_model=render_model, vision_model=vision_model,
                            extra_instructions=extra, max_refs=max_refs, max_retries=max_retries,
                            log=logbox.write)
        except Exception as e:
            logbox.update(label="Failed", state="error")
            st.error(gemini.explain(e))
            st.stop()
        logbox.update(label="Done" if res.passed else "Done — review flagged issues",
                      state="complete" if res.passed else "error")
        st.session_state["res"] = res
        st.session_state["sketch"] = sketch

res = st.session_state.get("res")
if up and res:
    sketch = st.session_state["sketch"]
    col1, col2 = st.columns(2)
    col1.image(sketch, caption="Input", use_container_width=True)
    col2.image(res.image, caption="Render" + ("" if res.passed else " (did not pass all checks)"),
               use_container_width=True)
    buf = io.BytesIO(); res.image.save(buf, "JPEG", quality=95)
    st.download_button("⬇️ Download render", buf.getvalue(), "render.jpg", "image/jpeg")

    t1, t2, t3, t4, t5 = st.tabs(["Fidelity overlay", "Attempts", "Material assignment", "Swatches used",
                                  "Scene analysis"])
    with t1:
        st.caption("Red = SketchUp lines drawn over the render. Lines should sit on the same edges.")
        st.image(fidelity.overlay(sketch, res.image), use_container_width=True)
    with t2:
        for i, a in enumerate(res.attempts, 1):
            st.markdown(f"**Attempt {i}** — edge fidelity `{a['edge_fidelity']}` · QA `{a['qa']['score']}/10` · "
                        f"{'✅ pass' if a['passed'] else '❌ fail'}")
            c1, c2 = st.columns([1, 1])
            c1.image(a["image"], use_container_width=True)
            c2.json(a["qa"])
    with t3:
        st.table([a.model_dump() for a in res.plan.assignments])
    with t4:
        cols = st.columns(4)
        for i, e in enumerate(res.references):
            cols[i % 4].image(catalog.load_ref(e, 400), caption=f"IMAGE {i + 2} · {e['label']}",
                              use_container_width=True)
    with t5:
        st.json(res.analysis.model_dump())
