"""✨~*~ TOTALLY RADICAL UI ~*~✨  (Word-2003 WordArt energy, made by a 13-year-old)

Everything visual lives here so app.py stays readable. `apply(serious=True)` turns it all off.
Uses only Streamlit's stable data-testid selectors. Animations respect prefers-reduced-motion.
"""
import random

import streamlit as st

FONTS = "https://fonts.googleapis.com/css2?family=Bangers&family=Comic+Neue:wght@400;700&family=Rubik+Glitch&family=Monoton&display=swap"

# Sparkly cursor (inline SVG emoji, no remote image)
CURSOR = ("url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='32' height='32'>"
          "<text y='24' font-size='24'>✨</text></svg>\") 4 4, auto")

CSS = """
<style>
@import url('%(fonts)s');

/* ---------- the whole page: rainbow + stars background ---------- */
[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(circle at 20px 20px, rgba(255,255,255,.9) 2px, transparent 3px) 0 0/48px 48px,
    radial-gradient(circle at 36px 8px, rgba(255,255,0,.8) 1.5px, transparent 2.5px) 0 0/48px 48px,
    linear-gradient(135deg,#ff00cc,#ffcc00,#00ffcc,#3366ff,#cc00ff,#ff00cc);
  background-size: 48px 48px, 48px 48px, 400%% 400%%;
  animation: rainbowBg 18s ease infinite;
  cursor: %(cursor)s;
}
@keyframes rainbowBg{0%%{background-position:0 0,0 0,0%% 50%%}50%%{background-position:0 0,0 0,100%% 50%%}100%%{background-position:0 0,0 0,0%% 50%%}}

/* main content = a "notebook page" so the real stuff is readable */
.block-container{
  background:#fffef2; border:6px dashed #ff00cc; border-radius:28px;
  box-shadow: 10px 10px 0 #3300ff, 20px 20px 0 #00e5ff;
  padding-top:2rem !important; margin-top:2rem;
}
html, body, p, label, li, input, textarea, .stMarkdown, [data-testid="stWidgetLabel"],
[data-testid="stCaptionContainer"], [data-testid="stExpander"] summary p, [data-baseweb="select"] div{
  font-family:'Comic Sans MS','Comic Neue',cursive !important; }
/* keep Streamlit's icon font intact (otherwise icons render as words like "arrow_right") */
[data-testid="stIconMaterial"], .material-symbols-rounded, [class*="material-symbols"]{
  font-family:'Material Symbols Rounded' !important; -webkit-text-stroke:0 !important; text-shadow:none !important; }

/* ---------- headings: Impact + outline + 3D shadow ---------- */
h1,h2,h3{ font-family:Impact,'Bangers',sans-serif !important; letter-spacing:2px; color:#ffea00 !important;
  -webkit-text-stroke:2px #ff0066; text-shadow:3px 3px 0 #3300ff,6px 6px 0 #00e5ff; }

/* ---------- WordArt title ---------- */
.wordart-wrap{ text-align:center; perspective:600px; margin:-.5rem 0 .2rem; }
.wordart{
  display:inline-block; font-family:Impact,'Bangers',sans-serif; font-size:clamp(38px,6.5vw,86px);
  line-height:1; letter-spacing:3px; text-transform:uppercase;
  background:linear-gradient(90deg,#ff0000,#ff8a00,#ffe600,#14e81e,#00b7ff,#8a2be2,#ff00cc,#ff0000);
  background-size:300%% 100%%; -webkit-background-clip:text; background-clip:text; color:transparent;
  -webkit-text-stroke:3px #1a0066;
  filter: drop-shadow(4px 4px 0 #ff00cc) drop-shadow(8px 8px 0 #00e5ff) drop-shadow(12px 12px 0 #1a0066);
  transform: rotateX(18deg) skewX(-10deg); animation: rainbowText 4s linear infinite, wobble 3s ease-in-out infinite;
}
@keyframes rainbowText{to{background-position:300%% 0}}
@keyframes wobble{0%%,100%%{transform:rotateX(18deg) skewX(-10deg) rotate(-2deg)}50%%{transform:rotateX(18deg) skewX(-10deg) rotate(2deg) scale(1.04)}}

.subtitle{ text-align:center; font-family:'Comic Sans MS','Comic Neue',cursive; font-size:20px; font-weight:700;
  color:#ff00cc; text-shadow:2px 2px 0 #ffea00; margin:.4rem 0 .8rem; }
.blink{ animation: blink 1s steps(2,start) infinite; color:#ff0000; font-weight:900; }
@keyframes blink{to{visibility:hidden}}

.marquee{ background:#000; color:#39ff14; font-family:'Courier New',monospace; font-weight:700; font-size:18px;
  border:4px ridge #ff00cc; border-radius:10px; padding:4px 0; overflow:hidden; white-space:nowrap; margin-bottom:12px; }
.marquee span{ display:inline-block; padding-left:100%%; animation: scroll 16s linear infinite; }
@keyframes scroll{to{transform:translateX(-100%%)}}

.construction{ text-align:center; font-family:Impact,'Bangers',sans-serif; font-size:18px; color:#000; letter-spacing:2px;
  padding:6px; border-radius:6px; margin:10px 0;
  background:repeating-linear-gradient(45deg,#ffd000 0 18px,#111 18px 36px); }
.construction b{ background:#ffd000; padding:2px 10px; border:3px solid #111; }

.counter{ text-align:center; margin-top:18px; font-family:'Courier New',monospace; }
.counter .digits{ display:inline-block; background:#000; color:#ff2d2d; padding:2px 8px; letter-spacing:4px;
  border:3px inset #888; font-size:20px; font-weight:700; }

/* ---------- buttons: bouncy candy ---------- */
.stButton>button, .stDownloadButton>button, [data-testid="stBaseButton-primary"], [data-testid="stBaseButton-secondary"]{
  font-family:Impact,'Bangers',sans-serif !important; font-size:22px !important; letter-spacing:2px;
  color:#fff !important; -webkit-text-stroke:1px #1a0066;
  background:linear-gradient(180deg,#ff6ad5 0%%,#ff00cc 45%%,#c3008f 100%%) !important;
  border:4px solid #1a0066 !important; border-radius:999px !important; box-shadow:0 6px 0 #1a0066 !important;
  transition: transform .15s;
}
.stButton>button:hover, .stDownloadButton>button:hover{ transform: rotate(-3deg) scale(1.08); animation: jiggle .3s infinite; }
.stButton>button:active{ transform: translateY(4px); box-shadow:0 2px 0 #1a0066 !important; }
@keyframes jiggle{0%%,100%%{transform:rotate(-3deg) scale(1.08)}50%%{transform:rotate(3deg) scale(1.08)}}

/* ---------- sidebar: neon lime notebook ---------- */
[data-testid="stSidebar"]{ background: repeating-linear-gradient(0deg,#ccff00 0 30px,#b8f000 30px 31px) !important;
  border-right:8px dotted #ff00cc; }
[data-testid="stSidebar"] *{ color:#1a0066; }

/* ---------- inputs ---------- */
[data-testid="stFileUploaderDropzone"]{ background:#e6fffa !important; border:5px dashed #00b7ff !important; border-radius:20px !important; }
[data-testid="stTextInput"] input, textarea{ background:#fff0fb !important; border:3px solid #ff00cc !important; border-radius:14px !important; }
[data-testid="stExpander"]{ border:4px double #8a2be2 !important; border-radius:16px !important; background:#f6f0ff; }
[data-testid="stTabs"] button p{ font-family:Impact,'Bangers',sans-serif !important; font-size:18px; color:#ff00cc; }
[data-testid="stImage"] img{ border:6px solid #fff; outline:5px dashed #ff00cc; border-radius:10px; box-shadow:8px 8px 0 #3300ff; }

@media (prefers-reduced-motion: reduce){ *{ animation:none !important; transition:none !important; } }
</style>
""" % {"fonts": FONTS, "cursor": CURSOR}

MARQUEE = ("✨✨ WELCOME 2 MY SUPER RENDER APP!!! ✨✨ ~*~ drop ur SketchUp pic ~*~ "
           "🌈 100%% NO HALLUCINATIONS (we hope lol) 🌈 ~*~ best viewed in Internet Explorer 6 ~*~ 💖💖💖")


def apply(serious: bool = False) -> None:
    if not serious:
        st.markdown(CSS, unsafe_allow_html=True)


def header(serious: bool = False) -> None:
    if serious:
        st.title("🏬 SketchUp → Brand Render")
        st.caption("Applies the brand's materials to a SketchUp screenshot without changing the design.")
        return
    st.markdown(f'<div class="marquee"><span>{MARQUEE.replace("%%", "%")}</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="wordart-wrap"><span class="wordart">✨ Render-O-Matic 3000 ✨</span></div>',
                unsafe_allow_html=True)
    st.markdown('<div class="subtitle">ur SketchUp → a REAL store!!!1! <span class="blink">★ NEW ★</span> '
                'no design changes, pinky promise 🤙</div>', unsafe_allow_html=True)


def construction(serious: bool = False) -> None:
    if not serious:
        st.markdown('<div class="construction"><b>🚧 UNDER CONSTRUCTION 🚧</b></div>', unsafe_allow_html=True)


def counter(serious: bool = False) -> None:
    if serious:
        return
    n = st.session_state.setdefault("_visitor", random.randint(1000, 99999))
    st.markdown(f'<div class="counter">u r visitor # <span class="digits">{n:06d}</span> '
                f'<br><small>(fake counter lol)</small></div>', unsafe_allow_html=True)


def label(serious: bool, normal: str, silly: str) -> str:
    """Pick button/heading text for the current mode."""
    return normal if serious else silly
