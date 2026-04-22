"""
app.py  ·  Visual Inspection Agent  ·  Ultra Premium Interactive UI v4
"""
import os, io, tempfile
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from defect_detector import detect_and_annotate
from llava_agent import analyze_image_with_llava
from report_generator import generate_pdf_report
from utils import save_upload_to_temp, cleanup_temp, compute_verdict

st.set_page_config(page_title="Visual Inspection Agent", page_icon="🔬", layout="wide")

# ═══════════════════════════════════════════════════════
#  GLOBAL CSS + ANIMATIONS
# ═══════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800;900&display=swap');
*,[class*="css"]{font-family:'Outfit',sans-serif!important;box-sizing:border-box}
#MainMenu,footer,header{visibility:hidden}
.block-container{padding-top:.4rem!important;max-width:1400px}

/* ── Animated mesh background ── */
.stApp{background:#07071a;color:#e2e8f0;overflow-x:hidden}
.stApp::before{
  content:"";position:fixed;inset:0;z-index:-2;
  background:
    radial-gradient(ellipse at 10% 15%,rgba(139,92,246,.28) 0%,transparent 50%),
    radial-gradient(ellipse at 90% 85%,rgba(6,182,212,.22) 0%,transparent 50%),
    radial-gradient(ellipse at 50% 50%,rgba(236,72,153,.1)  0%,transparent 65%),
    linear-gradient(160deg,#07071a 0%,#0b0b24 60%,#07071a 100%);
}

/* ── Floating orbs ── */
.orb{position:fixed;border-radius:50%;filter:blur(80px);pointer-events:none;z-index:-1;animation:orbFloat ease-in-out infinite}
.orb1{width:420px;height:420px;background:rgba(139,92,246,.12);top:-80px;left:-100px;animation-duration:14s}
.orb2{width:340px;height:340px;background:rgba(6,182,212,.1);bottom:-60px;right:-80px;animation-duration:18s;animation-delay:-6s}
.orb3{width:260px;height:260px;background:rgba(236,72,153,.08);top:40%;left:60%;animation-duration:22s;animation-delay:-10s}
@keyframes orbFloat{0%,100%{transform:translate(0,0) scale(1)}33%{transform:translate(30px,-40px) scale(1.05)}66%{transform:translate(-20px,30px) scale(.97)}}

/* ── Animated grid overlay ── */
.stApp::after{
  content:"";position:fixed;inset:0;z-index:-1;pointer-events:none;
  background-image:linear-gradient(rgba(129,140,248,.03) 1px,transparent 1px),
                   linear-gradient(90deg,rgba(129,140,248,.03) 1px,transparent 1px);
  background-size:60px 60px;
  animation:gridPan 25s linear infinite;
}
@keyframes gridPan{0%{background-position:0 0}100%{background-position:60px 60px}}

/* ── HERO ── */
.hero{text-align:center;padding:2.6rem 1rem 2rem;position:relative;z-index:1}
.hero-tag{
  display:inline-block;
  background:linear-gradient(135deg,rgba(139,92,246,.3),rgba(6,182,212,.2));
  border:1px solid rgba(139,92,246,.5);border-radius:50px;
  padding:5px 22px;font-size:.72rem;font-weight:700;letter-spacing:3px;
  color:#a78bfa;text-transform:uppercase;margin-bottom:1rem;
  animation:tagPulse 3s ease-in-out infinite;
}
@keyframes tagPulse{0%,100%{box-shadow:0 0 0 0 rgba(139,92,246,.3)}50%{box-shadow:0 0 0 8px rgba(139,92,246,0)}}
.hero-title{
  font-size:3.4rem;font-weight:900;line-height:1.1;
  background:linear-gradient(135deg,#a78bfa 0%,#818cf8 25%,#38bdf8 55%,#34d399 80%,#a78bfa 100%);
  background-size:200% auto;
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
  animation:textShine 4s linear infinite;
  filter:drop-shadow(0 0 40px rgba(139,92,246,.5));
  margin-bottom:.6rem;
}
@keyframes textShine{0%{background-position:0% 50%}100%{background-position:200% 50%}}
.hero-sub{color:#94a3b8;font-size:.97rem;max-width:560px;margin:0 auto 1.3rem;line-height:1.7}
.hero-line{
  width:90px;height:3px;margin:0 auto;border-radius:99px;
  background:linear-gradient(90deg,#818cf8,#38bdf8,#818cf8);
  background-size:200% auto;animation:lineSweep 2.5s linear infinite;
}
@keyframes lineSweep{0%{background-position:0%}100%{background-position:200%}}

/* ── UPLOAD ZONE — styled Streamlit uploader ── */
div[data-testid="stFileUploader"]{
  border:2px dashed rgba(129,140,248,.45)!important;border-radius:20px!important;
  padding:1.8rem 1rem!important;text-align:center!important;
  background:linear-gradient(135deg,rgba(129,140,248,.05),rgba(6,182,212,.04))!important;
  transition:all .4s ease!important;position:relative!important;
  margin-bottom:1.2rem!important;
}
div[data-testid="stFileUploader"]:hover{
  border-color:rgba(129,140,248,.85)!important;
  background:rgba(129,140,248,.1)!important;
  box-shadow:0 0 40px rgba(129,140,248,.2)!important;
}
div[data-testid="stFileUploader"] label{color:#e2e8f0!important;font-weight:700!important;font-size:1rem!important}
[data-testid="stFileUploaderDropzone"]{
  background:transparent!important;border:none!important;
  cursor:pointer!important;
}
[data-testid="stFileUploaderDropzoneInstructions"] span{
  color:#a78bfa!important;font-size:.88rem!important;font-weight:600!important;
}
[data-testid="stFileUploaderDropzoneInstructions"] small{
  color:#64748b!important;
}
button[data-testid="baseButton-secondary"]{
  background:linear-gradient(135deg,rgba(99,102,241,.3),rgba(6,182,212,.2))!important;
  border:1px solid rgba(129,140,248,.5)!important;border-radius:10px!important;
  color:#a78bfa!important;font-weight:700!important;
}

/* ── SECTION HEADER ── */
.shdr{display:flex;align-items:center;gap:12px;margin:1.8rem 0 .9rem}
.shdr-text{font-size:.98rem;font-weight:700;color:#e2e8f0;white-space:nowrap;display:flex;align-items:center;gap:7px}
.shdr-line{flex:1;height:1px;background:linear-gradient(90deg,rgba(129,140,248,.5),transparent)}

/* ── GLASS CARD ── */
.gcard{
  background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.09);
  border-radius:18px;padding:1.2rem 1.4rem;margin-bottom:1rem;
  backdrop-filter:blur(14px);
  box-shadow:0 8px 32px rgba(0,0,0,.35),inset 0 1px 0 rgba(255,255,255,.06);
  transition:transform .25s,box-shadow .25s;
}
.gcard:hover{transform:translateY(-2px);box-shadow:0 16px 48px rgba(0,0,0,.45)}

/* ── ANIMATED METRIC CARDS ── */
.mgrid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:.5rem 0 1.2rem}
.mcard{
  border-radius:18px;padding:1.1rem 1rem;text-align:center;
  border:1px solid rgba(255,255,255,.08);
  position:relative;overflow:hidden;
  transition:transform .25s,box-shadow .25s;
  cursor:default;
}
.mcard::after{
  content:"";position:absolute;inset:0;border-radius:18px;
  background:linear-gradient(135deg,rgba(255,255,255,.06),transparent);
  pointer-events:none;
}
.mcard:hover{transform:translateY(-5px) scale(1.02)}
.mcard.T{background:linear-gradient(135deg,rgba(99,102,241,.25),rgba(56,189,248,.15));border-color:rgba(99,102,241,.45);box-shadow:0 0 35px rgba(99,102,241,.25)}
.mcard.H{background:linear-gradient(135deg,rgba(239,68,68,.25),rgba(220,38,38,.14));border-color:rgba(239,68,68,.45);box-shadow:0 0 35px rgba(239,68,68,.22)}
.mcard.M{background:linear-gradient(135deg,rgba(251,146,60,.25),rgba(234,88,12,.14));border-color:rgba(251,146,60,.45);box-shadow:0 0 35px rgba(251,146,60,.18)}
.mcard.L{background:linear-gradient(135deg,rgba(250,204,21,.22),rgba(202,138,4,.12));border-color:rgba(250,204,21,.38);box-shadow:0 0 35px rgba(250,204,21,.14)}
.mcard-icon{font-size:1.9rem;margin-bottom:6px;display:block}
.mcard-val{font-size:2.8rem;font-weight:900;line-height:1;margin-bottom:4px}
.mcard-lbl{font-size:.7rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;opacity:.6}
.mcard.T .mcard-val{color:#818cf8}.mcard.H .mcard-val{color:#f87171}
.mcard.M .mcard-val{color:#fb923c}.mcard.L .mcard-val{color:#fbbf24}
/* Animated ring */
.mcard-ring{
  position:absolute;top:-20px;right:-20px;width:80px;height:80px;
  border-radius:50%;border:2px solid;opacity:.15;
  animation:ringPulse 2s ease-in-out infinite;
}
.mcard.T .mcard-ring{border-color:#818cf8}.mcard.H .mcard-ring{border-color:#f87171}
.mcard.M .mcard-ring{border-color:#fb923c}.mcard.L .mcard-ring{border-color:#fbbf24}
@keyframes ringPulse{0%,100%{transform:scale(1);opacity:.15}50%{transform:scale(1.3);opacity:.05}}

/* ── DETECTION CARDS ── */
.dcard{
  border-radius:14px;padding:1rem 1.1rem;margin-bottom:9px;
  border-left:4px solid;display:flex;align-items:flex-start;gap:12px;
  transition:transform .22s,box-shadow .22s;position:relative;overflow:hidden;
}
.dcard::before{content:"";position:absolute;inset:0;border-radius:14px;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.03),transparent);
  transform:translateX(-100%);animation:cardShimmer 4s ease infinite;}
@keyframes cardShimmer{0%{transform:translateX(-100%)}100%{transform:translateX(100%)}}
.dcard:hover{transform:translateX(5px)}
.dcard.HIGH  {background:rgba(239,68,68,.09);border-color:#ef4444;box-shadow:0 0 20px rgba(239,68,68,.1)}
.dcard.MEDIUM{background:rgba(251,146,60,.09);border-color:#fb923c;box-shadow:0 0 20px rgba(251,146,60,.1)}
.dcard.LOW   {background:rgba(250,204,21,.07);border-color:#fbbf24;box-shadow:0 0 20px rgba(250,204,21,.08)}
.dnum{width:32px;height:32px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:.8rem;flex-shrink:0;margin-top:2px}
.dcard.HIGH   .dnum{background:rgba(239,68,68,.3);color:#fca5a5}
.dcard.MEDIUM .dnum{background:rgba(251,146,60,.3);color:#fdba74}
.dcard.LOW    .dnum{background:rgba(250,204,21,.25);color:#fde68a}
.dtitle{font-size:.97rem;font-weight:700;color:#f1f5f9;margin-bottom:5px;display:flex;align-items:center;gap:8px}
.dmeta{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:7px}
.chip{font-size:.72rem;font-weight:600;padding:2px 9px;border-radius:20px;background:rgba(255,255,255,.07);color:#cbd5e1;border:1px solid rgba(255,255,255,.1)}
.sbadge{font-size:.68rem;font-weight:800;padding:2px 9px;border-radius:20px;letter-spacing:1px}
.sbadge.HIGH  {background:rgba(239,68,68,.25);color:#fca5a5;border:1px solid rgba(239,68,68,.4)}
.sbadge.MEDIUM{background:rgba(251,146,60,.25);color:#fdba74;border:1px solid rgba(251,146,60,.4)}
.sbadge.LOW   {background:rgba(250,204,21,.2) ;color:#fde68a;border:1px solid rgba(250,204,21,.3)}
.cbar-wrap{height:7px;background:rgba(255,255,255,.07);border-radius:99px;overflow:hidden;margin-top:6px}
.cbar{height:100%;border-radius:99px}
.cbar.HIGH  {background:linear-gradient(90deg,#ef4444,#fca5a5);box-shadow:0 0 8px #ef4444}
.cbar.MEDIUM{background:linear-gradient(90deg,#fb923c,#fdba74);box-shadow:0 0 8px #fb923c}
.cbar.LOW   {background:linear-gradient(90deg,#fbbf24,#fde68a);box-shadow:0 0 8px #fbbf24}

/* ── VERDICT ── */
.vdict{border-radius:22px;padding:2.2rem 1.5rem;text-align:center;margin:.8rem 0;position:relative;overflow:hidden}
.vdict::before{content:"";position:absolute;inset:0;border-radius:22px;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.05),transparent);
  animation:vShimmer 3s ease infinite;}
@keyframes vShimmer{0%{transform:translateX(-100%)}100%{transform:translateX(100%)}}
.vdict.PASS   {background:linear-gradient(135deg,rgba(16,185,129,.22),rgba(5,150,105,.14));border:1px solid rgba(16,185,129,.45);box-shadow:0 0 80px rgba(16,185,129,.25)}
.vdict.WARNING{background:linear-gradient(135deg,rgba(251,146,60,.22),rgba(217,119,6,.14));border:1px solid rgba(251,146,60,.45);box-shadow:0 0 80px rgba(251,146,60,.22)}
.vdict.FAIL   {background:linear-gradient(135deg,rgba(239,68,68,.25),rgba(185,28,28,.14));border:1px solid rgba(239,68,68,.5); box-shadow:0 0 80px rgba(239,68,68,.28)}
.vicon{font-size:3.2rem;margin-bottom:.3rem;animation:vIconBounce 1.5s ease-in-out infinite}
@keyframes vIconBounce{0%,100%{transform:scale(1)}50%{transform:scale(1.15)}}
.vlbl{font-size:2.5rem;font-weight:900;letter-spacing:6px;text-transform:uppercase}
.vdict.PASS    .vlbl{color:#34d399;text-shadow:0 0 30px rgba(52,211,153,.5)}
.vdict.WARNING .vlbl{color:#fb923c;text-shadow:0 0 30px rgba(251,146,60,.5)}
.vdict.FAIL    .vlbl{color:#f87171;text-shadow:0 0 30px rgba(248,113,113,.5)}
.vsub{font-size:.9rem;color:#94a3b8;margin-top:.45rem}

/* ── AI PANEL ── */
.aipanel{
  background:linear-gradient(135deg,rgba(129,140,248,.09),rgba(192,132,252,.07),rgba(56,189,248,.07));
  border:1px solid rgba(129,140,248,.25);border-radius:18px;padding:1.5rem 1.8rem;
  position:relative;overflow:hidden;
}
.aipanel::before{content:'"';position:absolute;top:-14px;left:16px;font-size:9rem;
  color:rgba(129,140,248,.08);font-family:Georgia,serif;line-height:1;pointer-events:none}
.aipanel::after{
  content:"";position:absolute;inset:0;border-radius:18px;
  background:linear-gradient(90deg,transparent,rgba(129,140,248,.06),transparent);
  animation:aiShimmer 5s ease infinite;
}
@keyframes aiShimmer{0%{transform:translateX(-100%)}100%{transform:translateX(100%)}}
.aihead{display:flex;align-items:center;gap:10px;margin-bottom:1rem;position:relative;z-index:1}
.aihtitle{font-size:1rem;font-weight:700;color:#a78bfa}
.aibody{font-size:.93rem;color:#cbd5e1;line-height:1.8;white-space:pre-wrap;position:relative;z-index:1}

/* ── PROGRESS BAR (custom) ── */
.prog-wrap{margin-bottom:16px}
.prog-header{display:flex;justify-content:space-between;margin-bottom:5px;font-size:.87rem;font-weight:600}
.prog-track{height:12px;background:rgba(255,255,255,.06);border-radius:99px;overflow:hidden;position:relative}
.prog-fill{height:100%;border-radius:99px;position:relative;overflow:hidden}
.prog-fill::after{content:"";position:absolute;inset:0;
  background:linear-gradient(90deg,transparent 0%,rgba(255,255,255,.3) 50%,transparent 100%);
  animation:progShine 2s ease infinite;}
@keyframes progShine{0%{transform:translateX(-100%)}100%{transform:translateX(100%)}}

/* ── TABS ── */
button[data-baseweb="tab"]{
  background:rgba(255,255,255,.04)!important;border-radius:12px!important;
  font-weight:700!important;color:#64748b!important;
  border:1px solid rgba(255,255,255,.07)!important;margin-right:8px!important;
  padding:.45rem 1.1rem!important;transition:all .2s!important;
}
button[data-baseweb="tab"]:hover{color:#e2e8f0!important;background:rgba(255,255,255,.08)!important}
button[data-baseweb="tab"][aria-selected="true"]{
  background:linear-gradient(135deg,rgba(99,102,241,.35),rgba(6,182,212,.25))!important;
  color:#e2e8f0!important;border-color:rgba(99,102,241,.5)!important;
  box-shadow:0 0 20px rgba(99,102,241,.3)!important;
}

/* ── FILE UPLOADER (see upload zone section above) ── */

/* ── SIDEBAR ── */
[data-testid="stSidebar"]{background:rgba(5,5,16,.97)!important;border-right:1px solid rgba(129,140,248,.14)!important}
.sdot{width:11px;height:11px;border-radius:50%;flex-shrink:0}
.sdot.H{background:#ef4444;box-shadow:0 0 8px #ef4444}.sdot.M{background:#fb923c;box-shadow:0 0 8px #fb923c}.sdot.L{background:#fbbf24;box-shadow:0 0 8px #fbbf24}
.sleg{display:flex;align-items:center;gap:10px;padding:8px 12px;border-radius:10px;margin-bottom:6px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.07);font-size:.86rem}

/* ── DOWNLOAD BTN ── */
.stDownloadButton>button{
  background:linear-gradient(135deg,#6366f1,#8b5cf6,#06b6d4)!important;
  color:#fff!important;border:none!important;border-radius:14px!important;
  padding:.75rem 2rem!important;font-size:1rem!important;font-weight:700!important;
  width:100%!important;box-shadow:0 4px 30px rgba(99,102,241,.5)!important;
  transition:all .25s!important;letter-spacing:.5px!important;
}
.stDownloadButton>button:hover{opacity:.85!important;transform:translateY(-2px)!important;box-shadow:0 8px 40px rgba(99,102,241,.6)!important}

/* ── FEATURE CARDS ── */
.feat-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:1.6rem 0 2rem}
.feat-card{
  border-radius:16px;padding:1.1rem .9rem;text-align:center;
  border:1px solid rgba(255,255,255,.08);
  background:rgba(255,255,255,.04);
  backdrop-filter:blur(12px);
  position:relative;overflow:hidden;
  transition:transform .25s,box-shadow .25s;
  cursor:default;
}
.feat-card::before{
  content:"";position:absolute;inset:0;border-radius:16px;
  background:linear-gradient(135deg,rgba(255,255,255,.06),transparent 60%);
  pointer-events:none;
}
.feat-card:hover{transform:translateY(-5px)}
.feat-card.c1{border-color:rgba(139,92,246,.35);box-shadow:0 0 28px rgba(139,92,246,.18)}
.feat-card.c2{border-color:rgba(6,182,212,.35); box-shadow:0 0 28px rgba(6,182,212,.15)}
.feat-card.c3{border-color:rgba(236,72,153,.32); box-shadow:0 0 28px rgba(236,72,153,.13)}
.feat-card.c4{border-color:rgba(52,211,153,.32); box-shadow:0 0 28px rgba(52,211,153,.13)}
.feat-card.c5{border-color:rgba(251,191,36,.3);  box-shadow:0 0 28px rgba(251,191,36,.12)}
.feat-card:hover.c1{box-shadow:0 0 50px rgba(139,92,246,.35)}
.feat-card:hover.c2{box-shadow:0 0 50px rgba(6,182,212,.3)}
.feat-card:hover.c3{box-shadow:0 0 50px rgba(236,72,153,.28)}
.feat-card:hover.c4{box-shadow:0 0 50px rgba(52,211,153,.28)}
.feat-card:hover.c5{box-shadow:0 0 50px rgba(251,191,36,.25)}
.feat-icon{font-size:2rem;margin-bottom:7px;display:block}
.feat-name{font-size:.82rem;font-weight:800;letter-spacing:.5px;margin-bottom:4px}
.feat-desc{font-size:.7rem;color:#64748b;line-height:1.5}
.feat-card.c1 .feat-name{color:#a78bfa}
.feat-card.c2 .feat-name{color:#38bdf8}
.feat-card.c3 .feat-name{color:#f472b6}
.feat-card.c4 .feat-name{color:#34d399}
.feat-card.c5 .feat-name{color:#fbbf24}
/* animated dot */
.feat-dot{
  position:absolute;top:9px;right:9px;width:7px;height:7px;
  border-radius:50%;animation:dotBlink 2s ease-in-out infinite;
}
.feat-card.c1 .feat-dot{background:#a78bfa;box-shadow:0 0 6px #a78bfa;animation-delay:0s}
.feat-card.c2 .feat-dot{background:#38bdf8;box-shadow:0 0 6px #38bdf8;animation-delay:.4s}
.feat-card.c3 .feat-dot{background:#f472b6;box-shadow:0 0 6px #f472b6;animation-delay:.8s}
.feat-card.c4 .feat-dot{background:#34d399;box-shadow:0 0 6px #34d399;animation-delay:1.2s}
.feat-card.c5 .feat-dot{background:#fbbf24;box-shadow:0 0 6px #fbbf24;animation-delay:1.6s}
@keyframes dotBlink{0%,100%{opacity:.4;transform:scale(1)}50%{opacity:1;transform:scale(1.4)}}

/* ── LANDING ── */
.landing{
  text-align:center;padding:4rem 2rem;margin-top:.8rem;
  border:2px dashed rgba(129,140,248,.22);border-radius:24px;
  background:radial-gradient(ellipse at center,rgba(129,140,248,.08) 0%,transparent 70%);
  position:relative;overflow:hidden;
}
.landing::before{
  content:"";position:absolute;inset:0;border-radius:24px;
  background:linear-gradient(90deg,transparent,rgba(129,140,248,.06),transparent);
  animation:shimmer 4s ease infinite;
}
@keyframes shimmer{0%{transform:translateX(-100%)}100%{transform:translateX(100%)}}
.licon{font-size:4.5rem;display:block;animation:iconPulse 2.5s ease-in-out infinite}
@keyframes iconPulse{0%,100%{filter:drop-shadow(0 0 16px rgba(139,92,246,.5)) drop-shadow(0 0 40px rgba(139,92,246,.2))}50%{filter:drop-shadow(0 0 30px rgba(139,92,246,.9)) drop-shadow(0 0 80px rgba(139,92,246,.4))}}
.ltitle{font-size:1.45rem;font-weight:800;color:#a78bfa;margin:.9rem 0 .45rem}
.lsub{color:#64748b;font-size:.93rem;line-height:1.7;max-width:420px;margin:0 auto}
.steps{display:flex;gap:14px;justify-content:center;flex-wrap:wrap;margin:2rem 0 0}
.step{
  display:flex;flex-direction:column;align-items:center;gap:6px;
  padding:14px 20px;background:rgba(255,255,255,.04);border-radius:16px;
  border:1px solid rgba(255,255,255,.08);min-width:105px;text-align:center;
  transition:transform .25s,box-shadow .25s;
}
.step:hover{transform:translateY(-5px);box-shadow:0 10px 30px rgba(129,140,248,.2);border-color:rgba(129,140,248,.35)}
.stepicon{font-size:1.7rem}.steplbl{font-size:.74rem;font-weight:700;color:#64748b;letter-spacing:.5px}

/* Image label */
.imlbl{font-size:.73rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#475569;text-align:center;padding:5px 0 3px}
div[data-testid="stSpinner"]>div{color:#818cf8!important}
hr{border-color:rgba(255,255,255,.07)!important}
</style>

<!-- Floating orbs -->
<div class="orb orb1"></div>
<div class="orb orb2"></div>
<div class="orb orb3"></div>
""", unsafe_allow_html=True)


# ── YOLO model ─────────────────────────────────────────
@st.cache_resource(show_spinner="⚡ Loading YOLOv8…")
def load_yolo():
    return YOLO("yolov8n.pt")
load_yolo()

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Outfit", color="#94a3b8"),
    margin=dict(l=10, r=10, t=40, b=10),
)

# ── SIDEBAR ────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:.4rem 0 1rem">
        <div style="font-size:2.8rem;animation:iconPulse 2.5s ease-in-out infinite;display:block">🔬</div>
        <div style="font-size:.95rem;font-weight:800;background:linear-gradient(135deg,#a78bfa,#38bdf8);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-top:5px">
            Visual Inspection Agent
        </div>
        <div style="font-size:.72rem;color:#475569;margin-top:2px">AI-Powered QC System</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**⚙️ SEVERITY SCALE**")
    for cls,lbl,rule,col in [("H","HIGH","≥ 0.80","#f87171"),("M","MEDIUM","≥ 0.50","#fb923c"),("L","LOW","< 0.50","#fbbf24")]:
        st.markdown(f'<div class="sleg"><div class="sdot {cls}"></div><div><b style="color:{col}">{lbl}</b><br><span style="font-size:.75rem;color:#475569">confidence {rule}</span></div></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**🏁 VERDICT LOGIC**")
    for ico,v,d,c in [("❌","FAIL","Any HIGH detected","#f87171"),("⚠️","WARNING","MEDIUM, no HIGH","#fb923c"),("✅","PASS","Only LOW or none","#34d399")]:
        st.markdown(f'<div style="padding:5px 9px;border-radius:8px;margin-bottom:5px;background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.06);font-size:.83rem">{ico} <b style="color:{c}">{v}</b> <span style="color:#475569">— {d}</span></div>', unsafe_allow_html=True)
    st.markdown("---")
    enable_llava = st.toggle("🦙 Enable LLaVA Analysis", value=True, help="Requires Ollama + llava model")
    st.markdown("---")
    for t,i in [("YOLOv8n","🧠"),("OpenCV","🖼️"),("LLaVA / Ollama","🦙"),("FPDF2","📄"),("Plotly","📊")]:
        st.markdown(f'<div style="font-size:.82rem;color:#64748b;padding:2px 0;display:flex;gap:8px">{i}<span>{t}</span></div>', unsafe_allow_html=True)

# ── HERO ───────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-tag">🔬 Manufacturing QC · Multi-Agent AI System</div>
    <div class="hero-title">Visual Inspection Agent</div>
    <div class="hero-sub">Upload any product image — YOLOv8 detects defects with severity grading,
        interactive charts visualise results, LLaVA provides expert AI analysis,
        and a full PDF report is ready to download.</div>
    <div class="hero-line"></div>
</div>""", unsafe_allow_html=True)

# ── FEATURE CARDS ──────────────────────────────────────
st.markdown("""
<div class="feat-grid">
  <div class="feat-card c1">
    <div class="feat-dot"></div>
    <span class="feat-icon">🧠</span>
    <div class="feat-name">YOLOv8</div>
    <div class="feat-desc">Real-time object &amp; defect detection with bounding boxes</div>
  </div>
  <div class="feat-card c2">
    <div class="feat-dot"></div>
    <span class="feat-icon">🖼️</span>
    <div class="feat-name">OpenCV</div>
    <div class="feat-desc">Image annotation, colour-coded severity overlays</div>
  </div>
  <div class="feat-card c3">
    <div class="feat-dot"></div>
    <span class="feat-icon">🦙</span>
    <div class="feat-name">LLaVA AI</div>
    <div class="feat-desc">Multimodal language model for expert QC analysis</div>
  </div>
  <div class="feat-card c4">
    <div class="feat-dot"></div>
    <span class="feat-icon">📊</span>
    <div class="feat-name">Plotly</div>
    <div class="feat-desc">Interactive charts — donut, bar, scatter &amp; radar</div>
  </div>
  <div class="feat-card c5">
    <div class="feat-dot"></div>
    <span class="feat-icon">📄</span>
    <div class="feat-name">FPDF2</div>
    <div class="feat-desc">Auto-generated downloadable PDF inspection report</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── UPLOAD ZONE ─────────────────────────────────────────
st.markdown('<div class="shdr"><div class="shdr-text">📂 Upload Product Image</div><div class="shdr-line"></div></div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drop your image here — JPG · JPEG · PNG", type=["jpg","jpeg","png"], label_visibility="visible"
)

# ═══════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════
if uploaded_file:
    file_bytes = uploaded_file.read()
    suffix     = os.path.splitext(uploaded_file.name)[-1] or ".jpg"
    tmp_path   = save_upload_to_temp(file_bytes, suffix=suffix)

    try:
        with st.spinner("⚡ Running YOLOv8 defect detection…"):
            annotated_rgb, report = detect_and_annotate(tmp_path)

        high   = sum(1 for d in report if d["severity"] == "HIGH")
        medium = sum(1 for d in report if d["severity"] == "MEDIUM")
        low    = sum(1 for d in report if d["severity"] == "LOW")
        total  = len(report)

        tab1, tab2, tab3, tab4 = st.tabs(["🖼️  Inspection", "📊  Analytics", "📋  Detections", "🦙  AI Report"])

        # ── TAB 1 ─────────────────────────────────────
        with tab1:
            st.markdown(f"""
            <div class="mgrid">
              <div class="mcard T"><div class="mcard-ring"></div><span class="mcard-icon">🔍</span><div class="mcard-val">{total}</div><div class="mcard-lbl">Total Detected</div></div>
              <div class="mcard H"><div class="mcard-ring"></div><span class="mcard-icon">🔴</span><div class="mcard-val">{high}</div><div class="mcard-lbl">High Severity</div></div>
              <div class="mcard M"><div class="mcard-ring"></div><span class="mcard-icon">🟠</span><div class="mcard-val">{medium}</div><div class="mcard-lbl">Medium Severity</div></div>
              <div class="mcard L"><div class="mcard-ring"></div><span class="mcard-icon">🟡</span><div class="mcard-val">{low}</div><div class="mcard-lbl">Low Severity</div></div>
            </div>""", unsafe_allow_html=True)

            st.markdown('<div class="shdr"><div class="shdr-text">🖼️ Image Comparison</div><div class="shdr-line"></div></div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2, gap="large")
            with c1:
                st.markdown('<div class="imlbl">📷 Original Image</div>', unsafe_allow_html=True)
                st.image(Image.open(io.BytesIO(file_bytes)), use_container_width=True)
            with c2:
                st.markdown('<div class="imlbl">🎯 Annotated — Defects Highlighted</div>', unsafe_allow_html=True)
                st.image(annotated_rgb, use_container_width=True)

            st.markdown('<div class="shdr"><div class="shdr-text">🏁 Overall Verdict</div><div class="shdr-line"></div></div>', unsafe_allow_html=True)
            verdict_label, _, verdict_icon = compute_verdict(report)
            vsub = {"FAIL":"Critical defects found — immediate corrective action required.","WARNING":"Moderate issues detected — further inspection recommended.","PASS":"No critical defects found — product meets quality standards."}[verdict_label]
            st.markdown(f"""
            <div class="vdict {verdict_label}">
                <div class="vicon">{verdict_icon}</div>
                <div class="vlbl">{verdict_label}</div>
                <div class="vsub">{vsub}</div>
            </div>""", unsafe_allow_html=True)

        # ── TAB 2: ANALYTICS ──────────────────────────
        with tab2:
            if total == 0:
                st.info("No detections — nothing to chart.")
            else:
                ca, cb = st.columns(2, gap="large")

                with ca:
                    fig_donut = go.Figure(go.Pie(
                        labels=["HIGH","MEDIUM","LOW"], values=[high,medium,low], hole=.65,
                        marker=dict(colors=["#ef4444","#fb923c","#fbbf24"],line=dict(color="#07071a",width=3)),
                        textinfo="label+percent",textfont=dict(size=13,color="#e2e8f0"),
                        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
                        pull=[.06 if high else 0, .04 if medium else 0, .02 if low else 0],
                    ))
                    fig_donut.update_layout(**PLOTLY_LAYOUT, height=310,
                        title=dict(text="Severity Distribution",font=dict(size=14,color="#a78bfa")),
                        showlegend=True, legend=dict(font=dict(color="#94a3b8"),bgcolor="rgba(0,0,0,0)"),
                        annotations=[dict(text=f"<b>{total}</b><br><span style='font-size:10px'>found</span>",
                            x=.5,y=.5,font=dict(size=17,color="#e2e8f0"),showarrow=False)],
                    )
                    st.plotly_chart(fig_donut, use_container_width=True)

                with cb:
                    names  = [f"#{i+1} {d['object'][:11]}" for i,d in enumerate(report)]
                    confs  = [d["confidence"] for d in report]
                    colors = [{"HIGH":"#ef4444","MEDIUM":"#fb923c","LOW":"#fbbf24"}[d["severity"]] for d in report]
                    fig_bar = go.Figure(go.Bar(
                        x=confs, y=names, orientation="h",
                        marker=dict(color=colors,opacity=.88,line=dict(width=0)),
                        text=[f"{c:.0%}" for c in confs], textposition="outside",
                        textfont=dict(color="#94a3b8",size=11),
                        hovertemplate="<b>%{y}</b><br>Confidence: %{x:.2%}<extra></extra>",
                    ))
                    fig_bar.update_layout(**PLOTLY_LAYOUT, height=310,
                        title=dict(text="Confidence per Detection",font=dict(size=14,color="#a78bfa")),
                        xaxis=dict(range=[0,1.15],tickformat=".0%",gridcolor="rgba(255,255,255,.05)",zeroline=False),
                        yaxis=dict(gridcolor="rgba(255,255,255,.04)"), bargap=.28,
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                # Animated progress bars
                st.markdown('<div class="shdr"><div class="shdr-text">📈 Severity Breakdown</div><div class="shdr-line"></div></div>', unsafe_allow_html=True)
                for lbl, cnt, col, glow in [
                    ("🔴 HIGH",   high,   "#ef4444", "rgba(239,68,68,.5)"),
                    ("🟠 MEDIUM", medium, "#fb923c", "rgba(251,146,60,.45)"),
                    ("🟡 LOW",    low,    "#fbbf24", "rgba(250,204,21,.4)"),
                ]:
                    pct = int(cnt/total*100) if total else 0
                    st.markdown(f"""
                    <div class="prog-wrap">
                        <div class="prog-header">
                            <span style="color:#e2e8f0">{lbl}</span>
                            <span style="color:#475569">{cnt} detection{'s' if cnt!=1 else ''} &nbsp;·&nbsp; {pct}%</span>
                        </div>
                        <div class="prog-track">
                            <div class="prog-fill" style="width:{pct}%;background:linear-gradient(90deg,{col},{col}aa);box-shadow:0 0 12px {glow}"></div>
                        </div>
                    </div>""", unsafe_allow_html=True)

                # Scatter
                if total >= 2:
                    st.markdown('<div class="shdr"><div class="shdr-text">🎯 Confidence Scatter</div><div class="shdr-line"></div></div>', unsafe_allow_html=True)
                    fig_sc = px.scatter(
                        x=list(range(1,total+1)), y=confs,
                        color=[d["severity"] for d in report],
                        size=[max(c*50,8) for c in confs],
                        color_discrete_map={"HIGH":"#ef4444","MEDIUM":"#fb923c","LOW":"#fbbf24"},
                        hover_name=[d["object"] for d in report],
                        labels={"x":"Detection #","y":"Confidence Score","color":"Severity"},
                    )
                    fig_sc.update_traces(marker=dict(line=dict(width=0)))
                    fig_sc.update_layout(**PLOTLY_LAYOUT, height=300,
                        title=dict(text="Confidence by Detection Index",font=dict(size=14,color="#a78bfa")),
                        xaxis=dict(gridcolor="rgba(255,255,255,.05)",zeroline=False,tickmode="linear",dtick=1),
                        yaxis=dict(range=[0,1.08],tickformat=".0%",gridcolor="rgba(255,255,255,.05)"),
                        legend=dict(font=dict(color="#94a3b8"),bgcolor="rgba(0,0,0,0)"),
                    )
                    st.plotly_chart(fig_sc, use_container_width=True)

                # Radar / polar if ≥ 3 detections
                if total >= 3:
                    st.markdown('<div class="shdr"><div class="shdr-text">🕸️ Detection Radar</div><div class="shdr-line"></div></div>', unsafe_allow_html=True)
                    fig_radar = go.Figure(go.Scatterpolar(
                        r=[d["confidence"] for d in report] + [report[0]["confidence"]],
                        theta=[f"#{i+1} {d['object'][:8]}" for i,d in enumerate(report)] + [f"#1 {report[0]['object'][:8]}"],
                        fill="toself", fillcolor="rgba(129,140,248,.12)",
                        line=dict(color="#818cf8",width=2),
                        hovertemplate="<b>%{theta}</b><br>Confidence: %{r:.2%}<extra></extra>",
                    ))
                    fig_radar.update_layout(**PLOTLY_LAYOUT, height=340,
                        title=dict(text="Confidence Radar",font=dict(size=14,color="#a78bfa")),
                        polar=dict(bgcolor="rgba(0,0,0,0)",
                            radialaxis=dict(range=[0,1],tickformat=".0%",gridcolor="rgba(255,255,255,.08)",linecolor="rgba(255,255,255,.08)"),
                            angularaxis=dict(gridcolor="rgba(255,255,255,.08)",linecolor="rgba(255,255,255,.08)"),
                        ),
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)

        # ── TAB 3: DETECTIONS ─────────────────────────
        with tab3:
            st.markdown('<div class="shdr"><div class="shdr-text">📋 All Detections</div><div class="shdr-line"></div></div>', unsafe_allow_html=True)
            if not report:
                st.markdown('<div class="gcard" style="text-align:center;color:#34d399;font-size:1.1rem;font-weight:700;padding:2rem">✅ No defects detected — product appears acceptable.</div>', unsafe_allow_html=True)
            else:
                fc1, _ = st.columns([1.5, 3])
                with fc1:
                    sev_filter = st.multiselect("Filter by Severity", ["HIGH","MEDIUM","LOW"], default=["HIGH","MEDIUM","LOW"])
                for i, d in enumerate([x for x in report if x["severity"] in sev_filter]):
                    sev = d["severity"]
                    ico = {"HIGH":"🔴","MEDIUM":"🟠","LOW":"🟡"}.get(sev,"⚪")
                    pct = int(d["confidence"]*100)
                    st.markdown(f"""
                    <div class="dcard {sev}">
                        <div class="dnum">#{i+1}</div>
                        <div style="flex:1">
                            <div class="dtitle">{ico} {d['object'].title()} <span class="sbadge {sev}">{sev}</span></div>
                            <div class="dmeta">
                                <span class="chip">🎯 {pct}% confidence</span>
                                <span class="chip">📍 {d['location']}</span>
                            </div>
                            <div class="cbar-wrap"><div class="cbar {sev}" style="width:{pct}%"></div></div>
                        </div>
                    </div>""", unsafe_allow_html=True)

                st.markdown('<div class="shdr"><div class="shdr-text">📑 Data Table</div><div class="shdr-line"></div></div>', unsafe_allow_html=True)
                df = pd.DataFrame(report)
                df.index = range(1, len(df)+1)
                st.dataframe(
                    df.style.applymap(
                        lambda v: "color:#f87171;font-weight:700" if v=="HIGH"
                        else "color:#fb923c;font-weight:700" if v=="MEDIUM"
                        else "color:#fbbf24;font-weight:700" if v=="LOW" else "",
                        subset=["severity"]
                    ),
                    use_container_width=True, height=230,
                )

        # ── TAB 4: AI REPORT ──────────────────────────
        with tab4:
            st.markdown('<div class="shdr"><div class="shdr-text">🦙 LLaVA AI Analysis</div><div class="shdr-line"></div></div>', unsafe_allow_html=True)
            llava_text = ""
            if enable_llava:
                with st.spinner("🦙 LLaVA is analysing the image…"):
                    llava_text = analyze_image_with_llava(tmp_path)
                st.markdown(f"""
                <div class="aipanel">
                    <div class="aihead">
                        <span style="font-size:1.5rem">🦙</span>
                        <div class="aihtitle">Expert Quality Control Analysis</div>
                        <span style="font-size:.68rem;color:#334155;margin-left:auto">Powered by LLaVA via Ollama</span>
                    </div>
                    <div class="aibody">{llava_text}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown('<div class="gcard" style="color:#475569;font-style:italic;text-align:center;padding:1.5rem">LLaVA is disabled — toggle it on in the sidebar.</div>', unsafe_allow_html=True)

            st.markdown('<div class="shdr"><div class="shdr-text">📄 Download Report</div><div class="shdr-line"></div></div>', unsafe_allow_html=True)
            with st.spinner("📄 Generating PDF…"):
                pdf_path = os.path.join(tempfile.gettempdir(), "inspection_report.pdf")
                generate_pdf_report(tmp_path, annotated_rgb, report, llava_text or "(LLaVA not enabled)", pdf_path)
            with open(pdf_path,"rb") as f:
                pdf_bytes = f.read()

            dl_col, info_col = st.columns([2,3], gap="large")
            with dl_col:
                st.download_button("⬇️  Download PDF Inspection Report", data=pdf_bytes,
                    file_name="inspection_report.pdf", mime="application/pdf", use_container_width=True)
            with info_col:
                st.markdown(f"""
                <div class="gcard" style="margin:0;padding:.9rem 1.1rem">
                    <div style="font-size:.72rem;font-weight:700;color:#818cf8;letter-spacing:2px;text-transform:uppercase;margin-bottom:6px">Report Includes</div>
                    <div style="font-size:.84rem;color:#94a3b8;line-height:1.8">
                        ✦ Timestamp &amp; verdict badge<br>
                        ✦ Original + annotated images<br>
                        ✦ Detection table ({total} item{'s' if total!=1 else ''})<br>
                        ✦ LLaVA AI analysis
                    </div>
                </div>""", unsafe_allow_html=True)
            cleanup_temp(pdf_path)

    finally:
        cleanup_temp(tmp_path)

# ── LANDING ────────────────────────────────────────────
else:
    st.markdown("""
    <div class="landing">
        <span class="licon">🔬</span>
        <div class="ltitle">Ready for Inspection</div>
        <div class="lsub">Drop any product or scene image into the upload zone above
            to start AI-powered defect detection and analysis.</div>
        <div class="steps">
            <div class="step"><div class="stepicon">📂</div><div class="steplbl">Upload Image</div></div>
            <div class="step"><div class="stepicon">🧠</div><div class="steplbl">YOLOv8 Detect</div></div>
            <div class="step"><div class="stepicon">📊</div><div class="steplbl">View Analytics</div></div>
            <div class="step"><div class="stepicon">🦙</div><div class="steplbl">AI Analysis</div></div>
            <div class="step"><div class="stepicon">📄</div><div class="steplbl">PDF Report</div></div>
        </div>
    </div>""", unsafe_allow_html=True)