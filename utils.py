"""
utils.py
────────
Shared helper functions used across the Visual Inspection Agent.
"""

import os
import tempfile
from typing import Tuple

import numpy as np
from PIL import Image


# ── IMAGE HELPERS ────────────────────────────────────────
def save_upload_to_temp(file_bytes: bytes, suffix: str = ".jpg") -> str:
    """
    Write *file_bytes* to a NamedTemporaryFile and return its path.
    Caller is responsible for calling  os.unlink(path)  when done.
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(file_bytes)
    tmp.close()
    return tmp.name


def cleanup_temp(path: str) -> None:
    """Silently delete a temp file if it still exists."""
    try:
        if path and os.path.exists(path):
            os.unlink(path)
    except OSError:
        pass


def ndarray_to_pil(img_rgb: np.ndarray) -> Image.Image:
    """Convert an HxWx3 uint8 RGB numpy array to a PIL Image."""
    return Image.fromarray(img_rgb.astype(np.uint8))


# ── SEVERITY HELPERS ─────────────────────────────────────
SEVERITY_EMOJI = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟡"}
SEVERITY_COLOR  = {"HIGH": "#e53935", "MEDIUM": "#fb8c00", "LOW": "#fdd835"}

def severity_emoji(s: str) -> str:
    return SEVERITY_EMOJI.get(s, "⚪")

def severity_html_badge(s: str) -> str:
    """Return an HTML <span> badge for use in st.markdown(..., unsafe_allow_html=True)."""
    bg = SEVERITY_COLOR.get(s, "#aaaaaa")
    return (
        f'<span style="background:{bg};color:#fff;border-radius:6px;'
        f'padding:2px 10px;font-weight:700;font-size:0.85rem;">{s}</span>'
    )


# ── VERDICT HELPERS ──────────────────────────────────────
def compute_verdict(report) -> Tuple[str, str, str]:
    """
    Return (verdict_label, badge_color_hex, emoji) based on the report.
    """
    if any(d["severity"] == "HIGH" for d in report):
        return "FAIL", "#e53935", "❌"
    if any(d["severity"] == "MEDIUM" for d in report):
        return "WARNING", "#fb8c00", "⚠️"
    return "PASS", "#2e7d32", "✅"


def verdict_html_banner(verdict: str, color: str, emoji: str) -> str:
    """Return a styled HTML banner for the overall verdict."""
    return (
        f'<div style="text-align:center;padding:14px 0;background:{color};'
        f'border-radius:10px;color:#fff;font-size:1.5rem;font-weight:800;'
        f'letter-spacing:2px;">{emoji}  {verdict}</div>'
    )
