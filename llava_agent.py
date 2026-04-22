"""
llava_agent.py
──────────────
Connects to a locally-running LLaVA model via Ollama and returns a
natural-language quality-control analysis for a given image.

Requirements
────────────
• Ollama must be installed and running:  https://ollama.ai
• LLaVA model pulled:  ollama pull llava
"""

import base64
import os
from typing import Optional
import ollama


# ── PROMPT TEMPLATE ──────────────────────────────────────
QC_PROMPT = """You are a quality control expert in a manufacturing plant.
Analyze the provided image and describe:
1. What defects or anomalies you see
2. Location of each defect in the image (top-left, center, etc.)
3. Severity assessment (HIGH / MEDIUM / LOW) for each defect
4. Recommended corrective action for each defect

If no defects are visible, state clearly that the product appears acceptable.
Be concise, structured, and use bullet points."""


# ── MAIN AGENT FUNCTION ──────────────────────────────────
def analyze_image_with_llava(image_path: str,
                              model_name: str = "llava") -> str:
    """
    Send *image_path* to a locally-running LLaVA model via Ollama and
    return the model's natural-language quality-control analysis.

    Parameters
    ----------
    image_path : str
        Absolute path to the image file (jpg / png).
    model_name : str
        Ollama model tag.  Default: "llava".

    Returns
    -------
    str
        LLaVA's analysis, or an error message if Ollama is unavailable.
    """
    if not os.path.exists(image_path):
        return f"❌ Image file not found: {image_path}"

    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": QC_PROMPT,
                    "images": [image_path],
                }
            ],
        )
        return response["message"]["content"]

    except Exception as exc:
        # Ollama is not running or model is not pulled → graceful fallback
        return (
            f"⚠️  LLaVA analysis unavailable.\n\n"
            f"**Reason:** {exc}\n\n"
            "To enable AI analysis:\n"
            "1. Install Ollama → https://ollama.ai\n"
            "2. Run: `ollama pull llava`\n"
            "3. Start Ollama and re-run the inspection."
        )


# ── STANDALONE TEST ──────────────────────────────────────
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "test_image.jpg"
    print(analyze_image_with_llava(path))
