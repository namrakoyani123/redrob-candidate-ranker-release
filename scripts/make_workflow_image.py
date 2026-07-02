"""Generate end_to_end_workflow.png for README."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "end_to_end_workflow.png"

BG = (13, 13, 13)
TITLE = (210, 210, 215)
TEXT = (185, 190, 198)
PURPLE = (168, 85, 247)
TEAL = (45, 212, 191)

STEPS = [
    "JD + 100K JSONL in",
    "Clean → parse JD → prescreen → score + traps",
    "Hybrid semantic (BM25 + BGE + RRF)",
    "Blend, calibrate, export top-100 CSV",
    "Validate format & monotonic scores",
]

W, H = 1280, 360
img = Image.new("RGB", (W, H), BG)
draw = ImageDraw.Draw(img)

try:
    font_title = ImageFont.truetype("segoeui.ttf", 34)
    font_body = ImageFont.truetype("consola.ttf", 28)
except OSError:
    font_title = ImageFont.load_default()
    font_body = ImageFont.load_default()

# Title row
cy = 48
draw.ellipse((48, cy - 6, 64, cy + 10), fill=PURPLE)
draw.text((80, cy - 18), "End-to-end workflow", fill=TITLE, font=font_title)

# Steps
y = 110
for step in STEPS:
    tri = [(56, y + 4), (56, y + 22), (72, y + 13)]
    draw.polygon(tri, fill=TEAL)
    draw.text((88, y - 2), step, fill=TEXT, font=font_body)
    y += 46

OUT.parent.mkdir(parents=True, exist_ok=True)
img.save(OUT, optimize=True)
print(f"Wrote {OUT}")
