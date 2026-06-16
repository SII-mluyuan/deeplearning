from __future__ import annotations

import os
from pathlib import Path
from textwrap import dedent

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-class6")

import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "images" / "lesson6_multimodal"
NOTEBOOK_PATH = ROOT / "content.ipynb"
PAPER_DIR = ROOT / "papers" / "lesson6_figures"
ORIGINAL_FIGURE_DIR = IMAGE_DIR / "paper_originals"


def md(text: str):
    return new_markdown_cell(dedent(text).strip())


def code(text: str):
    return new_code_cell(dedent(text).strip("\n"))


def box(ax, x, y, w, h, text, fc, ec="#333333", fontsize=12, weight="normal"):
    rect = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=2,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(rect)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#222222",
        fontweight=weight,
    )


def arrow(ax, x0, y0, x1, y1, text: str = "", color: str = "#333333", fontsize: int = 12):
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="->", lw=2, color=color),
    )
    if text:
        ax.text(
            (x0 + x1) / 2,
            (y0 + y1) / 2 + 0.03,
            text,
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=color,
        )


def create_multimodal_map():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, 0.03, 0.55, 0.14, 0.20, "vision", "#E8F1FB", fontsize=15)
    box(ax, 0.03, 0.28, 0.14, 0.20, "text", "#DDF2E6", fontsize=15)
    box(ax, 0.03, 0.01, 0.14, 0.20, "audio", "#FDE7D3", fontsize=15)

    box(ax, 0.32, 0.28, 0.20, 0.28, "shared embedding\nspace", "#F4E0F5", fontsize=16)
    box(ax, 0.67, 0.62, 0.24, 0.16, "zero-shot classification", "#FFF2C7", fontsize=14)
    box(ax, 0.67, 0.39, 0.24, 0.16, "image-text retrieval", "#FFF2C7", fontsize=14)
    box(ax, 0.67, 0.16, 0.24, 0.16, "answer ranking", "#FFF2C7", fontsize=14)

    arrow(ax, 0.17, 0.65, 0.32, 0.46, "align")
    arrow(ax, 0.17, 0.38, 0.32, 0.42, "align")
    arrow(ax, 0.17, 0.11, 0.32, 0.38, "bind")
    arrow(ax, 0.52, 0.46, 0.67, 0.70)
    arrow(ax, 0.52, 0.42, 0.67, 0.47)
    arrow(ax, 0.52, 0.38, 0.67, 0.24)

    ax.text(0.03, 0.90, "Lesson 6 Roadmap", fontsize=18, fontweight="bold")
    ax.text(
        0.03,
        0.84,
        "One space, three visible abilities: classification, retrieval, and answer ranking.",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "multimodal_map.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def create_three_hour_timeline():
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 1)
    ax.axis("off")

    segments = [
        (0, 20, "open question\nhuman CLIP", "#E8F1FB"),
        (20, 45, "paper idea\ncontrastive loss", "#DDF2E6"),
        (45, 85, "toy CLIP\nfrom scratch", "#FDE7D3"),
        (85, 100, "summary\nbreak", "#EEEEEE"),
        (100, 135, "real CLIP\nprompt demo", "#FFF2C7"),
        (135, 155, "failure cases\nlimits", "#FFE5EC"),
        (155, 170, "CLIP/T5\nSD3", "#EADCF8"),
        (170, 180, "Bind\nfuture", "#DCEFE8"),
    ]
    y = 0.35
    for start, end, label, color in segments:
        width = end - start
        rect = patches.Rectangle((start, y), width, 0.25, facecolor=color, edgecolor="#333333", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(start + width / 2, y + 0.125, label, ha="center", va="center", fontsize=10)
        ax.text(start, y - 0.07, f"{start}", ha="center", va="top", fontsize=9)
    ax.text(180, y - 0.07, "180", ha="center", va="top", fontsize=9)
    ax.text(0, 0.84, "3-hour lecture flow", fontsize=18, fontweight="bold")
    ax.text(0, 0.75, "Every 20-25 minutes: ask, predict, run a demo, discuss what changed.", fontsize=12)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "three_hour_timeline.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def create_human_clip_matrix():
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.axis("off")

    for i in range(4):
        ax.text(0.55 + i, 4.55, f"text {i + 1}", ha="center", va="center", fontsize=11)
        ax.text(0.05, 3.55 - i, f"image {i + 1}", ha="left", va="center", fontsize=11)
    for r in range(4):
        for c in range(4):
            color = "#F7E08A" if r == c else "#E8F1FB"
            rect = patches.Rectangle((0.5 + c, 3.1 - r), 0.8, 0.8, facecolor=color, edgecolor="#555555")
            ax.add_patch(rect)
            ax.text(0.9 + c, 3.5 - r, "high" if r == c else "low", ha="center", va="center", fontsize=10)
    ax.text(0.0, 4.9, "Human CLIP warm-up: which pairs should match?", fontsize=16, fontweight="bold")
    ax.text(0.0, 0.25, "Training goal: brighten the diagonal, dim the off-diagonal.", fontsize=12)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "human_clip_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def create_paper_figure_map():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.03, 0.93, "Paper figure guide for Lesson 6", fontsize=18, fontweight="bold")
    ax.text(0.03, 0.86, "Use original figures in class as evidence; use redraws in this notebook for clean explanation.", fontsize=11)

    cards = [
        (0.04, 0.63, "CLIP", "Fig. 1 overview\ntraining -> zero-shot"),
        (0.28, 0.63, "ALIGN", "dual encoder\nnoisy alt-text scale"),
        (0.52, 0.63, "SigLIP", "softmax loss vs\nsigmoid pair loss"),
        (0.76, 0.63, "BLIP-2", "Q-Former bridge\nvision -> LLM"),
        (0.04, 0.35, "Flamingo", "frozen vision + LLM\nfew-shot VLM"),
        (0.28, 0.35, "LLaVA", "CLIP vision encoder\ninstruction tuning"),
        (0.52, 0.35, "T5", "text-to-text\nunified format"),
        (0.76, 0.35, "Imagen / SD3", "large text encoder\ncomplex prompts"),
        (0.28, 0.07, "ImageBind", "one space\nsix modalities"),
        (0.52, 0.07, "Class link", "figure -> question\n-> demo result"),
    ]

    for x, y, title, detail in cards:
        box(ax, x, y, 0.18, 0.18, f"{title}\n{detail}", "#F7F3D7", fontsize=10)

    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "paper_figure_map.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def create_original_paper_figures():
    try:
        import fitz
    except ImportError:
        print("PyMuPDF is not installed; skip cropping original paper figures.")
        return

    ORIGINAL_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    crops = {
        "clip_fig1_method.png": ("clip.pdf", 2, (55, 135, 1045, 555)),
        "clip_fig2_zeroshot_efficiency.png": ("clip.pdf", 3, (65, 120, 545, 450)),
        "clip_fig4_prompt_engineering.png": ("clip.pdf", 7, (560, 105, 1045, 555)),
        "align_fig1_method.png": ("align.pdf", 2, (70, 105, 1035, 505)),
        "align_fig2_noisy_pairs.png": ("align.pdf", 3, (60, 1000, 1040, 1295)),
        "siglip_fig1_loss.png": ("siglip.pdf", 3, (65, 110, 1040, 520)),
        "siglip_fig2_batch_size.png": ("siglip.pdf", 4, (70, 95, 1035, 455)),
        "blip2_fig1_framework.png": ("blip2.pdf", 1, (535, 390, 1015, 735)),
        "blip2_fig2_qformer.png": ("blip2.pdf", 3, (45, 70, 1045, 335)),
        "flamingo_fig1_examples.png": ("flamingo.pdf", 2, (190, 135, 925, 1290)),
        "flamingo_fig3_architecture.png": ("flamingo.pdf", 4, (60, 90, 1045, 430)),
        "llava_fig1_architecture.png": ("llava.pdf", 4, (290, 520, 820, 735)),
        "llava_fig2_demo.png": ("llava.pdf", 16, (125, 210, 1000, 1040)),
        "t5_fig1_text_to_text.png": ("t5.pdf", 3, (145, 165, 960, 615)),
        "imagen_fig1_samples.png": ("imagen.pdf", 2, (65, 100, 1045, 1190)),
        "imagen_fig4_findings.png": ("imagen.pdf", 8, (110, 65, 1030, 540)),
        "sd3_fig1_samples.png": ("sd3.pdf", 1, (95, 335, 970, 900)),
        "sd3_fig2_architecture.png": ("sd3.pdf", 5, (80, 95, 1030, 575)),
        "imagebind_fig1_capabilities.png": ("imagebind.pdf", 1, (85, 390, 1015, 790)),
        "imagebind_fig2_overview.png": ("imagebind.pdf", 3, (70, 75, 1040, 365)),
    }

    rendered_pages = {}
    for output_name, (pdf_name, page_no, crop_box) in crops.items():
        pdf_path = PAPER_DIR / pdf_name
        if not pdf_path.exists():
            print(f"Missing {pdf_path}; skip {output_name}.")
            continue
        key = (pdf_name, page_no)
        if key not in rendered_pages:
            doc = fitz.open(pdf_path)
            page = doc[page_no - 1]
            pix = page.get_pixmap(matrix=fitz.Matrix(1.8, 1.8), alpha=False)
            rendered_pages[key] = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        rendered_pages[key].crop(crop_box).save(ORIGINAL_FIGURE_DIR / output_name, quality=95)

    generated = [ORIGINAL_FIGURE_DIR / name for name in crops if (ORIGINAL_FIGURE_DIR / name).exists()]
    if generated:
        thumb_w, thumb_h = 390, 245
        label_h = 22
        pad = 16
        cols = 4
        rows = (len(generated) + cols - 1) // cols
        sheet = Image.new(
            "RGB",
            (cols * (thumb_w + pad) + pad, rows * (thumb_h + label_h + pad) + pad),
            "white",
        )
        draw = ImageDraw.Draw(sheet)
        for idx, path in enumerate(generated):
            row, col = divmod(idx, cols)
            x = pad + col * (thumb_w + pad)
            y = pad + row * (thumb_h + label_h + pad)
            draw.text((x, y), path.name, fill=(0, 0, 0))
            with Image.open(path) as img:
                img = img.convert("RGB")
                img.thumbnail((thumb_w, thumb_h))
                sheet.paste(img, (x, y + label_h))
        sheet.save(ORIGINAL_FIGURE_DIR / "contact_sheet.png", quality=95)


def create_clip_paper_recipe():
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.03, 0.92, "CLIP paper idea, redrawn for class", fontsize=18, fontweight="bold")
    ax.text(0.03, 0.85, "Source idea: Radford et al., 2021. This is a teaching redraw, not the original figure.", fontsize=11)

    box(ax, 0.04, 0.56, 0.16, 0.16, "image batch\nx1, x2, ...", "#E8F1FB", fontsize=13)
    box(ax, 0.04, 0.24, 0.16, 0.16, "text batch\ny1, y2, ...", "#DDF2E6", fontsize=13)
    box(ax, 0.28, 0.56, 0.16, 0.16, "image\nencoder", "#FDE7D3", fontsize=13)
    box(ax, 0.28, 0.24, 0.16, 0.16, "text\nencoder", "#FDE7D3", fontsize=13)
    box(ax, 0.53, 0.39, 0.18, 0.20, "N x N\nsimilarity\nmatrix", "#F4E0F5", fontsize=14)
    box(ax, 0.80, 0.55, 0.15, 0.14, "train:\nmatch pairs", "#FFF2C7", fontsize=13)
    box(ax, 0.80, 0.25, 0.15, 0.14, "test:\ncompare prompts", "#FFF2C7", fontsize=13)

    arrow(ax, 0.20, 0.64, 0.28, 0.64)
    arrow(ax, 0.20, 0.32, 0.28, 0.32)
    arrow(ax, 0.44, 0.64, 0.53, 0.51)
    arrow(ax, 0.44, 0.32, 0.53, 0.46)
    arrow(ax, 0.71, 0.49, 0.80, 0.62)
    arrow(ax, 0.71, 0.49, 0.80, 0.32)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "clip_paper_recipe.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def create_clip_pipeline():
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, 0.04, 0.55, 0.16, 0.22, "image\nencoder", "#E8F1FB", fontsize=15)
    box(ax, 0.04, 0.18, 0.16, 0.22, "text\nencoder", "#DDF2E6", fontsize=15)
    box(ax, 0.33, 0.55, 0.16, 0.22, "image\nembedding", "#FDE7D3", fontsize=15)
    box(ax, 0.33, 0.18, 0.16, 0.22, "text\nembedding", "#FDE7D3", fontsize=15)
    box(ax, 0.60, 0.32, 0.16, 0.28, "similarity\nmatrix", "#F4E0F5", fontsize=15)
    box(ax, 0.84, 0.32, 0.12, 0.28, "contrastive\nloss", "#FFF2C7", fontsize=15)

    arrow(ax, 0.20, 0.66, 0.33, 0.66, "encode")
    arrow(ax, 0.20, 0.29, 0.33, 0.29, "encode")
    arrow(ax, 0.49, 0.66, 0.60, 0.49, "normalize")
    arrow(ax, 0.49, 0.29, 0.60, 0.43, "normalize")
    arrow(ax, 0.76, 0.46, 0.84, 0.46, "diagonal up,\noff-diagonal down", fontsize=11)

    ax.text(0.03, 0.90, "CLIP training objective", fontsize=18, fontweight="bold")
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "clip_pipeline.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def create_zero_shot_pipeline():
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, 0.05, 0.32, 0.14, 0.28, "query image", "#E8F1FB", fontsize=15)
    box(ax, 0.28, 0.32, 0.16, 0.28, "image embedding", "#FDE7D3", fontsize=15)
    box(ax, 0.57, 0.55, 0.18, 0.14, "a photo of a cat", "#DDF2E6", fontsize=13)
    box(ax, 0.57, 0.35, 0.18, 0.14, "a photo of a dog", "#DDF2E6", fontsize=13)
    box(ax, 0.57, 0.15, 0.18, 0.14, "a photo of a truck", "#DDF2E6", fontsize=13)
    box(ax, 0.84, 0.32, 0.11, 0.28, "best\nprompt", "#FFF2C7", fontsize=15)

    arrow(ax, 0.19, 0.46, 0.28, 0.46)
    arrow(ax, 0.44, 0.46, 0.57, 0.62, "compare")
    arrow(ax, 0.44, 0.46, 0.57, 0.42, "compare")
    arrow(ax, 0.44, 0.46, 0.57, 0.22, "compare")
    arrow(ax, 0.75, 0.42, 0.84, 0.46, "argmax")

    ax.text(0.03, 0.88, "Zero-shot classification = prompt bank + similarity ranking", fontsize=18, fontweight="bold")
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "zero_shot_pipeline.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def create_failure_cases():
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.03, 0.90, "Where CLIP-style matching is easy to over-trust", fontsize=18, fontweight="bold")
    cards = [
        (0.04, 0.55, "counting", "two dogs vs three dogs"),
        (0.29, 0.55, "spatial relation", "left of / right of"),
        (0.54, 0.55, "fine detail", "similar species or models"),
        (0.79, 0.55, "forced choice", "all answers can be wrong"),
        (0.17, 0.18, "text in image", "small letters and typography"),
        (0.42, 0.18, "reasoning", "needs multiple steps"),
        (0.67, 0.18, "dataset bias", "looks familiar, not certain"),
    ]
    for x, y, title, detail in cards:
        box(ax, x, y, 0.18, 0.17, f"{title}\n{detail}", "#FFE5EC", fontsize=11)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "clip_failure_cases.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def create_encoder_roles():
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.03, 0.92, "CLIP and T5 encoders play different jobs", fontsize=18, fontweight="bold")
    ax.text(0.03, 0.85, "CLIP is good at matching. T5 is good at reading detailed text conditions.", fontsize=12)

    box(ax, 0.04, 0.58, 0.15, 0.15, "image", "#E8F1FB", fontsize=14)
    box(ax, 0.04, 0.25, 0.15, 0.15, "short prompt\nor class name", "#DDF2E6", fontsize=13)
    box(ax, 0.28, 0.58, 0.16, 0.15, "CLIP image\nencoder", "#FDE7D3", fontsize=13)
    box(ax, 0.28, 0.25, 0.16, 0.15, "CLIP text\nencoder", "#FDE7D3", fontsize=13)
    box(ax, 0.52, 0.42, 0.16, 0.18, "shared\nembedding\nspace", "#F4E0F5", fontsize=14)
    box(ax, 0.75, 0.42, 0.18, 0.18, "matching,\nretrieval,\nzero-shot", "#FFF2C7", fontsize=13)

    arrow(ax, 0.19, 0.655, 0.28, 0.655)
    arrow(ax, 0.19, 0.325, 0.28, 0.325)
    arrow(ax, 0.44, 0.655, 0.52, 0.53)
    arrow(ax, 0.44, 0.325, 0.52, 0.48)
    arrow(ax, 0.68, 0.51, 0.75, 0.51)

    box(ax, 0.04, 0.02, 0.20, 0.13, "long prompt:\nobjects, relations,\nwritten text", "#DDF2E6", fontsize=12)
    box(ax, 0.33, 0.02, 0.16, 0.13, "T5 encoder", "#FDE7D3", fontsize=14)
    box(ax, 0.58, 0.02, 0.17, 0.13, "token-level\ntext context", "#F4E0F5", fontsize=13)
    box(ax, 0.82, 0.02, 0.14, 0.13, "generator /\nVLM backbone", "#FFF2C7", fontsize=12)

    arrow(ax, 0.24, 0.085, 0.33, 0.085)
    arrow(ax, 0.49, 0.085, 0.58, 0.085)
    arrow(ax, 0.75, 0.085, 0.82, 0.085)

    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "encoder_roles.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def create_imagen_text_encoder_card():
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.03, 0.90, "Imagen / SD3 lesson: text understanding matters", fontsize=18, fontweight="bold")
    ax.text(0.03, 0.82, "Teaching redraw from paper conclusions: large text encoders help complex prompts.", fontsize=11)

    box(ax, 0.05, 0.42, 0.22, 0.20, "long prompt\nobjects + relations + text", "#DDF2E6", fontsize=13)
    box(ax, 0.37, 0.42, 0.18, 0.20, "T5 encoder\nreads details", "#FDE7D3", fontsize=14)
    box(ax, 0.66, 0.42, 0.22, 0.20, "image generator\nuses text context", "#FFF2C7", fontsize=13)
    arrow(ax, 0.27, 0.52, 0.37, 0.52)
    arrow(ax, 0.55, 0.52, 0.66, 0.52)
    ax.text(
        0.05,
        0.19,
        "Classroom takeaway: CLIP answers 'does this image match this phrase?'; T5 helps preserve what the full phrase asks for.",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "imagen_text_encoder_card.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def create_imagebind_map():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (0.04, 0.70, "image", "#E8F1FB"),
        (0.04, 0.46, "text", "#DDF2E6"),
        (0.04, 0.22, "audio", "#FDE7D3"),
        (0.04, 0.00, "depth / thermal /\nIMU", "#FFE5EC"),
    ]
    for x, y, label, color in boxes:
        box(ax, x, y, 0.18, 0.18, label, color, fontsize=14)

    box(ax, 0.40, 0.30, 0.22, 0.26, "shared\nmultimodal\nspace", "#F4E0F5", fontsize=16)
    box(ax, 0.76, 0.58, 0.17, 0.14, "cross-modal\nretrieval", "#FFF2C7", fontsize=13)
    box(ax, 0.76, 0.36, 0.17, 0.14, "open-world\nbinding", "#FFF2C7", fontsize=13)
    box(ax, 0.76, 0.14, 0.17, 0.14, "beyond\nimage-text", "#FFF2C7", fontsize=13)

    for _, y, _, _ in boxes:
        arrow(ax, 0.22, y + 0.09, 0.40, 0.43, "align" if y >= 0.46 else "")
    arrow(ax, 0.62, 0.43, 0.76, 0.65)
    arrow(ax, 0.62, 0.43, 0.76, 0.43)
    arrow(ax, 0.62, 0.43, 0.76, 0.21)

    ax.text(0.03, 0.93, "ImageBind extends CLIP-style alignment to more modalities", fontsize=17, fontweight="bold")
    ax.text(0.03, 0.86, "Teaching redraw from Girdhar et al., 2023.", fontsize=11)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "imagebind_map.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_images():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    create_multimodal_map()
    create_three_hour_timeline()
    create_human_clip_matrix()
    create_paper_figure_map()
    create_original_paper_figures()
    create_clip_paper_recipe()
    create_clip_pipeline()
    create_zero_shot_pipeline()
    create_failure_cases()
    create_encoder_roles()
    create_imagen_text_encoder_card()
    create_imagebind_map()


def build_notebook():
    cells = [
        md(
            """
            # 第六讲：多模态、CLIP 与文本编码器

            这一讲重新设计成 3 小时课堂，不讲成“又出现几个大模型”。主线只有一句话：

            > **一张图片为什么可以用一句话来分类？CLIP 的答案是：把图片和文字放到同一个空间里，让它们互相找朋友。**

            课堂采用四步循环：

            1. 先抛开放问题，让学生猜。
            2. 再看论文图示、公式或结论。
            3. 接着跑一个 demo。
            4. 最后讨论 demo 是否支持刚才的判断。

            这样 3 小时不是靠堆内容，而是让学生不断经历“预测、验证、修正”的过程。
            """
        ),
        md(
            """
            ## 本节课学习目标

            1. 能用通俗语言解释“共享表示空间”是什么。
            2. 能看懂 CLIP 的图文对比学习：为什么对角线要亮，非对角线要暗。
            3. 能从零实现一个最小 toy CLIP，并解释 loss、相似度矩阵、zero-shot、检索结果。
            4. 能使用预训练 OpenAI CLIP 做 zero-shot 分类、图文检索、图片到文本匹配、多选式 VQA。
            5. 能说出 CLIP 的边界：它更像“匹配器”，不是万能视觉推理器。
            6. 能区分 CLIP 编码器和 T5 编码器：一个擅长“配不配”，一个擅长“读得细”。
            7. 能理解 ImageBind 为什么是 CLIP 思路的自然扩展。
            """
        ),
        md(
            """
            ## 3 小时时间安排

            <img src="images/lesson6_multimodal/three_hour_timeline.png" width="980">

            这节课每 20 到 25 分钟安排一次停顿：

            - **猜想型问题**：先让学生判断结果会怎样。
            - **诊断型问题**：跑完 demo 后解释现象。
            - **辩论型问题**：把 CLIP 的能力和边界讲清楚。

            教师提醒：不要急着给答案。CLIP 这节课最有价值的地方，是让学生亲眼看到“分类、检索、选答案”其实可以用同一套相似度比较来做。
            """
        ),
        md(
            """
            ## 更细的 3 小时讲稿节奏

            下面这张表可以直接当教师讲稿使用：

            | 时间 | 内容 | 学生活动 | 产出 |
            |---|---|---|---|
            | 0-10 分钟 | 分类、检索、问答三个任务开场 | 判断它们是不是同一件事 | 得到“相似度比较”的直觉 |
            | 10-25 分钟 | 人肉 CLIP + 4x4 矩阵 | 手动画正例/负例 | 明白对角线 |
            | 25-42 分钟 | CLIP / ALIGN / SigLIP 原论文图 | 每张图只回答一个问题 | 建立论文证据链 |
            | 42-58 分钟 | 对比学习公式 + temperature demo | 改 temperature 看概率变化 | 理解 logit_scale |
            | 58-85 分钟 | toy CLIP 训练 | 观察 loss、相似度矩阵、混淆矩阵 | 看见图文对齐发生 |
            | 85-95 分钟 | 小结 + 休息 | 说出 zero-shot 不神奇在哪里 | 前半段收束 |
            | 95-120 分钟 | 预训练 CLIP + prompt challenge | 学生自己写 prompt | 理解 prompt 影响文本向量 |
            | 120-135 分钟 | 检索、图片选文本、多选 VQA | 解释输出排序 | 统一三个任务 |
            | 135-150 分钟 | 失败案例：空间关系、强制选择 | 讨论“不知道”机制 | 理解边界 |
            | 150-162 分钟 | BLIP-2 / Flamingo / LLaVA 原图 | 读图 worksheet | 从 CLIP 走到 VLM |
            | 162-172 分钟 | T5 / Imagen / SD3 原图 | 判断 CLIP/T5 分工 | 理解文本编码器 |
            | 172-178 分钟 | ImageBind 原图 | 设计一个跨模态应用 | 扩展到更多模态 |
            | 178-180 分钟 | 小测 + 作业说明 | 记录课后任务 | 检查理解 |

            如果学生基础较弱，就把 ImageBind 压缩成 3 分钟，把节省出来的时间给 toy CLIP 和 prompt demo。
            """
        ),
        md(
            """
            ## 全课路线图

            <img src="images/lesson6_multimodal/multimodal_map.png" width="980">

            这节课两条线并行：

            - **线 1：从零理解 CLIP。** 用彩色几何图形把图文对齐讲透。
            - **线 2：理解真实 CLIP 能做什么、不能做什么。** 用 CIFAR10 做 zero-shot、检索、prompt 对照、失败案例。

            中段补一个今天很实用的问题：为什么很多多模态生成模型同时用 CLIP 和 T5？

            最后用 ImageBind 收束：CLIP 不是终点，它是“共享空间”路线里最经典的一步。
            """
        ),
        md(
            """
            ## 原论文图引用导航

            <img src="images/lesson6_multimodal/paper_figure_map.png" width="980">

            建议讲课时多打开原论文 PDF 里的图，但 notebook 里尽量用课堂重绘图承接。这样有两个好处：

            - 学生知道观点来自哪里，不是老师凭空总结。
            - 重绘图更干净，适合课堂解释；原图更适合做“证据”和“论文阅读训练”。

            可以重点引用这些原论文图：

            1. **CLIP**：[Radford et al., 2021](https://arxiv.org/abs/2103.00020) 的总览图。看“图文对比训练”和“zero-shot prompt 分类”如何接起来。
            2. **ALIGN**：[Jia et al., 2021](https://arxiv.org/abs/2102.05918) 的双编码器路线图。看“大规模噪声图文对”为什么也能学出可迁移表示。
            3. **SigLIP**：[Zhai et al., 2023](https://arxiv.org/abs/2303.15343) 的损失函数对比图。看 softmax 对比学习之外还有什么做法。
            4. **BLIP-2**：[Li et al., 2023](https://arxiv.org/abs/2301.12597) 的 Q-Former 架构图。看“视觉编码器”和“大语言模型”之间怎么接桥。
            5. **Flamingo**：[Alayrac et al., 2022](https://arxiv.org/abs/2204.14198) 的 VLM 架构图。看冻结视觉模型和冻结语言模型如何被连接。
            6. **LLaVA**：[Liu et al., 2023](https://arxiv.org/abs/2304.08485) 的视觉指令微调流程图。看 CLIP 视觉编码器如何进入聊天式 VLM。
            7. **T5**：[Raffel et al., 2019](https://arxiv.org/abs/1910.10683) 的 text-to-text 框架图。看为什么 T5 适合把任务都变成“读文本”。
            8. **Imagen**：[Saharia et al., 2022](https://arxiv.org/abs/2205.11487) 的文生图结构图或文本编码器对比图。看强文本编码器为什么重要。
            9. **Stable Diffusion 3**：[Esser et al., 2024](https://arxiv.org/abs/2403.03206) 的架构图。看 CLIP 和 T5 为什么会同时出现。
            10. **ImageBind**：[Girdhar et al., 2023](https://arxiv.org/abs/2305.05665) 的多模态共享空间图。看 CLIP 思路如何扩展到更多模态。

            课堂用法：每张图只问一个问题，不要把论文图当成知识点清单逐项念。
            """
        ),
        md(
            """
            ## 本 notebook 使用的论文原图说明

            下面各章节中插入的论文图，来自相应论文 PDF 的原图裁剪，保留图号或图注，方便课堂投屏讲解。  
            建议讲课时这样处理：

            - 原图用于“证据”：告诉学生这个结论来自哪篇论文。
            - 代码 demo 用于“验证”：让学生看到这个思想在小例子里真的能跑通。
            - 讨论问题用于“转化”：把论文图变成学生自己的理解。
            """
        ),
        code(
            """
            # ------------------------------
            # 这段代码做什么：
            # 1. 导入本节课需要的依赖
            # 2. 固定随机种子，保证结果更稳定
            # 3. 配置是否运行 ImageBind 扩展
            # ------------------------------
            from pathlib import Path
            import math
            import random
            import warnings
            import logging
            import numpy as np
            import matplotlib.pyplot as plt
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            from torch.utils.data import Dataset, DataLoader, random_split
            from torchvision import datasets
            from PIL import Image, ImageDraw
            from transformers import CLIPModel, CLIPProcessor

            plt.style.use("seaborn-v0_8-whitegrid")

            SEED = 42
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)

            warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
            logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            RUN_IMAGEBIND = False

            print("device =", device)
            print("RUN_IMAGEBIND =", RUN_IMAGEBIND)
            """
        ),
        md(
            """
            # Part 1. 开场：这些任务是不是同一件事？

            先不要讲定义。先给学生三个任务：

            1. 给一张图分类：cat / dog / truck。
            2. 用一句话找图片：`a red car on the street`。
            3. 给图片做多选题：`the answer is cat / dog / bird`。

            **开放问题 1：**

            > 这三个任务看起来不一样，它们有没有可能本质上是同一件事？

            两人一组讨论 2 分钟。收几个答案后再给出本节课的核心说法：

            > 它们都可以看成“图片向量”和“文字向量”的相似度比较。
            """
        ),
        md(
            """
            ## 课堂任务单：先把任务翻译成“相似度比较”

            给学生 6 分钟完成下面这张表。可以两人一组，不要求马上答对，重点是训练“把任务改写成比较问题”的能力。

            | 原任务 | 可以怎么改写成相似度比较 | 候选文本从哪里来 |
            |---|---|---|
            | 图片分类 | 图片 vs 类别 prompt | 类别名 |
            | 文本搜图 | 查询文本 vs 图片库 | 用户输入 |
            | 图片搜描述 | 图片 vs 描述库 | 标题/标签/候选说明 |
            | 多选式 VQA | 图片 vs 候选答案 | 选项 |
            | 内容审核 | 图片 vs 风险描述 | 审核规则 |
            | 商品检索 | 商品图 vs 用户搜索词 | 商品标题/类目 |

            讨论问题：

            > 如果一个任务可以被改写成“图片和文本谁更像”，它就一定适合用 CLIP 吗？

            这会自然引出后面的边界：能排序，不代表能可靠推理。
            """
        ),
        md(
            """
            ## 人肉 CLIP demo

            <img src="images/lesson6_multimodal/human_clip_matrix.png" width="760">

            老师可以在黑板上画 4 张图片和 4 句话，让学生手动连线。

            课堂问题：

            > 哪些格子应该分数最高？为什么正确答案刚好在对角线上？

            这一步的目的不是讲数学，而是让学生先看到 CLIP 训练目标的形状：**正确配对要亮，错误配对要暗。**
            """
        ),
        md(
            """
            ## 板书活动：4x4 相似度矩阵怎么变成 loss

            老师可以在黑板上画一个 4x4 表格：

            | | text 1 | text 2 | text 3 | text 4 |
            |---|---|---|---|---|
            | image 1 | 高 | 低 | 低 | 低 |
            | image 2 | 低 | 高 | 低 | 低 |
            | image 3 | 低 | 低 | 高 | 低 |
            | image 4 | 低 | 低 | 低 | 高 |

            然后问三步：

            1. 如果只从 image 1 看，它要在 4 个 text 里选谁？
            2. 如果只从 text 1 看，它要在 4 张 image 里选谁？
            3. 为什么 CLIP 的 loss 要算两个方向？

            这一段建议讲 8 到 10 分钟。不要急着写公式，先让学生把“图找文、文找图”说出来。
            """
        ),
        md(
            """
            # Part 2. CLIP 论文思想：图文对比学习

            <img src="images/lesson6_multimodal/clip_paper_recipe.png" width="980">

            论文证据卡：

            - 论文：Radford et al., 2021, [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
            - 关键结论：用大量图文对训练后，模型可以用自然语言描述视觉概念，并迁移到很多分类任务。
            - 课堂翻译：图片和文字被放进同一个空间后，分类可以变成“拿图片去找最像的一句话”。

            这张图是课堂重绘图，不是论文原图。它保留的重点是：图像编码器、文本编码器、相似度矩阵、zero-shot prompt bank。
            """
        ),
        md(
            """
            ## 论文原图：CLIP 方法图和 zero-shot 效果

            **CLIP Figure 1：方法总览**

            <img src="images/lesson6_multimodal/paper_originals/clip_fig1_method.png" width="980">

            这张原图课堂上重点看两件事：

            - 训练阶段：图片和文本配对，形成相似度矩阵。
            - 测试阶段：类别名被写成 prompt，再和图片比较相似度。

            **CLIP Figure 2：zero-shot 迁移效率**

            <img src="images/lesson6_multimodal/paper_originals/clip_fig2_zeroshot_efficiency.png" width="620">

            课堂问题：

            > 这张图为什么能说明“自然语言监督”比固定分类标签更灵活？
            """
        ),
        md(
            """
            ## 同一条路线上的论文图

            这里可以连续展示三张原论文图，每张只讲一个点：

            1. **CLIP 总览图**：看“训练时配图文，测试时拿类别名写 prompt”。  
               课堂问题：为什么 zero-shot 分类不需要重新训练分类头？

            2. **ALIGN 双编码器图**：看 CLIP 这类路线可以靠更大规模、更嘈杂的图文对继续放大。  
               课堂问题：数据有噪声时，为什么规模可能部分弥补噪声？

            3. **SigLIP 损失对比图**：看对比学习并不只有 softmax 一种写法，也可以把每个图文对看成独立二分类。  
               课堂问题：如果 batch 很小，softmax 对比学习会遇到什么问题？

            这三张图对应三句话：

            - CLIP 讲清楚“图文对齐能做 zero-shot”。
            - ALIGN 讲清楚“规模和噪声之间可以做权衡”。
            - SigLIP 讲清楚“训练目标本身还能继续改进”。
            """
        ),
        md(
            """
            ## 论文原图：ALIGN 和 SigLIP

            **ALIGN Figure 1：大规模噪声图文对训练**

            <img src="images/lesson6_multimodal/paper_originals/align_fig1_method.png" width="980">

            这张图重点看：不是每条文本都干净、标准，但规模很大时，模型仍然能学到有用的图文对齐。

            **ALIGN Figure 2：真实网络图文对的噪声**

            <img src="images/lesson6_multimodal/paper_originals/align_fig2_noisy_pairs.png" width="920">

            课堂问题：

            > 如果一部分图文对是错的，为什么模型没有完全学坏？

            **SigLIP Figure 1：sigmoid loss 的实现思路**

            <img src="images/lesson6_multimodal/paper_originals/siglip_fig1_loss.png" width="980">

            **SigLIP Figure 2：batch size 对结果的影响**

            <img src="images/lesson6_multimodal/paper_originals/siglip_fig2_batch_size.png" width="980">

            课堂问题：

            > CLIP 的 softmax loss 很依赖 batch 里的负例。那如果 batch 不够大，有没有别的训练目标可以缓解？
            """
        ),
        md(
            """
            ## 对比学习公式：只看懂这一版就够了

            图像编码器得到图像向量：

            ```latex
            v_i = \\frac{f_I(x_i)}{\\|f_I(x_i)\\|}
            ```

            文本编码器得到文本向量：

            ```latex
            t_j = \\frac{f_T(y_j)}{\\|f_T(y_j)\\|}
            ```

            图文相似度：

            ```latex
            S_{ij} = \\exp(\\tau) \\cdot v_i^\\top t_j
            ```

            损失函数可以理解成：

            ```latex
            L = \\frac{1}{2}\\left[CE(S, labels) + CE(S^T, labels)\\right]
            ```

            通俗解释：

            - `S`：让每张图找对自己的文字。
            - `S.T`：让每句话也找对自己的图片。
            - `labels=[0,1,2,...]`：正确答案在对角线上。

            课堂问题：

            > 如果一个 batch 里面有两张很像的图，非对角线也变亮，会发生什么？
            """
        ),
        code(
            """
            # ------------------------------
            # 这段代码做什么：
            # 1. 用一个小分数向量演示 temperature 的作用
            # 2. 说明 logit_scale 为什么会影响模型“有多自信”
            # ------------------------------
            raw_scores = torch.tensor([2.0, 1.6, 0.8, -0.2])
            temperatures = [2.0, 1.0, 0.5, 0.1]

            plt.figure(figsize=(7, 3))
            for tau in temperatures:
                probs = torch.softmax(raw_scores / tau, dim=0).numpy()
                plt.plot(range(len(raw_scores)), probs, marker="o", label=f"temperature={tau}")

            plt.xticks(range(len(raw_scores)), ["candidate A", "B", "C", "D"])
            plt.ylabel("softmax probability")
            plt.title("Lower temperature makes similarity ranking sharper")
            plt.legend()
            plt.tight_layout()
            plt.show()

            for tau in temperatures:
                probs = torch.softmax(raw_scores / tau, dim=0)
                print(f"temperature={tau}: {probs.round(decimals=3).tolist()}")
            """
        ),
        md(
            """
            课堂追问：

            > 如果 temperature 很低，模型看起来会更“自信”。但这种自信一定可靠吗？

            这里可以提醒学生：softmax 的最高概率只是相对候选项最高，不等于答案绝对正确。后面“候选答案都不对”的 demo 会再次看到这一点。
            """
        ),
        md(
            """
            <img src="images/lesson6_multimodal/clip_pipeline.png" width="980">

            ### CLIP 的一句话直觉

            - 图像编码器把图片编码成向量。
            - 文本编码器把句子编码成向量。
            - 同一对图文的相似度要高，错误配对的相似度要低。

            这就是为什么 CLIP 训练完以后，不用再额外训练分类头，也能拿文本提示词直接做分类。
            """
        ),
        md(
            """
            # Part 3. 从零实现一个 toy CLIP

            真实图文数据很复杂，课堂上容易被噪声盖住重点。所以我们先造一个小世界：

            - 图像：彩色几何图形。
            - 文本：`a red circle`、`a blue square`。

            **猜想型问题：**

            > 这个小模型最后是在学“红色”和“圆形”，还是只是在背训练集？

            先让学生投票，后面用 zero-shot 和检索结果验证。
            """
        ),
        code(
            """
            # ------------------------------
            # 这段代码做什么：
            # 1. 生成一个彩色几何图形数据集
            # 2. 每张图都配一个简短文本描述，形成最小图文对
            # ------------------------------
            colors = ["red", "blue", "green", "yellow"]
            shapes = ["circle", "square", "triangle", "diamond"]
            templates = [
                "a {color} {shape}",
                "one {color} {shape}",
                "a small {color} {shape}",
                "a centered {color} {shape}",
            ]


            def render_shape(shape, color, size=64):
                img = Image.new("RGB", (size, size), "white")
                draw = ImageDraw.Draw(img)
                cx = random.randint(18, 46)
                cy = random.randint(18, 46)
                radius = random.randint(12, 18)

                if shape == "circle":
                    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color)
                elif shape == "square":
                    draw.rectangle((cx - radius, cy - radius, cx + radius, cy + radius), fill=color)
                elif shape == "triangle":
                    draw.polygon([(cx, cy - radius), (cx - radius, cy + radius), (cx + radius, cy + radius)], fill=color)
                elif shape == "diamond":
                    draw.polygon([(cx, cy - radius), (cx - radius, cy), (cx, cy + radius), (cx + radius, cy)], fill=color)

                arr = np.asarray(img).astype("float32") / 255.0
                return torch.tensor(arr).permute(2, 0, 1)


            toy_samples = []
            for color in colors:
                for shape in shapes:
                    for _ in range(120):
                        caption = random.choice(templates).format(color=color, shape=shape)
                        label = f"{color} {shape}"
                        toy_samples.append((render_shape(shape, color), caption, label))

            random.shuffle(toy_samples)

            fig, axes = plt.subplots(2, 4, figsize=(10, 5))
            for ax, (image, caption, label) in zip(axes.flat, toy_samples[:8]):
                ax.imshow(image.permute(1, 2, 0))
                ax.set_title(caption, fontsize=9)
                ax.axis("off")
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            课堂讨论：

            > 如果训练数据里从来没有出现过 `a tiny green diamond`，模型有没有可能仍然选到绿色菱形？

            引导学生区分两件事：

            - 背模板：只记住见过的整句话。
            - 学组合：知道 green 和 diamond 可以组合。
            """
        ),
        code(
            """
            # ------------------------------
            # 这段代码做什么：
            # 1. 构造最小 tokenizer 和 Dataset
            # 2. 把图像、token ids 和语义标签组织到一起
            # ------------------------------
            vocab = sorted(set(" ".join(caption for _, caption, _ in toy_samples).split()))
            word2idx = {word: i + 1 for i, word in enumerate(vocab)}  # 0 留给 padding


            def tokenize(text, max_len=6):
                ids = [word2idx[word] for word in text.split() if word in word2idx]
                ids = ids[:max_len] + [0] * max(0, max_len - len(ids))
                return torch.tensor(ids, dtype=torch.long)


            class ShapeCaptionDataset(Dataset):
                def __init__(self, samples):
                    self.samples = samples

                def __len__(self):
                    return len(self.samples)

                def __getitem__(self, idx):
                    image, caption, label = self.samples[idx]
                    return image, tokenize(caption), caption, label


            full_ds = ShapeCaptionDataset(toy_samples)
            train_size = int(0.8 * len(full_ds))
            val_size = len(full_ds) - train_size
            train_ds, val_ds = random_split(
                full_ds,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(SEED),
            )

            train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
            val_dl = DataLoader(val_ds, batch_size=128, shuffle=False)

            print("vocab size =", len(vocab))
            print("vocab =", vocab)
            print("train size =", len(train_ds), "val size =", len(val_ds))
            """
        ),
        md(
            """
            ## 代码前先看结构

            这个 toy CLIP 只有三个核心部件：

            1. `TinyImageEncoder`：小 CNN，把图像变成一个向量。
            2. `TinyTextEncoder`：词向量求平均，把短句变成一个向量。
            3. `clip_loss`：让正确图文对相似度变高，错误配对变低。

            注意：这里故意不用复杂 Transformer。课堂目标是把 CLIP 的训练信号讲清楚。
            """
        ),
        code(
            """
            # ------------------------------
            # 这段代码做什么：
            # 1. 定义一个最小 image encoder 和 text encoder
            # 2. 用对比学习目标训练它们进入同一个语义空间
            # ------------------------------
            class TinyImageEncoder(nn.Module):
                def __init__(self, embed_dim=64):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Conv2d(3, 24, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(24, 48, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(48, 96, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.AdaptiveAvgPool2d(1),
                    )
                    self.proj = nn.Linear(96, embed_dim)

                def forward(self, images):
                    feats = self.net(images).flatten(1)
                    return self.proj(feats)


            class TinyTextEncoder(nn.Module):
                def __init__(self, vocab_size, embed_dim=64):
                    super().__init__()
                    self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
                    self.proj = nn.Sequential(
                        nn.Linear(embed_dim, embed_dim),
                        nn.ReLU(),
                        nn.Linear(embed_dim, embed_dim),
                    )

                def forward(self, token_ids):
                    token_embed = self.embedding(token_ids)
                    mask = (token_ids != 0).unsqueeze(-1)
                    pooled = (token_embed * mask).sum(1) / mask.sum(1).clamp(min=1)
                    return self.proj(pooled)


            class TinyCLIP(nn.Module):
                def __init__(self, vocab_size, embed_dim=64):
                    super().__init__()
                    self.image_encoder = TinyImageEncoder(embed_dim=embed_dim)
                    self.text_encoder = TinyTextEncoder(vocab_size=vocab_size, embed_dim=embed_dim)
                    self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

                def forward(self, images, token_ids):
                    image_embeds = F.normalize(self.image_encoder(images), dim=1)
                    text_embeds = F.normalize(self.text_encoder(token_ids), dim=1)
                    scale = self.logit_scale.exp().clamp(max=100)
                    logits = image_embeds @ text_embeds.T * scale
                    return logits, image_embeds, text_embeds


            toy_clip = TinyCLIP(vocab_size=len(vocab), embed_dim=64).to(device)
            toy_opt = torch.optim.Adam(toy_clip.parameters(), lr=1e-3)


            def clip_loss(logits):
                labels = torch.arange(len(logits), device=logits.device)
                loss_i = F.cross_entropy(logits, labels)
                loss_t = F.cross_entropy(logits.T, labels)
                return (loss_i + loss_t) / 2


            print(toy_clip)
            """
        ),
        code(
            """
            # ------------------------------
            # 这段代码做什么：
            # 1. 训练 toy CLIP
            # 2. 观察对比学习损失是否下降
            # ------------------------------
            toy_losses = []

            for epoch in range(12):
                batch_losses = []
                toy_clip.train()
                for images, token_ids, captions, labels in train_dl:
                    images = images.to(device)
                    token_ids = token_ids.to(device)

                    logits, _, _ = toy_clip(images, token_ids)
                    loss = clip_loss(logits)

                    toy_opt.zero_grad()
                    loss.backward()
                    toy_opt.step()
                    batch_losses.append(float(loss.item()))

                epoch_loss = float(np.mean(batch_losses))
                toy_losses.append(epoch_loss)

                if epoch in [0, 3, 7, 11]:
                    print(f"Epoch {epoch + 1}: loss={epoch_loss:.4f}")
            """
        ),
        code(
            """
            # ------------------------------
            # 这段代码做什么：
            # 1. 画出 toy CLIP 的训练 loss
            # 2. 看看图文对齐是不是在逐步变好
            # ------------------------------
            plt.figure(figsize=(6, 3))
            plt.plot(toy_losses, marker="o")
            plt.title("Toy CLIP training loss")
            plt.xlabel("epoch")
            plt.ylabel("contrastive loss")
            plt.show()
            """
        ),
        code(
            """
            # ------------------------------
            # 这段代码做什么：
            # 1. 可视化一个 mini-batch 的相似度矩阵
            # 2. 看看对角线是否逐渐变亮
            # ------------------------------
            toy_clip.eval()
            images, token_ids, captions, labels = next(iter(val_dl))
            images = images[:8].to(device)
            token_ids = token_ids[:8].to(device)

            with torch.no_grad():
                logits, image_embeds, text_embeds = toy_clip(images, token_ids)
                sim = logits.cpu().numpy()

            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            im = ax.imshow(sim, cmap="viridis")
            ax.set_title("image-text similarity matrix")
            ax.set_xlabel("texts")
            ax.set_ylabel("images")
            fig.colorbar(im, ax=ax)
            plt.show()

            print("captions in this mini-batch:")
            for i, caption in enumerate(captions[:8]):
                print(i, caption)
            """
        ),
        md(
            """
            诊断型问题：

            > 如果对角线没有明显更亮，可能是哪一步出了问题？

            可以让学生从这几个角度猜：

            - 数据配对错了。
            - encoder 太弱。
            - 训练太少。
            - batch 太小，负例不够。
            - 文本 tokenizer 把关键信息弄丢了。
            """
        ),
        code(
            """
            # ------------------------------
            # 这段代码做什么：
            # 1. 用一组文本提示词做 zero-shot 分类
            # 2. 看 toy CLIP 是否已经学会颜色 + 形状的语义对齐
            # ------------------------------
            combo_prompts = [f"a {color} {shape}" for color in colors for shape in shapes]
            combo_token_ids = torch.stack([tokenize(text) for text in combo_prompts]).to(device)

            with torch.no_grad():
                prompt_embeds = F.normalize(toy_clip.text_encoder(combo_token_ids), dim=1)

                total = 0
                correct = 0
                for images, token_ids, captions, labels in val_dl:
                    images = images.to(device)
                    image_embeds = F.normalize(toy_clip.image_encoder(images), dim=1)
                    pred = (image_embeds @ prompt_embeds.T).argmax(dim=1)
                    pred_labels = [combo_prompts[i].replace("a ", "") for i in pred.tolist()]
                    correct += sum(int(p == y) for p, y in zip(pred_labels, labels))
                    total += len(labels)

            print("toy zero-shot accuracy =", round(correct / total, 3))
            """
        ),
        code(
            """
            # ------------------------------
            # 这段代码做什么：
            # 1. 选一张 toy 图片
            # 2. 画出它和所有 prompt 的相似度排名
            # ------------------------------
            demo_image, _, demo_caption, demo_label = val_ds[0]
            demo_image_batch = demo_image.unsqueeze(0).to(device)

            with torch.no_grad():
                image_embed = F.normalize(toy_clip.image_encoder(demo_image_batch), dim=1)
                scores = (image_embed @ prompt_embeds.T).squeeze(0).cpu()

            order = scores.argsort(descending=True)
            top_prompts = [combo_prompts[i].replace("a ", "") for i in order[:8].tolist()]
            top_scores = scores[order[:8]].numpy()

            plt.figure(figsize=(8, 3))
            plt.subplot(1, 2, 1)
            plt.imshow(demo_image.permute(1, 2, 0))
            plt.title(f"true: {demo_label}")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.barh(top_prompts[::-1], top_scores[::-1])
            plt.title("top prompt similarities")
            plt.xlabel("similarity")
            plt.tight_layout()
            plt.show()

            print("original caption:", demo_caption)
            """
        ),
        code(
            """
            # ------------------------------
            # 这段代码做什么：
            # 1. 统计 toy CLIP 在 16 个颜色+形状类别上的混淆情况
            # 2. 看模型主要错在颜色、形状，还是两者都错
            # ------------------------------
            label_names = [f"{color} {shape}" for color in colors for shape in shapes]
            label_to_id = {name: i for i, name in enumerate(label_names)}
            confusion = np.zeros((len(label_names), len(label_names)), dtype=int)

            with torch.no_grad():
                for images, token_ids, captions, labels in val_dl:
                    images = images.to(device)
                    image_embeds = F.normalize(toy_clip.image_encoder(images), dim=1)
                    pred = (image_embeds @ prompt_embeds.T).argmax(dim=1)
                    pred_labels = [combo_prompts[i].replace("a ", "") for i in pred.tolist()]
                    for y_true, y_pred in zip(labels, pred_labels):
                        confusion[label_to_id[y_true], label_to_id[y_pred]] += 1

            plt.figure(figsize=(7, 6))
            plt.imshow(confusion, cmap="Blues")
            plt.xticks(range(len(label_names)), label_names, rotation=90, fontsize=7)
            plt.yticks(range(len(label_names)), label_names, fontsize=7)
            plt.xlabel("predicted")
            plt.ylabel("true")
            plt.title("Toy CLIP zero-shot confusion matrix")
            plt.colorbar()
            plt.tight_layout()
            plt.show()
            """
        ),
        code(
            """
            # ------------------------------
            # 这段代码做什么：
            # 1. 给一个文本查询，从 toy gallery 里检索最相近图像
            # 2. 对应 text-to-image retrieval
            # ------------------------------
            query_text = "a green triangle"
            gallery_images, _, gallery_captions, gallery_labels = next(iter(val_dl))
            gallery_images = gallery_images[:24].to(device)

            with torch.no_grad():
                gallery_embeds = F.normalize(toy_clip.image_encoder(gallery_images), dim=1)
                query_embed = F.normalize(toy_clip.text_encoder(tokenize(query_text).unsqueeze(0).to(device)), dim=1)
                scores = (gallery_embeds @ query_embed.T).squeeze(1)
                topk = scores.topk(4).indices.cpu().tolist()

            fig, axes = plt.subplots(1, 4, figsize=(10, 3))
            for ax, idx in zip(axes, topk):
                ax.imshow(gallery_images[idx].cpu().permute(1, 2, 0))
                ax.set_title(gallery_labels[idx], fontsize=9)
                ax.axis("off")
            plt.suptitle(f'Toy retrieval for query: "{query_text}"')
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            讨论题：看混淆矩阵时，不要只问“准确率多少”，要问：

            - 错误是不是集中在同一颜色？
            - 错误是不是集中在同一形状？
            - 如果把训练数据里的模板减少一半，模型会更像“背答案”还是更像“学组合”？

            这部分可以安排 6 到 8 分钟，让学生用自己的话解释一张图。
            """
        ),
        code(
            """
            # ------------------------------
            # 这段代码做什么：
            # 1. 测试 toy tokenizer 遇到没见过的词会怎样
            # 2. 说明 toy 模型的泛化边界
            # ------------------------------
            test_queries = [
                "a green triangle",
                "a tiny green triangle",
                "a bright green triangle",
                "a green object",
            ]

            gallery_images, _, gallery_captions, gallery_labels = next(iter(val_dl))
            gallery_images = gallery_images[:32].to(device)

            with torch.no_grad():
                gallery_embeds = F.normalize(toy_clip.image_encoder(gallery_images), dim=1)

            for query_text in test_queries:
                query_ids = tokenize(query_text).unsqueeze(0).to(device)
                with torch.no_grad():
                    query_embed = F.normalize(toy_clip.text_encoder(query_ids), dim=1)
                    scores = (gallery_embeds @ query_embed.T).squeeze(1)
                top = scores.argmax().item()
                print(f"{query_text:>24s} -> top match: {gallery_labels[top]}")

            print("\\n注意：tiny、bright、object 如果不在词表里，会被这个最小 tokenizer 忽略。")
            """
        ),
        md(
            """
            小结讨论：

            > zero-shot 分类到底神奇在哪里？又不神奇在哪里？

            可以引导学生回答：

            - 神奇：没有给新分类头训练参数，直接用文本当类别。
            - 不神奇：它本质上还是相似度比较，不是凭空冒出来的能力。
            """
        ),
        md(
            """
            # Part 4. 预训练 OpenAI CLIP：真实 demo

            现在把视角从 toy 数据换到真实模型。

            原书里的 `OpenAI_clip.ipynb` 用的是 OpenAI 官方 repo，这里改成 `transformers` 版本：

            - 环境更干净。
            - CPU 上也能直接跑。
            - 更方便和 Hugging Face 生态衔接。
            """
        ),
        md(
            """
            <img src="images/lesson6_multimodal/zero_shot_pipeline.png" width="980">

            论文证据卡：

            - CLIP 论文强调：训练后可以用自然语言引用视觉概念。
            - 课堂翻译：类别不一定要是固定分类头，也可以是一组文本 prompt。

            课堂问题：

            > `cat` 和 `a photo of a cat` 对人来说差不多，为什么对模型可能不一样？
            """
        ),
        md(
            """
            ## 论文原图：prompt engineering 的影响

            **CLIP Figure 4：prompt engineering and ensembling**

            <img src="images/lesson6_multimodal/paper_originals/clip_fig4_prompt_engineering.png" width="700">

            这张原图非常适合放在 prompt demo 前面。

            课堂问题：

            > prompt 明明只是换了说法，为什么模型结果会变？

            接下来用 CIFAR10 的 prompt 对照实验来验证这个问题。
            """
        ),
        code(
            """
            # ------------------------------
            # 这段代码做什么：
            # 1. 载入预训练 OpenAI CLIP
            # 2. 准备 CIFAR10 作为真实图像演示数据
            # ------------------------------
            cifar_val = datasets.CIFAR10(root="cifar_data", train=False, download=True)
            cifar_classes = cifar_val.classes

            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=False)
            clip_model.eval()

            print("num classes =", len(cifar_classes))
            print("classes =", cifar_classes)
            """
        ),
        code(
            """
            # ------------------------------
            # 这段代码做什么：
            # 1. 先看几张 CIFAR10 样本图
            # 2. 为后面的 zero-shot 和检索做参照
            # ------------------------------
            fig, axes = plt.subplots(2, 5, figsize=(10, 4))
            for ax, idx in zip(axes.flat, range(10)):
                image, label = cifar_val[idx]
                ax.imshow(image)
                ax.set_title(cifar_classes[label], fontsize=9)
                ax.axis("off")
            plt.tight_layout()
            plt.show()
            """
        ),
        code(
            """
            # ------------------------------
            # 这段代码做什么：
            # 1. 设计一个 prompt 挑战
            # 2. 比较不同文本模板的 zero-shot 准确率
            # ------------------------------
            eval_images = [cifar_val[i][0] for i in range(120)]
            eval_true = [cifar_classes[cifar_val[i][1]] for i in range(120)]

            prompt_sets = {
                "label_only": list(cifar_classes),
                "photo_prompt": [f"a photo of a {name}" for name in cifar_classes],
                "blurry_photo": [f"a blurry photo of a {name}" for name in cifar_classes],
                "small_object": [f"a small object that is a {name}" for name in cifar_classes],
            }

            prompt_scores = {}
            for prompt_name, texts in prompt_sets.items():
                inputs = clip_processor(text=texts, images=eval_images, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    probs = clip_model(**inputs).logits_per_image.softmax(dim=1)

                preds = [cifar_classes[i] for i in probs.argmax(dim=1).tolist()]
                acc = sum(int(p == y) for p, y in zip(preds, eval_true)) / len(eval_true)
                prompt_scores[prompt_name] = acc
                print(f"{prompt_name:>14s}: acc={acc:.3f}")

            plt.figure(figsize=(7, 3))
            plt.bar(prompt_scores.keys(), prompt_scores.values())
            plt.xticks(rotation=20, ha="right")
            plt.ylim(0, 1)
            plt.ylabel("accuracy on 120 CIFAR10 images")
            plt.title("Prompt wording changes CLIP zero-shot behavior")
            plt.tight_layout()
            plt.show()
            """
        ),
        code(
            """
            # ------------------------------
            # 这段代码做什么：
            # 1. 用多个 prompt 模板做 prompt ensemble
            # 2. 对比单模板和多模板平均的效果
            # ------------------------------
            ensemble_templates = [
                "a photo of a {}",
                "a blurry photo of a {}",
                "a small photo of a {}",
                "a close-up photo of a {}",
                "a low resolution photo of a {}",
            ]

            all_texts = [template.format(name) for template in ensemble_templates for name in cifar_classes]
            text_inputs = clip_processor(text=all_texts, return_tensors="pt", padding=True)
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

            image_inputs = clip_processor(images=eval_images, return_tensors="pt")
            image_inputs = {k: v.to(device) for k, v in image_inputs.items()}

            with torch.no_grad():
                text_features = clip_model.get_text_features(**text_inputs)
                image_features = clip_model.get_image_features(**image_inputs)

            text_features = F.normalize(text_features, dim=1)
            image_features = F.normalize(image_features, dim=1)

            n_templates = len(ensemble_templates)
            n_classes = len(cifar_classes)
            text_features = text_features.reshape(n_templates, n_classes, -1)

            single_text_features = text_features[0]
            ensemble_text_features = F.normalize(text_features.mean(dim=0), dim=1)

            single_pred = (image_features @ single_text_features.T).argmax(dim=1).cpu().tolist()
            ensemble_pred = (image_features @ ensemble_text_features.T).argmax(dim=1).cpu().tolist()

            single_acc = sum(cifar_classes[i] == y for i, y in zip(single_pred, eval_true)) / len(eval_true)
            ensemble_acc = sum(cifar_classes[i] == y for i, y in zip(ensemble_pred, eval_true)) / len(eval_true)

            print("single prompt acc   =", round(single_acc, 3))
            print("prompt ensemble acc =", round(ensemble_acc, 3))
            """
        ),
        code(
            """
            # ------------------------------
            # 这段代码做什么：
            # 1. 画出预训练 CLIP 在 CIFAR10 小样本上的混淆矩阵
            # 2. 观察它更容易混淆哪些类别
            # ------------------------------
            confusion = np.zeros((len(cifar_classes), len(cifar_classes)), dtype=int)
            for pred_id, true_name in zip(ensemble_pred, eval_true):
                true_id = cifar_classes.index(true_name)
                confusion[true_id, pred_id] += 1

            plt.figure(figsize=(6, 5))
            plt.imshow(confusion, cmap="Oranges")
            plt.xticks(range(len(cifar_classes)), cifar_classes, rotation=45, ha="right")
            plt.yticks(range(len(cifar_classes)), cifar_classes)
            plt.xlabel("predicted")
            plt.ylabel("true")
            plt.title("OpenAI CLIP zero-shot confusion on 120 CIFAR10 images")
            plt.colorbar()
            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            诊断型问题：

            > prompt 是给人看的，还是给模型看的？

            建议让学生先自己写 2 到 3 个 prompt，再替换上面 `prompt_sets` 里的模板。

            讨论重点：prompt 不是装饰，它会改变文本向量，所以会改变相似度排序。
            """
        ),
        code(
            """
            # ------------------------------
            # 这段代码做什么：
            # 1. 对几张真实图像做 zero-shot 分类
            # 2. 打印 top-3 文本提示词概率
            # ------------------------------
            sample_indices = [0, 1, 3, 8, 12]
            sample_images = [cifar_val[i][0] for i in sample_indices]
            sample_true = [cifar_classes[cifar_val[i][1]] for i in sample_indices]
            candidate_texts = [f"a photo of a {name}" for name in cifar_classes]

            inputs = clip_processor(text=candidate_texts, images=sample_images, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                probs = clip_model(**inputs).logits_per_image.softmax(dim=1).cpu()

            fig, axes = plt.subplots(1, len(sample_images), figsize=(12, 3))
            for ax, image, true_label, prob in zip(axes, sample_images, sample_true, probs):
                top_vals, top_idx = prob.topk(3)
                pred_name = cifar_classes[int(top_idx[0])]
                ax.imshow(image)
                ax.set_title(f"true={true_label}\\npred={pred_name}", fontsize=9)
                ax.axis("off")

                print("\\ntrue =", true_label)
                for value, idx in zip(top_vals.tolist(), top_idx.tolist()):
                    print(f"  {cifar_classes[idx]:>10s}: {value:.3f}")

            plt.tight_layout()
            plt.show()
            """
        ),
        code(
            """
            # ------------------------------
            # 这段代码做什么：
            # 1. 用文本查询在小型图像库里做检索
            # 2. 观察 CLIP 的 text-to-image retrieval 能力
            # ------------------------------
            gallery_indices = list(range(60))
            gallery_images = [cifar_val[i][0] for i in gallery_indices]
            gallery_labels = [cifar_classes[cifar_val[i][1]] for i in gallery_indices]
            query = "a photo of a truck"

            inputs = clip_processor(text=[query], images=gallery_images, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                scores = clip_model(**inputs).logits_per_text[0].cpu()

            top_idx = scores.topk(6).indices.tolist()

            fig, axes = plt.subplots(1, 6, figsize=(13, 2.8))
            for ax, idx in zip(axes, top_idx):
                ax.imshow(gallery_images[idx])
                ax.set_title(gallery_labels[idx], fontsize=8)
                ax.axis("off")
            plt.suptitle(f'Retrieval for query: "{query}"')
            plt.tight_layout()
            plt.show()
            """
        ),
        code(
            """
            # ------------------------------
            # 这段代码做什么：
            # 1. 反过来：给一张图片，从多个文本描述中找最匹配的一句
            # 2. 说明 image-to-text matching 和分类是同一类动作
            # ------------------------------
            image_idx = 0
            query_image, true_id = cifar_val[image_idx]
            text_bank = [
                "a photo of an airplane",
                "a photo of a ship",
                "a photo of a truck",
                "a close-up photo of a frog",
                "a photo of a cat",
                "a photo of a horse",
            ]

            inputs = clip_processor(text=text_bank, images=[query_image], return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                probs = clip_model(**inputs).logits_per_image.softmax(dim=1)[0].cpu()

            plt.figure(figsize=(2.5, 2.5))
            plt.imshow(query_image)
            plt.axis("off")
            plt.title(f"true={cifar_classes[true_id]}")
            plt.show()

            for text, prob in sorted(zip(text_bank, probs.tolist()), key=lambda x: x[1], reverse=True):
                print(f"{prob:.3f}  {text}")
            """
        ),
        code(
            """
            # ------------------------------
            # 这段代码做什么：
            # 1. 用 CLIP 做一个“多选式 VQA”示意
            # 2. 说明它本质上是答案排序，而不是自由生成
            # ------------------------------
            cat_idx = next(i for i, (_, y) in enumerate(cifar_val) if cifar_classes[y] == "cat")
            qa_image, _ = cifar_val[cat_idx]
            answer_choices = [
                "the answer is cat",
                "the answer is dog",
                "the answer is bird",
                "the answer is horse",
            ]

            inputs = clip_processor(text=answer_choices, images=[qa_image], return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                answer_probs = clip_model(**inputs).logits_per_image.softmax(dim=1)[0].cpu()

            plt.figure(figsize=(2.5, 2.5))
            plt.imshow(qa_image)
            plt.axis("off")
            plt.title("Multiple-choice VQA style example")
            plt.show()

            for choice, prob in zip(answer_choices, answer_probs.tolist()):
                print(f"{choice:>22s}: {prob:.3f}")
            """
        ),
        md(
            """
            课堂追问：

            > CLIP 选出了 `the answer is cat`，这算不算真正理解了问题？

            这里要明确区分：

            - CLIP 可以做一些“像问答”的事情。
            - 但它的机制仍然是“图片向量 vs 候选答案向量”的排序。
            - 所以它更像 answer selection，不是现代视觉语言模型那种自由生成式 VQA。
            """
        ),
        md(
            """
            ## 从 CLIP 到现代 VLM：再看三张论文图

            到这里可以插入三张原论文图，帮助学生理解：CLIP 是很重要的地基，但现代视觉语言模型还多了“桥”和“语言生成”。

            1. **BLIP-2 架构图**：看 Q-Former 怎么站在冻结图像编码器和冻结大语言模型中间。  
               课堂问题：为什么不直接把图片向量塞进大语言模型？

            2. **Flamingo 架构图**：看视觉特征如何通过 cross-attention 进入语言模型。  
               课堂问题：如果输入里有多张图和多段文字，模型应该怎么知道每句话对应哪张图？

            3. **LLaVA 流程图**：看 CLIP 视觉编码器、投影层、语言模型和指令数据如何连起来。  
               课堂问题：为什么“会匹配图文”还不等于“会聊天回答问题”？

            一句话总结：

            > CLIP 解决“图文能不能对齐”；VLM 还要解决“视觉信息怎么进入语言模型，并按人类指令回答”。
            """
        ),
        md(
            """
            ## 论文原图：BLIP-2、Flamingo、LLaVA

            **BLIP-2 Figure 1：整体框架**

            <img src="images/lesson6_multimodal/paper_originals/blip2_fig1_framework.png" width="760">

            **BLIP-2 Figure 2：Q-Former**

            <img src="images/lesson6_multimodal/paper_originals/blip2_fig2_qformer.png" width="980">

            这两张图要讲清楚：BLIP-2 不是简单把图片丢给 LLM，而是用 Q-Former 做中间桥。

            **Flamingo Figure 3：VLM 架构**

            <img src="images/lesson6_multimodal/paper_originals/flamingo_fig3_architecture.png" width="980">

            这张图重点看：视觉编码器和语言模型很多部分是冻结的，中间通过 cross-attention 类模块连接。

            **Flamingo Figure 1：少样本视觉语言任务效果**

            <img src="images/lesson6_multimodal/paper_originals/flamingo_fig1_examples.png" width="760">

            这张效果图适合让学生观察：同一个模型如何处理分类、问答、描述等不同任务。

            **LLaVA Figure 1：视觉指令微调架构**

            <img src="images/lesson6_multimodal/paper_originals/llava_fig1_architecture.png" width="800">

            这张图重点看：CLIP 视觉编码器输出图像特征，再通过投影层接入语言模型。

            **LLaVA Figure 2：根据草图生成网页**

            <img src="images/lesson6_multimodal/paper_originals/llava_fig2_demo.png" width="780">

            这张效果图适合讨论：

            > 为什么从“图文匹配”走到“按指令生成答案/代码”，中间还需要指令数据和语言模型？
            """
        ),
        md(
            """
            # Part 5. CLIP 的局限：它会怎么翻车？

            <img src="images/lesson6_multimodal/clip_failure_cases.png" width="980">

            辩论型问题：

            > CLIP 是在理解图像，还是在做高级匹配？

            这不是非黑即白的问题。更好的说法是：

            > CLIP 学到了很强的图文匹配能力，但“匹配很强”不等于“所有视觉推理都很强”。
            """
        ),
        code(
            """
            # ------------------------------
            # 这段代码做什么：
            # 1. 构造两个只有左右关系不同的合成图片
            # 2. 测试 CLIP 对 left/right 关系是否稳定
            # ------------------------------
            def render_two_shape_scene(red_left=True, size=224):
                img = Image.new("RGB", (size, size), "white")
                draw = ImageDraw.Draw(img)
                y = size // 2
                left_x, right_x = size // 3, 2 * size // 3
                red_x = left_x if red_left else right_x
                blue_x = right_x if red_left else left_x

                draw.ellipse((red_x - 32, y - 32, red_x + 32, y + 32), fill="red")
                draw.rectangle((blue_x - 32, y - 32, blue_x + 32, y + 32), fill="blue")
                return img


            spatial_images = [
                render_two_shape_scene(red_left=True),
                render_two_shape_scene(red_left=False),
            ]
            spatial_texts = [
                "a red circle to the left of a blue square",
                "a red circle to the right of a blue square",
                "a blue square to the left of a red circle",
                "a blue square to the right of a red circle",
            ]

            inputs = clip_processor(text=spatial_texts, images=spatial_images, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                probs = clip_model(**inputs).logits_per_image.softmax(dim=1).cpu()

            fig, axes = plt.subplots(1, 2, figsize=(7, 3))
            for ax, image, title in zip(axes, spatial_images, ["red left", "red right"]):
                ax.imshow(image)
                ax.set_title(title)
                ax.axis("off")
            plt.tight_layout()
            plt.show()

            for image_name, prob in zip(["red left image", "red right image"], probs):
                print("\\n", image_name)
                for text, value in sorted(zip(spatial_texts, prob.tolist()), key=lambda x: x[1], reverse=True):
                    print(f"{value:.3f}  {text}")
            """
        ),
        md(
            """
            讨论题：

            > 如果模型看到了 red、circle、blue、square，但 left/right 判断不稳定，这说明它缺的是视觉能力，还是语言关系理解能力？

            引导学生注意：很多图文模型对“物体是什么”很强，但对“谁在谁左边、数量是多少、细小文字写了什么”不一定稳定。
            """
        ),
        code(
            """
            # ------------------------------
            # 这段代码做什么：
            # 1. 构造一个“候选答案都不完美”的多选题
            # 2. 观察 CLIP 仍然会给出一个最高分
            # ------------------------------
            image_idx = next(i for i, (_, y) in enumerate(cifar_val) if cifar_classes[y] == "ship")
            tricky_image, true_id = cifar_val[image_idx]
            tricky_choices = [
                "the image shows two red cars",
                "the image shows a large animal",
                "the image shows an indoor bedroom",
                "the image shows a vehicle or vessel",
            ]

            inputs = clip_processor(text=tricky_choices, images=[tricky_image], return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                probs = clip_model(**inputs).logits_per_image.softmax(dim=1)[0].cpu()

            plt.figure(figsize=(2.5, 2.5))
            plt.imshow(tricky_image)
            plt.axis("off")
            plt.title(f"true={cifar_classes[true_id]}")
            plt.show()

            for choice, prob in sorted(zip(tricky_choices, probs.tolist()), key=lambda x: x[1], reverse=True):
                print(f"{prob:.3f}  {choice}")
            """
        ),
        md(
            """
            讨论提示：

            > 如果所有候选答案都不准确，CLIP 会不会说“我不知道”？

            通常不会。因为这里用的是 softmax 排序，它总会把概率分出去。  
            这就是为什么实际系统里还需要阈值、拒答机制、人工检查或更强的视觉语言模型。
            """
        ),
        md(
            """
            # Part 6. CLIP 编码器和 T5 编码器各负责什么

            <img src="images/lesson6_multimodal/encoder_roles.png" width="980">

            一句话版：

            > **CLIP 像一个会判断“图文像不像一对”的裁判；T5 编码器像一个读题很细的语文老师。**

            它们都叫 encoder，都是把输入变成向量，但后面的模型拿这些向量做的事不一样。

            - **CLIP image encoder**：把图片变成图像向量。
            - **CLIP text encoder**：把短文本、类别名、prompt 模板变成文本向量。
            - **CLIP 的强项**：图像向量和文本向量可以直接比相似度。
            - **T5 encoder**：把较长提示词变成一串 token 级别的语言表示。
            - **T5 的强项**：保留“谁修饰谁、位置关系、文字内容、长描述”等细节。
            """
        ),
        md(
            """
            ## T5 / Imagen / SD3 论文证据卡

            <img src="images/lesson6_multimodal/imagen_text_encoder_card.png" width="980">

            论文依据：

            1. [T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)  
               T5 把很多语言任务统一成“输入文本，输出文本”的形式，形成了很强的语言编码能力。
            2. [Imagen: Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding](https://arxiv.org/abs/2205.11487)  
               Imagen 使用冻结的 T5 文本编码器作为图像生成条件，并强调强文本编码器对复杂提示词很重要。
            3. [Stable Diffusion 3: Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206)  
               SD3 同时使用 CLIP 系列文本表示和 T5 文本表示，说明二者更像互补，而不是谁替代谁。
            """
        ),
        md(
            """
            ## 论文原图：T5、Imagen、SD3

            **T5 Figure 1：text-to-text 框架**

            <img src="images/lesson6_multimodal/paper_originals/t5_fig1_text_to_text.png" width="900">

            这张图要讲清楚：T5 把翻译、分类、问答、摘要等任务都写成“输入文本，输出文本”。

            **Imagen Figure 1：生成效果图**

            <img src="images/lesson6_multimodal/paper_originals/imagen_fig1_samples.png" width="920">

            这张图适合让学生观察：prompt 里的物体、风格、场景细节是否被保留下来。

            **Imagen Figure 4：关键实验发现**

            <img src="images/lesson6_multimodal/paper_originals/imagen_fig4_findings.png" width="920">

            这张图适合引出：强文本编码器对文生图效果很重要。

            **Stable Diffusion 3 Figure 1：高分辨率效果图**

            <img src="images/lesson6_multimodal/paper_originals/sd3_fig1_samples.png" width="900">

            **Stable Diffusion 3 Figure 2：模型架构图**

            <img src="images/lesson6_multimodal/paper_originals/sd3_fig2_architecture.png" width="980">

            这张架构图适合重点看：CLIP 和 T5 的文本表示为什么会同时出现在一个生成模型里。
            """
        ),
        md(
            """
            这里建议打开三类原论文图：

            1. **T5 的 text-to-text 框架图**  
               看所有任务如何被写成“输入一段文本，输出一段文本”。  
               课堂问题：如果所有任务都能写成文本输入，T5 encoder 为什么会很适合读 prompt？

            2. **Imagen 的结构图或文本编码器对比图**  
               看冻结 T5 文本编码器如何给扩散模型提供条件。  
               课堂问题：复杂 prompt 里，哪些信息最怕被文本编码器读丢？

            3. **SD3 的架构图**  
               看 CLIP 和 T5 的文本表示如何一起参与生成。  
               课堂问题：如果两个 encoder 都保留，是不是说明它们在做不同的事？

            这一组图的讲法要很直白：

            > CLIP 帮模型判断“像不像”；T5 帮模型读清楚“到底要什么”。
            """
        ),
        md(
            """
            公式直觉：

            ```latex
            z_{clip} = CLIPText(short\\ prompt)
            ```

            ```latex
            z_{t5} = T5(long\\ prompt)
            ```

            ```latex
            image = Generator(noise, z_{clip}, z_{t5})
            ```

            通俗解释：

            - CLIP 帮模型知道“总体上像不像这句话”。
            - T5 帮模型读清楚“这句话到底要求什么”。

            辩论型问题：

            > 如果一个文生图模型只能保留 CLIP 或 T5 一个，你会选谁？哪些 prompt 最容易先出问题？
            """
        ),
        md(
            """
            # Part 7. 从 CLIP 走到 ImageBind

            如果 CLIP 的关键思想是“把图像和文本放进同一个空间”，自然问题就是：

            > **这个空间能不能不只接图像和文本，还接音频、深度、热成像甚至 IMU？**

            ImageBind 的回答是：可以，而且不一定要求所有模态都两两成对出现。
            """
        ),
        md(
            """
            <img src="images/lesson6_multimodal/imagebind_map.png" width="980">

            论文证据卡：

            - 论文：Girdhar et al., 2023, [ImageBind: One Embedding Space To Bind Them All](https://arxiv.org/abs/2305.05665)
            - 关键结论：把图像、文字、声音、深度、热成像、IMU 放到一个联合空间里；不一定需要所有模态两两配对，围绕图像配对也能把模态“绑”起来。
            - 课堂翻译：CLIP 证明了“图片和文字可以对齐”，ImageBind 继续问“更多感官能不能也对齐”。

            开放问题：

            > 如果图片、文字、声音、深度、动作传感器都能放进同一个空间，未来会出现什么新应用？
            """
        ),
        md(
            """
            ## 论文原图：ImageBind 的多模态空间

            **ImageBind Figure 1：多模态能力效果图**

            <img src="images/lesson6_multimodal/paper_originals/imagebind_fig1_capabilities.png" width="980">

            这张图适合讲“效果”：音频、图像、深度、文本等模态进入同一个空间后，可以做跨模态检索和组合。

            **ImageBind Figure 2：方法总览图**

            <img src="images/lesson6_multimodal/paper_originals/imagebind_fig2_overview.png" width="980">

            这张图适合讲“方法”：不同模态不一定两两配对，只要都能围绕图像对齐，就有机会被绑到同一空间。
            """
        ),
        code(
            """
            # ------------------------------
            # 这段代码做什么：
            # 1. 给出一个 ImageBind 的最小可选演示入口
            # 2. 默认不运行，以免当前环境被安装和大模型拖慢
            # ------------------------------
            if RUN_IMAGEBIND:
                print("请参考 ImageBind 官方示例安装依赖并执行。")
                print("这一格保留的目的是把 ImageBind 放进课程主线，而不是在 CPU 环境里强行跑大模型。")
            else:
                print("跳过 ImageBind 实际推理。")
                print("默认讲解重点放在 CLIP 的共享空间思想，以及它如何扩展到更多模态。")
            """
        ),
        md(
            """
            # 课堂收束活动：把论文原图讲给别人听

            这一段建议用 12-15 分钟。每组任选一张论文原图，不要求读完整篇论文，只要求把图讲清楚。

            讲图时按下面 4 句话来：

            1. 这张图在回答什么问题？
            2. 图里最重要的两个模块是什么？
            3. 作者想让读者相信什么结论？
            4. 如果我是工程师，我能从这张图里学到什么做法？

            可选图：

            - CLIP Figure 1：图文对比训练怎样接到 zero-shot 分类。
            - ALIGN Figure 1：为什么双编码器适合大规模图文数据。
            - SigLIP Figure 1：为什么 loss 的写法会影响训练规模。
            - BLIP-2 Figure 1/2：Q-Former 为什么像“翻译接口”。
            - Flamingo Figure 3：冻结视觉模型和冻结语言模型，中间怎么接。
            - LLaVA Figure 1：视觉指令微调为什么让模型更像聊天助手。
            - Stable Diffusion 3 Figure 2：为什么 CLIP 和 T5 会同时出现。
            - ImageBind Figure 2：为什么围绕图像对齐，可以扩展到更多模态。
            """
        ),
        md(
            """
            # 课堂小测：8 个问题检查有没有听懂

            建议 8 分钟独立作答，随后 5 分钟同桌互改。答案不要求术语漂亮，能讲清楚即可。

            1. zero-shot 图像分类，在 CLIP 里本质上是在比较什么？
            2. 一个 4x4 图文相似度矩阵里，对角线通常代表什么？
            3. 为什么 `a photo of a cat` 和 `cat` 可能得到不同结果？
            4. CLIP 为什么适合做“多选题”，但不适合直接自由生成一段回答？
            5. ALIGN 的数据很噪声，为什么仍然可能训练出有用表示？
            6. SigLIP 相比普通 softmax 对比学习，想解决什么训练问题？
            7. BLIP-2 / Flamingo / LLaVA 都在做“连接视觉和语言”，它们连接方式有什么共同点？
            8. 文生图模型里同时使用 CLIP 和 T5，直觉上分别帮了什么忙？

            参考答案方向：

            - 不背定义，先说“图像向量和文本向量谁更接近”。
            - 遇到架构图，先找输入、桥接模块、输出。
            - 遇到效果图，先问“作者想证明模型会什么”。
            - 遇到公式，先问“它在奖励什么、惩罚什么”。
            """
        ),
        md(
            """
            # 课后作业：三档任选

            **基础档：prompt 对比实验**

            在真实 CLIP demo 中，给同一批 CIFAR-10 图片设计 3 套 prompt：

            - 极简版：`cat`
            - 图片描述版：`a photo of a cat`
            - 场景增强版：`a clear photo of a cat in the real world`

            交付：记录每套 prompt 的准确率，并写 150 字解释为什么会变。

            **进阶档：toy CLIP 扩展实验**

            在 toy 数据里增加一个新属性，例如大小 `small/large` 或背景色 `white/gray`。重新训练后观察：

            - loss 是否更难下降？
            - 混淆矩阵里最容易错的是颜色、形状，还是新属性？
            - 模型能不能处理训练中没见过的新组合？

            交付：1 张混淆矩阵 + 200 字分析。

            **挑战档：论文原图 5 分钟讲解**

            从本课 10 篇论文里任选 1 篇，找 1 张原图，做 5 分钟讲解：

            - 先说这张图要解决什么问题。
            - 再说图里每个模块在干什么。
            - 最后说它和 CLIP 思想有什么关系。

            交付：一页讲图提纲。可以截图论文原图，但要标明论文标题和 Figure 编号。
            """
        ),
        md(
            """
            # 小结：8 句话收住全课

            1. 多模态模型的核心目标，是把不同模态映射到同一个语义空间里。
            2. CLIP 通过图文对比学习，让图像和文本的相似度变成了可直接使用的推理信号。
            3. zero-shot 分类，本质上是“图像 vs 一组文本提示词”的相似度比较。
            4. 图文检索和多选式问答，本质上也是跨模态排序。
            5. prompt 模板会影响文本表示，所以 prompt engineering 会影响 zero-shot 效果。
            6. CLIP 很强，但它更像匹配器；遇到计数、细粒度关系、复杂推理时要小心。
            7. T5 编码器主要提供“提示词到底说了什么”的细粒度语言条件。
            8. ImageBind 延续了 CLIP 的思路，把共享空间扩展到了更多模态。

            ## 课后继续追问

            - 如果 toy CLIP 只在有限模板上训练，它的泛化会卡在哪？
            - 为什么 CLIP 能做多选式 VQA，却不擅长自由生成答案？
            - 如果一个文生图模型去掉 T5，只保留 CLIP，哪些 prompt 最可能先出问题？
            - ImageBind 如果继续往前走，和现代 VLM / 多模态大模型之间是什么关系？
            """
        ),
    ]

    nb = new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
    nbformat.write(nb, NOTEBOOK_PATH)


def main():
    build_images()
    build_notebook()
    print(f"wrote {NOTEBOOK_PATH}")
    print(f"images in {IMAGE_DIR}")


if __name__ == "__main__":
    main()
