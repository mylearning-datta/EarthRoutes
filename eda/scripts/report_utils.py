from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def ensure_directories(paths: Iterable[os.PathLike]) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def get_project_root(current_file: str) -> Path:
    p = Path(current_file).resolve()
    # current -> scripts -> eda -> project
    return p.parents[2]


def save_figure(fig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=200)
    

def _figure_html(image_rel: str, alt: str, caption: str | None = None, width_px: int = 720) -> list[str]:
    """
    Render a centered figure block for Markdown reports.
    Many Markdown renderers left-align images by default; HTML ensures consistent alignment.
    """
    lines: list[str] = []
    lines.append("<figure style=\"text-align: center;\">")
    lines.append(
        f"<img src=\"{image_rel}\" alt=\"{alt}\" style=\"max-width: 100%; width: {width_px}px; height: auto;\" />"
    )
    if caption:
        lines.append(f"<figcaption><em>Figure: {caption}</em></figcaption>")
    lines.append("</figure>")
    return lines


def write_markdown(output_path: Path, title: str, sections: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [f"# {title}", ""]
    for section in sections:
        heading = section.get("heading")
        content = section.get("content", "")
        images = section.get("images", [])
        image_captions = section.get("image_captions", [])
        if heading:
            lines.append(f"## {heading}")
        if content:
            lines.extend([content, ""]) 
        for idx, image_rel in enumerate(images):
            caption = image_captions[idx] if idx < len(image_captions) else ""
            alt = str(heading or "Figure")
            lines.extend(_figure_html(image_rel=image_rel, alt=alt, caption=caption or None))
            lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


