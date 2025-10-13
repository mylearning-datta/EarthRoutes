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
            lines.append(f"![{heading}]({image_rel})")
            # Optional human-written caption under the figure
            if idx < len(image_captions) and image_captions[idx]:
                lines.append("")
                lines.append(f"_Figure: {image_captions[idx]}_")
            lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


