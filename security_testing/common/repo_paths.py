"""Resolve repository root and common paths."""

from pathlib import Path


def find_repo_root() -> Path:
    p = Path(__file__).resolve()
    for d in [p.parent, *p.parents]:
        if (d / "configs" / "base.yaml").is_file():
            return d
    raise RuntimeError("Could not locate repo root (configs/base.yaml missing)")


def src_dir() -> Path:
    return find_repo_root() / "src"
