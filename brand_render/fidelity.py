"""Deterministic geometry check: does the render keep the sketch's lines?

SketchUp screenshots are line-heavy, so comparing edge maps is a cheap, model-free test that
catches moved walls, changed camera, or added/removed objects. Score = F1 of edge pixels matched
within a small tolerance (dilation), after resizing both images to the same size.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter, ImageOps


def _edges(img: Image.Image, size=(768, 512)) -> np.ndarray:
    g = ImageOps.grayscale(img.resize(size, Image.LANCZOS))
    g = ImageOps.autocontrast(g.filter(ImageFilter.GaussianBlur(1.2)))
    e = np.asarray(g.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    thr = max(np.percentile(e, 90), 20)
    return e > thr


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    img = Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(2 * r + 1))
    return np.asarray(img) > 0


def edge_fidelity(sketch: Image.Image, render: Image.Image, tol_px: int = 4) -> float:
    w, h = sketch.size
    size = (768, max(64, int(768 * h / w)))
    a, b = _edges(sketch, size), _edges(render, size)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    recall = (a & _dilate(b, tol_px)).sum() / a.sum()      # sketch lines present in render
    precision = (b & _dilate(a, tol_px)).sum() / b.sum()   # render lines explained by sketch
    # Renders legitimately add texture edges (wood grain, tiles), so weight recall higher (F2 score).
    return float((5 * precision * recall) / (4 * precision + recall + 1e-9))


def overlay(sketch: Image.Image, render: Image.Image) -> Image.Image:
    """Red = sketch edges over the render, for visual inspection in the app."""
    r = render.resize(sketch.size).convert("RGB")
    e = _edges(sketch, sketch.size)
    arr = np.asarray(r).copy()
    arr[e] = [230, 30, 30]
    return Image.fromarray(arr)
