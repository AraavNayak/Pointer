"""Mask helpers shared by the server, the SAM precompute script and the benchmark."""

import cv2
import numpy as np


def mask_to_record(mask: np.ndarray) -> dict:
    """Binary mask (H, W) -> {bbox, centroid, area, polygons} in pixel coords. {} if empty."""
    mask = (mask > 0).astype(np.uint8)
    if mask.sum() < 4:
        return {}
    ys, xs = np.nonzero(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) < 6:
            continue
        c = cv2.approxPolyDP(c, 1.0, True)
        polys.append(c.reshape(-1, 2).tolist())
    return {
        "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "centroid": [int(xs.mean()), int(ys.mean())],
        "area": int(mask.sum()),
        "polygons": polys,
    }


def record_to_mask(rec: dict, width: int, height: int) -> np.ndarray:
    """Rasterize a record's polygons back into a binary mask."""
    m = np.zeros((height, width), np.uint8)
    for p in rec.get("polygons", []):
        cv2.fillPoly(m, [np.array(p, np.int32)], 1)
    return m


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 1.0


def box_iou(a, b) -> float:
    """Boxes as [x1, y1, x2, y2] inclusive pixel coords."""
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xb - xa + 1) * max(0, yb - ya + 1)
    area = lambda r: max(0, r[2] - r[0] + 1) * max(0, r[3] - r[1] + 1)
    union = area(a) + area(b) - inter
    return inter / union if union else 0.0
