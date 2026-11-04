"""Pointing-accuracy benchmark: raw Muse Spark coordinates vs. SAM 3.1 grounding.

Ground truth comes from MuJoCo's segmentation render, so it is exact per pixel.

  A. Raw LLM pointing  - Muse Spark sees one frame and returns a bounding box for each object.
  B. SAM 3.1 tracking  - boxes from the precomputed SAM 3.1 masks (prompted once, on frame 0).

Both are scored with box IoU against ground truth on the same frames. SAM 3.1 is
also scored with mask IoU on every frame. Results go to eval/results.json and a
table is printed for the slide.

Run (after sam/precompute_masks.py):  python -m eval.benchmark [n_frames]
"""

import json
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from server import agent, meta_api  # noqa: E402
from server.masks import box_iou, mask_iou, record_to_mask  # noqa: E402

BOX_SCHEMA = {
    "type": "object",
    "properties": {
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "enum": ["red cube", "gripper"]},
                    "visible": {"type": "boolean"},
                    "x1": {"type": "integer"}, "y1": {"type": "integer"},
                    "x2": {"type": "integer"}, "y2": {"type": "integer"},
                },
                "required": ["name", "visible", "x1", "y1", "x2", "y2"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["objects"],
    "additionalProperties": False,
}


def raw_spark_boxes(jpeg: bytes, w: int, h: int) -> dict:
    out = meta_api.spark_json(
        "You locate objects in images precisely.",
        [meta_api.image_part(jpeg),
         {"type": "input_text", "text": (
             f"The image is {w}x{h} pixels, origin top-left. Return a tight bounding box in pixel "
             "coordinates for the red cube and for the robot gripper (the whole gripper including "
             "its vertical mast and fingers). Set visible=false if an object is not in view.")}],
        BOX_SCHEMA, "boxes")
    names = {"red cube": "1", "gripper": "2"}
    return {names[o["name"]]: [o["x1"], o["y1"], o["x2"], o["y2"]]
            for o in out.get("objects", []) if o.get("visible") and o.get("name") in names}


def main(n_frames: int = 24):
    ep = agent.EP_DIR
    meta = agent.meta()
    W, H = meta["width"], meta["height"]
    gt = json.loads((ep / "gt_masks.json").read_text())["frames"]
    sam_path = ep / "sam_masks.json"
    if not sam_path.exists():
        sys.exit("Run `python -m sam.precompute_masks` first.")
    sam = json.loads(sam_path.read_text())["frames"]

    # Mask IoU for SAM 3.1 across every frame
    sam_mask_iou = {"1": [], "2": []}
    for fi, objs in gt.items():
        for oid, g in objs.items():
            if not g:
                continue
            s = sam.get(fi, {}).get(oid) or {}
            sam_mask_iou[oid].append(mask_iou(record_to_mask(g, W, H), record_to_mask(s, W, H)) if s else 0.0)

    # Box IoU on a sample of frames: raw Muse Spark vs SAM 3.1
    idxs = np.linspace(0, meta["n_frames"] - 1, n_frames).round().astype(int).tolist()
    rows = []
    for fi in idxs:
        g = gt[str(fi)]
        try:
            raw = raw_spark_boxes(agent.frame_jpeg(fi, 95), W, H)
        except Exception as e:
            print(f"frame {fi}: Muse Spark error {e}")
            raw = {}
        for oid in ("1", "2"):
            if not g.get(oid):
                continue
            s = sam.get(str(fi), {}).get(oid) or {}
            rows.append({
                "frame": fi, "object": meta["objects"][oid],
                "raw_spark_iou": box_iou(raw[oid], g[oid]["bbox"]) if oid in raw else 0.0,
                "sam31_iou": box_iou(s["bbox"], g[oid]["bbox"]) if s else 0.0,
            })
        print(f"frame {fi:3d} " + "  ".join(
            f"{r['object']}: raw {r['raw_spark_iou']:.2f} / sam {r['sam31_iou']:.2f}"
            for r in rows if r["frame"] == fi))

    def mean(xs):
        return float(np.mean(xs)) if xs else 0.0

    def hit(xs, thr=0.5):
        return float(np.mean([x >= thr for x in xs])) if xs else 0.0

    summary = {}
    for name in ["red cube", "gripper", "all"]:
        sel = [r for r in rows if name == "all" or r["object"] == name]
        summary[name] = {
            "frames": len(sel),
            "raw_spark_box_iou": round(mean([r["raw_spark_iou"] for r in sel]), 3),
            "sam31_box_iou": round(mean([r["sam31_iou"] for r in sel]), 3),
            "raw_spark_hit@0.5": round(hit([r["raw_spark_iou"] for r in sel]), 3),
            "sam31_hit@0.5": round(hit([r["sam31_iou"] for r in sel]), 3),
        }
    summary["sam31_mask_iou_all_frames"] = {meta["objects"][k]: round(mean(v), 3) for k, v in sam_mask_iou.items()}
    summary["models"] = {"reasoning": meta_api.REASONING_MODEL, "segmentation": meta_api.SAM_MODEL}

    (ROOT / "eval" / "results.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print("\n                 box IoU (mean)      hit rate (IoU >= 0.5)")
    print("                 raw Spark  SAM 3.1  raw Spark  SAM 3.1")
    for name in ["red cube", "gripper", "all"]:
        s = summary[name]
        print(f"{name:<16} {s['raw_spark_box_iou']:>9.2f} {s['sam31_box_iou']:>8.2f} "
              f"{s['raw_spark_hit@0.5']:>10.0%} {s['sam31_hit@0.5']:>8.0%}")
    print(f"SAM 3.1 mask IoU over all frames: {summary['sam31_mask_iou_all_frames']}")


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
