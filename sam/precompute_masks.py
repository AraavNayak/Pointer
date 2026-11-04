"""Track the cube and the gripper through the episode video with SAM 3.1.

Prompts SAM 3.1 with one box per object on frame 0 (taken from the simulator's
first frame) and lets it propagate forward through all frames. Writes
episodes/<id>/sam_masks.json in the same format the app reads, plus the raw
special-token output for debugging.

Run:  python -m sam.precompute_masks
"""

import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from server import meta_api  # noqa: E402
from server.masks import mask_to_record  # noqa: E402


def main(episode_id: str = "pick_place_drop_001"):
    ep = ROOT / "episodes" / episode_id
    meta = json.loads((ep / "meta.json").read_text())
    gt = json.loads((ep / "gt_masks.json").read_text())
    W, H = meta["width"], meta["height"]
    boxes = {oid: rec["bbox"] for oid, rec in gt["frames"]["0"].items() if rec}
    print(f"SAM 3.1 tracking {list(meta['objects'].values())} from frame-0 boxes {boxes}")

    t0 = time.time()
    per_frame = meta_api.sam_track_video((ep / "video.mp4").read_bytes(), boxes, W, H,
                                         raw_out_path=str(ep / "sam_raw.txt"))
    dt = time.time() - t0
    out = {"source": meta_api.SAM_MODEL, "width": W, "height": H, "fps": meta["fps"],
           "objects": meta["objects"], "prompt_boxes": boxes, "latency_s": round(dt, 1),
           "frames": {}}
    for fi in range(meta["n_frames"]):
        objs = per_frame.get(fi, {})
        out["frames"][str(fi)] = {oid: mask_to_record(m) for oid, m in objs.items()}
    (ep / "sam_masks.json").write_text(json.dumps(out))
    covered = sum(1 for f in out["frames"].values() if f)
    print(f"done in {dt:.1f}s: masks on {covered}/{meta['n_frames']} frames -> {ep / 'sam_masks.json'}")


if __name__ == "__main__":
    main(*sys.argv[1:])
