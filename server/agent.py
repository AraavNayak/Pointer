"""Episodes, the evidence Muse Spark can read, and the final pointing answer.

Muse Spark never guesses pixel coordinates for tracked objects. Its answer
carries *references* (a timestamp, object ids, a telemetry signal, a log line
number) and the front end turns them into exact annotations using SAM 3.1 masks.
For anything that is not tracked, it may give a raw pixel point, which SAM 3.1
segments live (/api/refine).

The investigation that leads up to the answer lives in investigator.py.
"""

import io
import json
import os
from functools import cached_property, lru_cache
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
from PIL import Image

from . import meta_api

ROOT = Path(__file__).resolve().parent.parent
EPISODES_DIR = ROOT / "episodes"
DEFAULT_EPISODE = os.environ.get("POINTER_EPISODE", "pick_place_drop_001")

# Context size knobs. Images dominate request size and latency.
FRAME_WIDTH = int(os.environ.get("POINTER_FRAME_WIDTH", "320"))          # downscaled from 640

SUGGESTIONS = [
    "Why did this run fail?",
    "What happened right before the object fell?",
    "Was this a hardware fault or a planning problem?",
    "What should we change so this doesn't happen again?",
]

SIGNALS = {
    "finger_L_cmd_force_N": "left finger actuator force (N)",
    "finger_R_cmd_force_N": "right finger actuator force (N)",
    "contact_L_normal_N": "left pad contact normal force on the object (N)",
    "contact_R_normal_N": "right pad contact normal force on the object (N)",
    "finger_L_temp_C": "left finger driver temperature (C)",
    "cube_slip_mm": "object displacement relative to the grasp (mm)",
    "finger_L_pos_mm": "left finger closure (mm, 0 = open)",
    "finger_R_pos_mm": "right finger closure (mm, 0 = open)",
    "cube_z": "object height above the table (m)",
    "gripper_z": "gripper height (m)",
    "gripper_x": "gripper position along the pick-to-bin axis (m)",
    "obstacle_contact_N": "contact force between the carried object or gripper and any obstacle (N)",
}

ANSWER_SYSTEM = """You are Pointer, an agent that debugs robot failures. You have finished investigating one \
recorded episode, and the evidence you gathered is below. Now you answer the engineer out loud while \
pointing at that evidence on their screen.

Return 2 to 5 segments. Each segment is one or two short spoken sentences (under 30 words total, plain \
words, no markdown; write numbers as digits but spell out units, like 12 newtons, 80 degrees, 3.6 seconds; give times \
in seconds, never milliseconds) \
plus the evidence to point at while it is spoken:
- time_ms: the episode time to show (the video seeks there and the charts draw a cursor there).
- objects: ids of tracked objects to outline in the video at that time ("1" = the carried object, \
"2" = the gripper). Empty list if none.
- signal: one telemetry column to highlight, or "none".
- log_line: the number of a log line you read, or -1.
- point: a pixel location in a frame you looked at, for something that is NOT a tracked object \
(for example one finger pad or the edge of an obstacle), as {"x":..,"y":..,"label":".."}, in the pixels \
of the frames you were shown; x = -1 when not needed.

After the segments, give a takeaway:
- root_cause: one sentence naming the first thing that went wrong and why it mattered.
- fixes: one to three concrete engineering changes that would have prevented this failure, each a short \
imperative sentence naming the component and the change. Empty only if the question is not about what \
went wrong.
- say: one or two spoken sentences, under 35 words, stating the root cause and the most important fix.
- sim_patch: the fix you would test first, as simulator controller settings. The engineer's simulator \
will re-run the episode with them. Keep a setting at its current value if you do not want to change it. \
"why" is one sentence on what the change should do.

Rules:
- Only cite evidence that appears in your investigation. Timestamps must be real times from the telemetry \
or log. Log line numbers must be ones you read.
- Order segments as a causal chain: what went wrong first, then what it caused, then the visible outcome. \
Keep fixes for the takeaway.
- Prefer pointing at all three views when the evidence exists in all three.
- If the question asks about something the evidence does not show, say so plainly in one segment.
"""


def answer_schema(patch_keys: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "segments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "say": {"type": "string"},
                        "time_ms": {"type": "integer"},
                        "objects": {"type": "array", "items": {"type": "string", "enum": ["1", "2"]}},
                        "signal": {"type": "string", "enum": list(SIGNALS) + ["none"]},
                        "log_line": {"type": "integer"},
                        "point": {
                            "type": "object",
                            "properties": {"x": {"type": "integer"}, "y": {"type": "integer"},
                                           "label": {"type": "string"}},
                            "required": ["x", "y", "label"],
                            "additionalProperties": False,
                        },
                    },
                    "required": ["say", "time_ms", "objects", "signal", "log_line", "point"],
                    "additionalProperties": False,
                },
            },
            "takeaway": {
                "type": "object",
                "properties": {
                    "root_cause": {"type": "string"},
                    "fixes": {"type": "array", "items": {"type": "string"}},
                    "say": {"type": "string"},
                    "sim_patch": patch_schema(patch_keys),
                },
                "required": ["root_cause", "fixes", "say", "sim_patch"],
                "additionalProperties": False,
            },
        },
        "required": ["segments", "takeaway"],
        "additionalProperties": False,
    }


def patch_schema(keys: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {**{k: {"type": "number"} for k in keys}, "why": {"type": "string"}},
        "required": [*keys, "why"],
        "additionalProperties": False,
    }


SHORT = {
    "finger_L_cmd_force_N": "left actuator force", "finger_R_cmd_force_N": "right actuator force",
    "contact_L_normal_N": "left contact force", "contact_R_normal_N": "right contact force",
    "finger_L_temp_C": "left driver temperature", "cube_slip_mm": "object slip",
    "finger_L_pos_mm": "left finger position", "finger_R_pos_mm": "right finger position",
    "cube_z": "object height", "gripper_z": "gripper height", "gripper_x": "gripper position",
    "obstacle_contact_N": "obstacle contact force",
}


def parse_log(text: str) -> list[dict]:
    out = []
    for i, line in enumerate(text.splitlines(), start=1):
        ts = int(line[1:line.index("ms]")].strip()) if line.startswith("[") else None
        rest = line[line.index("]") + 1:].split(None, 2) if "]" in line else ["", "", line]
        level, src, msg = (rest + ["", "", ""])[:3]
        out.append({"n": i, "t_ms": ts, "level": level, "src": src, "msg": msg.strip(), "text": line})
    return out


class Episode:
    """One recorded episode on disk (an original or a re-run)."""

    def __init__(self, ep_dir: Path):
        self.dir = ep_dir
        self.id = ep_dir.name

    @cached_property
    def meta(self) -> dict:
        return json.loads((self.dir / "meta.json").read_text())

    @cached_property
    def telemetry(self) -> pd.DataFrame:
        return pd.read_csv(self.dir / "telemetry.csv")

    @cached_property
    def signals(self) -> dict:
        return {k: v for k, v in SIGNALS.items() if k in self.telemetry.columns}

    @cached_property
    def log(self) -> list[dict]:
        return parse_log((self.dir / "controller.log").read_text())

    @property
    def masks(self) -> dict:
        """Prefer SAM 3.1 masks; fall back to simulator ground truth (labeled in the UI)."""
        sam = self.dir / "sam_masks.json"
        return json.loads((sam if sam.exists() else self.dir / "gt_masks.json").read_text())

    @cached_property
    def frames(self) -> list[np.ndarray]:
        return [f for f in imageio.get_reader(self.dir / "video.mp4")]

    def frame_index(self, t_ms: int) -> int:
        return int(np.clip(round(t_ms / 1000 * self.meta["fps"]), 0, len(self.frames) - 1))

    def frame_jpeg(self, idx: int, quality: int = 85, width: int | None = None) -> bytes:
        img = Image.fromarray(self.frames[idx])
        if width and width < img.width:
            img = img.resize((width, round(img.height * width / img.width)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()

    @property
    def patchable(self) -> dict:
        return self.meta.get("patchable", {})

    @property
    def video_url(self) -> str:
        return f"/episode/{self.id}/video.mp4"

    @property
    def is_mystery(self) -> bool:
        return "mystery" in self.meta

    def public_meta(self) -> dict:
        """Metadata safe to show the browser: a mystery's hidden setup is removed."""
        if not self.is_mystery:
            return self.meta
        return {k: v for k, v in self.meta.items() if k not in ("mystery", "scenario")} | {
            "title": self.meta["mystery"].get("title", self.id), "mystery": True}

    def answer_key(self) -> dict:
        """What was actually planted in a mystery episode. Only served after the board decides."""
        h = self.meta["mystery"]
        sc = h["scenario"]
        if sc == "thermal":
            planted = (f"The left finger's motor driver heats {h['heat_gain']:g} degrees per second per newton of grip "
                       f"and derates at {h['temp_hard']:g} °C (warning at {h['temp_soft']:g} °C).")
            expect = "hardware"
        elif sc == "slip":
            planted = (f"The part's real friction is {h['friction']:g}, not the catalog 0.7, and the grip force "
                       f"is only {h['grip_force_N']:g} N.")
            expect = "grasp"
        else:
            f = h["fixture"]
            planted = (f"An unmapped fixture {f['top'] * 100:.1f} cm tall stands at x = {f['x']:.2f} m, in the "
                       f"transport path, above the 12.5 cm carry height.")
            expect = "environment"
        decoys = [f"{d[0]:.2f} s: {d[2]} {d[3]}" for d in h.get("decoys", [])]
        return {"scenario": sc, "planted": planted, "expected_agent": expect, "decoys": decoys, "seed": h.get("seed")}

    def payload(self) -> dict:
        df = self.telemetry
        return {
            "meta": self.public_meta(),
            "telemetry": {c: df[c].tolist() for c in ["timestamp_ms", *self.signals]},
            "signals": self.signals,
            "log": self.log,
            "masks": self.masks,
            "video_url": self.video_url,
            "suggestions": SUGGESTIONS,
        }


def list_episodes() -> list[dict]:
    out = []
    dirs = [d for d in EPISODES_DIR.iterdir() if not d.name.startswith("fix_") and (d / "meta.json").exists()]
    for d in sorted(dirs, key=lambda d: (d.name.startswith("mystery_"), d.name[-3:] if not d.name.startswith("mystery_")
                                         else -d.stat().st_mtime)):
        m = json.loads((d / "meta.json").read_text())
        mystery = "mystery" in m
        out.append({"id": d.name, "title": m["mystery"].get("title", d.name) if mystery else m.get("title", d.name),
                    "mystery": mystery, "outcome": m["outcome"]})
    return out


@lru_cache(maxsize=32)
def get_episode(episode_id: str | None = None) -> Episode:
    episode_id = episode_id or DEFAULT_EPISODE
    d = EPISODES_DIR / episode_id
    if not episode_id.replace("_", "").isalnum() or not (d / "meta.json").exists():
        raise KeyError(f"unknown episode {episode_id}")
    return Episode(d)


# Backward-compatible helpers for the default episode (used by eval/ and sam/).
EP_DIR = EPISODES_DIR / DEFAULT_EPISODE


def meta() -> dict:
    return get_episode().meta


def frame_jpeg(idx: int, quality: int = 85, width: int | None = None) -> bytes:
    return get_episode().frame_jpeg(idx, quality, width)


# ------------------------------------------------------------------ answer validation

def clean_patch(ep: Episode, p: dict) -> dict:
    out = {}
    for k, spec in ep.patchable.items():
        try:
            out[k] = round(float(np.clip(float(p.get(k, spec["default"])), spec["min"], spec["max"])), 2)
        except (TypeError, ValueError):
            out[k] = spec["default"]
    out["why"] = str(p.get("why", "")).strip()
    out["changed"] = any(out[k] != spec["default"] for k, spec in ep.patchable.items())
    return out


def validate(ep: Episode, answer: dict, frame_width: int = FRAME_WIDTH) -> dict:
    m = ep.meta
    n_log = len(ep.log)
    segs = []
    for s in answer.get("segments", [])[:6]:
        t = int(np.clip(int(s.get("time_ms", 0)), 0, m["duration_ms"]))
        ll = int(s.get("log_line", -1))
        pt = dict(s.get("point") or {"x": -1, "y": -1, "label": ""})
        scale = m["width"] / frame_width
        if pt.get("x", -1) >= 0 and pt.get("y", -1) >= 0:
            pt["x"], pt["y"] = round(pt["x"] * scale), round(pt["y"] * scale)
        if not (0 <= pt.get("x", -1) < m["width"] and 0 <= pt.get("y", -1) < m["height"]):
            pt = {"x": -1, "y": -1, "label": ""}
        segs.append({
            "say": str(s.get("say", "")).strip(),
            "time_ms": t,
            "objects": [o for o in s.get("objects", []) if o in m["objects"]],
            "signal": s.get("signal") if s.get("signal") in ep.signals else "none",
            "log_line": ll if 1 <= ll <= n_log else -1,
            "point": pt,
        })
    tk = answer.get("takeaway") or {}
    takeaway = {
        "root_cause": str(tk.get("root_cause", "")).strip(),
        "fixes": [str(f).strip() for f in tk.get("fixes", []) if str(f).strip()][:3],
        "say": str(tk.get("say", "")).strip(),
        "sim_patch": clean_patch(ep, tk.get("sim_patch") or {}),
    }
    return {"segments": segs, "takeaway": takeaway}


def refine_point(ep: Episode, t_ms: int, x: int, y: int) -> dict:
    """SAM 3.1 segments whatever is under (x, y) at t_ms. Returns a mask record."""
    from .masks import mask_to_record
    m = ep.meta
    idx = ep.frame_index(t_ms)
    if os.environ.get("POINTER_MOCK") == "1":
        return {"frame": idx, "record": {}, "model": "mock"}
    mask = meta_api.sam_segment_point(ep.frame_jpeg(idx, 95), x, y, m["width"], m["height"])
    return {"frame": idx, "record": mask_to_record(mask) if mask is not None else {},
            "model": meta_api.SAM_MODEL}
