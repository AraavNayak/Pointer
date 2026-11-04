"""Pointer's reasoning layer.

Muse Spark reads sampled video frames, the telemetry table and the numbered
controller log, and answers with a short chain of spoken segments. Every segment
carries *references* instead of free-floating prose: a timestamp, which tracked
objects to outline, which telemetry signal to highlight and which log line to
point at. The front end turns those references into exact on-screen annotations
using SAM 3.1 masks, so the model never has to guess pixel coordinates for
things we can ground precisely.

For anything that is not a tracked object (e.g. "the left finger pad"), Muse
Spark may return a raw pixel point; the server then asks SAM 3.1 to segment the
object under that point on that frame (`/api/refine`).
"""

import io
import json
import os
from functools import lru_cache
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd
from PIL import Image

from . import meta_api

ROOT = Path(__file__).resolve().parent.parent
EPISODE_ID = os.environ.get("POINTER_EPISODE", "pick_place_drop_001")
EP_DIR = ROOT / "episodes" / EPISODE_ID

SIGNALS = {
    "finger_L_cmd_force_N": "left finger actuator force (N)",
    "finger_R_cmd_force_N": "right finger actuator force (N)",
    "contact_L_normal_N": "left pad contact normal force on cube (N)",
    "contact_R_normal_N": "right pad contact normal force on cube (N)",
    "finger_L_temp_C": "left finger driver temperature (C)",
    "cube_slip_mm": "cube displacement relative to grasp (mm)",
    "finger_L_pos_mm": "left finger closure (mm, 0 = open)",
    "finger_R_pos_mm": "right finger closure (mm, 0 = open)",
    "cube_z": "cube height (m)",
    "gripper_z": "gripper height (m)",
}

SYSTEM = """You are Pointer, a debugging co-pilot for robot failures. You are looking at one recorded \
episode: a camera video, a 100 Hz telemetry log and the controller's stdout log.

You answer out loud while pointing at evidence on the engineer's screen. Return 2 to 5 segments. \
Each segment is one or two short spoken sentences (under 30 words total, plain words, no markdown, \
say numbers the way a person would) plus the evidence to point at while it is spoken:
- time_ms: the episode time to show (the video seeks there and the charts draw a cursor there).
- objects: ids of tracked objects to outline in the video at that time ("1" = red cube, "2" = gripper). Empty list if none.
- signal: one telemetry column to highlight, or "none".
- log_line: the line number of the log line to highlight, or -1.
- point: a pixel location in the video frame at time_ms for something that is NOT a tracked object \
(for example a single finger pad), as {"x":..,"y":..,"label":".."}; use x = -1 when not needed. \
Frames are 640x480 pixels.

Rules:
- Only cite evidence you can actually see in the data provided. Timestamps must be real times from the \
telemetry or log. Log line numbers must match the numbers shown in the log.
- Order segments as a causal chain: what went wrong first, then what it caused, then the visible outcome, \
then (if useful) the fix.
- Prefer pointing at all three views when the evidence exists in all three.
- If the question is about something the data does not show, say so plainly in one segment.
"""

SCHEMA = {
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
        "root_cause": {"type": "string"},
    },
    "required": ["segments", "root_cause"],
    "additionalProperties": False,
}


# ------------------------------------------------------------------ episode data

@lru_cache
def meta() -> dict:
    return json.loads((EP_DIR / "meta.json").read_text())


@lru_cache
def telemetry() -> pd.DataFrame:
    return pd.read_csv(EP_DIR / "telemetry.csv")


@lru_cache
def log_lines() -> list[dict]:
    out = []
    for i, line in enumerate((EP_DIR / "controller.log").read_text().splitlines(), start=1):
        ts = int(line[1:line.index("ms]")].strip()) if line.startswith("[") else None
        level = line[line.index("]") + 2:].split()[0] if "]" in line else ""
        out.append({"n": i, "t_ms": ts, "level": level, "text": line})
    return out


@lru_cache
def masks() -> dict:
    """Prefer SAM 3.1 masks; fall back to simulator ground truth (clearly labeled)."""
    sam = EP_DIR / "sam_masks.json"
    if sam.exists():
        return json.loads(sam.read_text())
    return json.loads((EP_DIR / "gt_masks.json").read_text())


@lru_cache
def frames() -> list[np.ndarray]:
    return [f for f in imageio.get_reader(EP_DIR / "video.mp4")]


def frame_index(t_ms: int) -> int:
    m = meta()
    return int(np.clip(round(t_ms / 1000 * m["fps"]), 0, len(frames()) - 1))


def frame_jpeg(idx: int, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(frames()[idx]).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def episode_payload() -> dict:
    df = telemetry()
    return {
        "meta": meta(),
        "telemetry": {c: df[c].tolist() for c in ["timestamp_ms", *SIGNALS]},
        "signals": SIGNALS,
        "log": log_lines(),
        "masks": masks(),
    }


# ------------------------------------------------------------------ context for Muse Spark

def telemetry_table(step_ms: int = 100) -> str:
    df = telemetry()
    sub = df[df.timestamp_ms % step_ms == 0]
    return sub[["timestamp_ms", *SIGNALS]].to_csv(index=False, float_format="%.3g")


def log_text() -> str:
    return "\n".join(f"{l['n']:>3} | {l['text']}" for l in log_lines())


def build_content(question: str, current_ms: int, history: list[dict]) -> list:
    m = meta()
    parts: list[dict] = [{"type": "input_text", "text": (
        f"Episode: {m['episode_id']}\nTask: {m['task']}\nOutcome: {m['outcome']}\n"
        f"Duration: {m['duration_ms']} ms, video {m['width']}x{m['height']} at {m['fps']} fps.\n"
        f"Tracked objects: {json.dumps(m['objects'])}\n"
        f"The engineer's video is currently paused at t={current_ms} ms.\n\n"
        f"TELEMETRY (every 100 ms; columns: {', '.join(f'{k} = {v}' for k, v in SIGNALS.items())}):\n"
        f"{telemetry_table()}\n"
        f"CONTROLLER LOG (line number | text):\n{log_text()}\n\n"
        "VIDEO FRAMES follow, each labeled with its time."
    )}]
    times = list(range(0, m["duration_ms"], 500))
    if current_ms not in times:
        times = sorted(set(times + [current_ms]))
    for t in times:
        idx = frame_index(t)
        parts.append({"type": "input_text", "text": f"Frame {idx} at t={t} ms:"})
        parts.append(meta_api.image_part(frame_jpeg(idx, quality=80)))
    if history:
        convo = "\n".join(f"Engineer: {h['q']}\nPointer: {h['a']}" for h in history[-4:])
        parts.append({"type": "input_text", "text": f"Earlier in this session:\n{convo}"})
    parts.append({"type": "input_text", "text": f"Engineer asks: {question}"})
    return parts


def validate(answer: dict) -> dict:
    m = meta()
    n_log = len(log_lines())
    segs = []
    for s in answer.get("segments", [])[:6]:
        t = int(np.clip(int(s.get("time_ms", 0)), 0, m["duration_ms"]))
        ll = int(s.get("log_line", -1))
        pt = s.get("point") or {"x": -1, "y": -1, "label": ""}
        if not (0 <= pt.get("x", -1) < m["width"] and 0 <= pt.get("y", -1) < m["height"]):
            pt = {"x": -1, "y": -1, "label": ""}
        segs.append({
            "say": str(s.get("say", "")).strip(),
            "time_ms": t,
            "objects": [o for o in s.get("objects", []) if o in m["objects"]],
            "signal": s.get("signal") if s.get("signal") in SIGNALS else "none",
            "log_line": ll if 1 <= ll <= n_log else -1,
            "point": pt,
        })
    return {"segments": segs, "root_cause": str(answer.get("root_cause", ""))}


def ask(question: str, current_ms: int, history: list[dict]) -> dict:
    if os.environ.get("POINTER_MOCK") == "1":
        return mock_answer(question)
    content = build_content(question, current_ms, history)
    raw = meta_api.spark_json(SYSTEM, content, SCHEMA, "pointer_answer")
    out = validate(raw)
    out["model"] = meta_api.SPARK_MODEL
    return out


def refine_point(t_ms: int, x: int, y: int) -> dict:
    """SAM 3.1 segments whatever is under (x, y) at t_ms. Returns a mask record."""
    from .masks import mask_to_record
    m = meta()
    idx = frame_index(t_ms)
    if os.environ.get("POINTER_MOCK") == "1":
        return {"frame": idx, "record": {}, "model": "mock"}
    mask = meta_api.sam_segment_point(frame_jpeg(idx, 95), x, y, m["width"], m["height"])
    return {"frame": idx, "record": mask_to_record(mask) if mask is not None else {},
            "model": meta_api.SAM_MODEL}


def mock_answer(question: str) -> dict:
    """UI test fixture only (POINTER_MOCK=1). Not model output; the UI labels it MOCK."""
    by_text = {l["text"].split("] ", 1)[1][:30]: l["n"] for l in log_lines()}
    find = lambda s: next((n for k, n in by_text.items() if s in k), -1)
    t_derate = next(l["t_ms"] for l in log_lines() if "thermal limit" in l["text"])
    t_drop = next(l["t_ms"] for l in log_lines() if "object dropped" in l["text"])
    return validate({
        "segments": [
            {"say": "Mock answer for UI testing. The left finger driver overheated during transport.",
             "time_ms": 3140, "objects": ["2"], "signal": "finger_L_temp_C",
             "log_line": find("WARN  finger_L"), "point": {"x": -1, "y": -1, "label": ""}},
            {"say": "At eighty degrees it derated its torque limit from twelve newtons to point six.",
             "time_ms": t_derate, "objects": ["2"], "signal": "finger_L_cmd_force_N",
             "log_line": find("ERROR finger_L"), "point": {"x": -1, "y": -1, "label": ""}},
            {"say": "Contact force on the left pad collapsed, so the cube slipped out.",
             "time_ms": t_derate + 40, "objects": ["1", "2"], "signal": "contact_L_normal_N",
             "log_line": find("ERROR grasp_mon"), "point": {"x": -1, "y": -1, "label": ""}},
            {"say": "It hit the table short of the bin.", "time_ms": t_drop, "objects": ["1"],
             "signal": "cube_z", "log_line": find("ERROR supervisor"),
             "point": {"x": -1, "y": -1, "label": ""}},
        ],
        "root_cause": "MOCK: left finger thermal derate during transport",
    }) | {"model": "MOCK"}
