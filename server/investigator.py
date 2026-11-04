"""Pointer's agent loop.

1. Investigate. Muse Spark is given only a brief of the episode (task, outcome,
   which signals exist) and works in steps. Each step it writes a short thought
   and calls one tool:
       read_log        warnings/errors or all lines in a time window
       read_telemetry  chosen signals over a time window
       view_frames     up to 4 video frames at chosen times
   or finishes. The tools return real data from the episode; nothing is
   pre-digested for it.
2. Answer. With the evidence it gathered, it answers out loud and points
   (agent.ANSWER_SYSTEM), and proposes its first fix as simulator settings.
3. Verify. MuJoCo re-runs the episode with those settings. The verdict is
   computed from the run (outcome, derate, slip, collision), not by the model.
   If the fix fails or is only marginal, Muse Spark sees the measurements and
   revises; up to three attempts.

Every step is streamed to the browser as an event.
"""

import json
import os
import subprocess
import sys
import threading
import time
from typing import Iterator

import numpy as np

from . import meta_api
from .agent import (ANSWER_SYSTEM, FRAME_WIDTH, ROOT, SHORT, Episode, answer_schema, clean_patch,
                    patch_schema, validate)

MAX_TOOL_STEPS = int(os.environ.get("POINTER_MAX_STEPS", "5"))
MAX_FIX_ATTEMPTS = 3

STEP_SYSTEM = """You are Pointer, an agent that investigates why a robot episode failed. You cannot see \
the data until you ask for it. Work in steps. In each step, write a short thought and either call one \
tool or finish.

Tools:
- read_log: controller log lines. Set level to "warn_error" for only warnings and errors, or "all" for \
every line, and a time window with t_from_ms and t_to_ms.
- read_telemetry: sensor signals over a time window. Set signals (1 to 4 names from the list), \
t_from_ms, t_to_ms, and step_ms (20 to 200; smaller for short windows).
- view_frames: look at the camera. Set times_ms to 1 to 4 episode times.
- segment: run SAM 3.1 on 1 to 3 frames (times_ms) to find every instance of a short noun phrase \
(prompt, for example "part", "gripper finger", "gray box", "obstacle"). Returns each match's box, center \
and area, measured from the masks. Set relative_to to a second noun phrase to also get the gap and \
vertical offset between the two objects. Use it to measure what you see instead of estimating: whether \
the part sits between the fingers, how far it has dropped, or whether it overlaps something in its path.
- finish: you know the root cause and can show the evidence for it.

How to work:
- Be efficient: usually 3 or 4 tool calls. Start broad, then confirm the key moment in the telemetry, \
and look at the frames around it so you can point at what physically happened.
- Separate cause from effect. Many alarms are consequences; find the first thing that went wrong.
- thought: one or two sentences in the first person, for the engineer watching you work: what you just \
learned and what you will check next and why. No markdown.
- Fill fields the chosen tool does not use with: level "warn_error", t_from_ms 0, t_to_ms the episode \
duration, signals [], step_ms 100, times_ms [], prompt "", relative_to "".
"""

REVISE_SYSTEM = """You are Pointer, an agent that fixes robot failures. You proposed a fix as simulator \
settings, the simulator re-ran the episode, and the result was not clean. Look at the measurements and \
choose the next settings to try. Change what the measurements say is still wrong; keep what worked. \
thought: one or two first-person sentences for the engineer on what the last run showed and what you are \
changing. why: one sentence on what the new settings should do.
"""


def brief(ep: Episode) -> str:
    m = ep.meta
    sigs = "\n".join(f"  {k}: {v}" for k, v in ep.signals.items())
    settings = "\n".join(f"  {k} = {s['default']:g} (allowed {s['min']:g} to {s['max']:g}): {s['desc']}"
                         for k, s in ep.patchable.items())
    return (
        f"Episode {m['episode_id']}: {m['task']}\nOutcome: {m['outcome']}\n"
        f"Duration {m['duration_ms']} ms. Camera {m['width']}x{m['height']} at {m['fps']} fps; frames you view "
        f"are shown at {FRAME_WIDTH} pixels wide.\nTracked objects: \"1\" = {m['objects']['1']}, \"2\" = gripper.\n"
        f"Object catalog (object_db): mass 0.25 kg, nominal pad friction coefficient 0.7.\n"
        f"The controller log has {len(ep.log)} lines.\nTelemetry signals at 100 Hz:\n{sigs}\n"
        f"Controller settings:\n{settings}\n"
    )


def step_schema(ep: Episode) -> dict:
    return {
        "type": "object",
        "properties": {
            "thought": {"type": "string"},
            "tool": {"type": "string", "enum": ["read_log", "read_telemetry", "view_frames", "segment", "finish"]},
            "level": {"type": "string", "enum": ["warn_error", "all"]},
            "t_from_ms": {"type": "integer"},
            "t_to_ms": {"type": "integer"},
            "signals": {"type": "array", "items": {"type": "string", "enum": list(ep.signals)}},
            "step_ms": {"type": "integer"},
            "times_ms": {"type": "array", "items": {"type": "integer"}},
            "prompt": {"type": "string"},
            "relative_to": {"type": "string"},
        },
        "required": ["thought", "tool", "level", "t_from_ms", "t_to_ms", "signals", "step_ms", "times_ms",
                     "prompt", "relative_to"],
        "additionalProperties": False,
    }


# ------------------------------------------------------------------ tools

def run_tool(ep: Episode, s: dict) -> dict:
    """Execute one tool call. Returns text for the model, images, and what the UI should point at."""
    dur = ep.meta["duration_ms"]
    t0 = int(np.clip(int(s.get("t_from_ms", 0)), 0, dur))
    t1 = int(np.clip(int(s.get("t_to_ms", dur)), t0, dur))
    tool = s["tool"]
    if tool == "read_log":
        lv = s.get("level", "warn_error")
        lines = [l for l in ep.log if l["t_ms"] is not None and t0 <= l["t_ms"] <= t1
                 and (lv == "all" or l["level"] in ("WARN", "ERROR"))][:60]
        text = "\n".join(f"{l['n']:>3} | {l['text']}" for l in lines) or "(no matching lines)"
        what = "warnings and errors" if lv == "warn_error" else "all lines"
        return {"text": text, "images": [], "ui": {"lines": [l["n"] for l in lines], "t0": t0, "t1": t1},
                "summary": f"{len(lines)} {what}, {t0 / 1000:.1f} to {t1 / 1000:.1f} s",
                "label": f"Read the log, {what}"}
    if tool == "read_telemetry":
        sigs = [k for k in s.get("signals", []) if k in ep.signals][:4] or ["cube_z"]
        step = int(np.clip(int(s.get("step_ms", 100)), 20, 200))
        if (t1 - t0) / step > 80:
            step = int(np.ceil((t1 - t0) / 80 / 10) * 10)
        df = ep.telemetry
        sub = df[(df.timestamp_ms >= t0) & (df.timestamp_ms <= t1) & (df.timestamp_ms % step == 0)]
        text = sub[["timestamp_ms", *sigs]].to_csv(index=False, float_format="%.4g")
        return {"text": text, "images": [], "ui": {"signals": sigs, "t0": t0, "t1": t1},
                "summary": f"{len(sub)} rows, {t0 / 1000:.1f} to {t1 / 1000:.1f} s",
                "label": "Read telemetry: " + ", ".join(SHORT.get(k, k) for k in sigs)}
    if tool == "view_frames":
        times = sorted({int(np.clip(int(t), 0, dur)) for t in s.get("times_ms", [])})[:4] or [t0]
        imgs, labels = [], []
        for t in times:
            idx = ep.frame_index(t)
            labels.append(f"Frame {idx} at t={t} ms")
            imgs.append(ep.frame_jpeg(idx, quality=75, width=FRAME_WIDTH))
        return {"text": "\n".join(labels), "images": list(zip(labels, imgs)), "ui": {"times": times},
                "summary": "frames at " + ", ".join(f"{t / 1000:.2f}" for t in times) + " s",
                "label": f"Looked at {len(times)} frame{'s' if len(times) > 1 else ''}"}
    if tool == "segment":
        return segment_tool(ep, s)
    return {"text": "", "images": [], "ui": {}, "summary": "", "label": "Finished investigating"}


# ------------------------------------------------------------------ SAM 3.1 as an agent tool

_seg_cache: dict[tuple, tuple[list, str]] = {}
_seg_lock = threading.Lock()
GT_WORDS = {"1": ("part", "cube", "block", "object", "box"), "2": ("gripper", "finger", "jaw", "hand", "claw", "pad")}


def _segment_frame(ep: Episode, idx: int, prompt: str) -> tuple[list[np.ndarray], str]:
    """All masks matching `prompt` on frame `idx`, from SAM 3.1 (cached). Without SAM access, falls
    back to the simulator's segmentation for the tracked objects, and says so."""
    key = (ep.id, idx, prompt.lower().strip())
    with _seg_lock:
        if key in _seg_cache:
            return _seg_cache[key]
    m = ep.meta
    if meta_api.sam_available():
        masks = meta_api.sam_segment_text(ep.frame_jpeg(idx, 95), prompt, m["width"], m["height"])
        out = ([mk for mk in masks if mk.sum() >= 4], meta_api.SAM_MODEL)
    else:
        from .masks import record_to_mask
        gt = json.loads((ep.dir / "gt_masks.json").read_text())["frames"].get(str(idx), {})
        p = prompt.lower()
        hits = [oid for oid, words in GT_WORDS.items() if any(w in p for w in words) and gt.get(oid)]
        out = ([record_to_mask(gt[oid], m["width"], m["height"]) for oid in hits],
               "simulator segmentation (SAM 3.1 needs MODEL_API_KEY)")
    with _seg_lock:
        _seg_cache[key] = out
    return out


def _mask_gap(a: np.ndarray, b: np.ndarray) -> float:
    """Smallest pixel distance between two masks (0 if they touch or overlap)."""
    import cv2
    if np.logical_and(a, b).any():
        return 0.0
    dist = cv2.distanceTransform((1 - b).astype(np.uint8), cv2.DIST_L2, 3)
    return float(dist[a > 0].min())


def segment_tool(ep: Episode, s: dict) -> dict:
    from .masks import mask_to_record
    dur = ep.meta["duration_ms"]
    prompt = (s.get("prompt") or "part").strip()[:60]
    rel = (s.get("relative_to") or "").strip()[:60]
    times = sorted({int(np.clip(int(t), 0, dur)) for t in s.get("times_ms", [])})[:3] or [dur // 2]
    k = FRAME_WIDTH / ep.meta["width"]          # report in the pixels of the frames the agent sees
    lines, ui_masks, source = [], [], ""
    for t in times:
        idx = ep.frame_index(t)
        masks, source = _segment_frame(ep, idx, prompt)
        rmasks = _segment_frame(ep, idx, rel)[0] if rel else []
        lines.append(f"Frame {idx} at t={t} ms, \"{prompt}\": {len(masks)} match{'es' if len(masks) != 1 else ''}")
        recs = []
        for i, mk in enumerate(masks, 1):
            r = mask_to_record(mk)
            if not r:
                continue
            recs.append(r)
            x1, y1, x2, y2 = (round(v * k) for v in r["bbox"])
            cx, cy = (round(v * k) for v in r["centroid"])
            line = f"  #{i}: box ({x1},{y1})-({x2},{y2}), center ({cx},{cy}), area {round(r['area'] * k * k)} px"
            if rmasks:
                j = min(range(len(rmasks)), key=lambda j: _mask_gap(mk, rmasks[j]))
                rr = mask_to_record(rmasks[j])
                gap = _mask_gap(mk, rmasks[j]) * k
                dy = (r["centroid"][1] - rr["centroid"][1]) * k
                overlap = int(np.logical_and(mk, rmasks[j]).sum() * k * k)
                line += (f"; vs \"{rel}\": gap {gap:.0f} px, center {abs(dy):.0f} px "
                         f"{'below' if dy > 0 else 'above'}" + (f", overlapping {overlap} px" if overlap else ""))
            lines.append(line)
        if rel and not rmasks:
            lines.append(f"  (no \"{rel}\" found on this frame)")
        ui_masks.append({"t": t, "frame": idx, "prompt": prompt, "records": recs,
                         "source": "sam" if source.startswith("sam") else "sim"})
    lines.append(f"Coordinates are in the {FRAME_WIDTH}-pixel-wide frames you view. Source: {source}.")
    n = sum(len(m["records"]) for m in ui_masks)
    return {"text": "\n".join(lines), "images": [], "ui": {"masks": ui_masks, "times": times},
            "summary": f"{n} mask{'s' if n != 1 else ''} on {len(times)} frame{'s' if len(times) > 1 else ''}"
                       + (f", gap to \"{rel}\"" if rel else ""),
            "label": f"SAM 3.1: segmented \"{prompt}\"" if source.startswith("sam")
                     else f"Segmented \"{prompt}\" (simulator)"}


def evidence_parts(ep: Episode, steps: list[dict]) -> list:
    parts = []
    for i, st in enumerate(steps, 1):
        call = {k: st["call"][k] for k in ("tool", "level", "t_from_ms", "t_to_ms", "signals", "step_ms", "times_ms",
                                           "prompt", "relative_to") if k in st["call"]}
        parts.append({"type": "input_text", "text": (
            f"Step {i}. Thought: {st['call']['thought']}\nCall: {json.dumps(call)}\nResult:\n{st['result']['text']}")})
        for label, jpeg in st["result"]["images"]:
            parts.append({"type": "input_text", "text": label + ":"})
            parts.append(meta_api.image_part(jpeg))
    return parts


# ------------------------------------------------------------------ simulation + verdict

def run_fix(ep: Episode, params: dict) -> Episode:
    keys = list(ep.patchable)
    tag = "_".join(f"{k.split('_')[0]}{params[k]:g}".replace(".", "p") for k in keys)
    run_dir = ep.dir.parent / f"fix_{ep.id}_{tag}"
    if not (run_dir / "meta.json").exists():
        env = dict(os.environ)
        if sys.platform.startswith("linux"):
            env.setdefault("MUJOCO_GL", "egl")
        proc = subprocess.run(
            [sys.executable, str(ROOT / "sim" / "generate_episode.py"), str(ep.dir.parent),
             "--scenario", ep.meta.get("scenario", "thermal"), "--id", run_dir.name,
             "--params", json.dumps({k: params[k] for k in keys})]
            + (["--hidden", json.dumps(ep.meta["mystery"])] if "mystery" in ep.meta else []),
            capture_output=True, text=True, env=env, timeout=600)
        if proc.returncode != 0 or not (run_dir / "meta.json").exists():
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-4:]
            raise RuntimeError("MuJoCo re-run failed: " + " | ".join(tail))
    return Episode(run_dir)


def verdict(m: dict) -> dict:
    obj = m["objects"]["1"]
    issues, say_issues = [], []
    if m.get("derate_ms") is not None:
        issues.append(f"the left driver still derated at {m['derate_ms'] / 1000:.2f} s")
        say_issues.append("the driver still overheats")
    if m.get("slip_ms") is not None:
        issues.append(f"the {obj} slipped in the grip at {m['slip_ms'] / 1000:.2f} s")
        say_issues.append(f"the {obj} still slips")
    if (m.get("max_obstacle_contact_N") or 0) > 1:
        issues.append(f"the carried load hit an obstacle with {m['max_obstacle_contact_N']:g} N")
        say_issues.append("it still clips the obstacle")
    ok = m["outcome"].startswith("OK")
    temp = f"Peak driver temperature {m['max_driver_temp_C']} °C."
    if ok and not issues:
        return {"kind": "ok", "label": "Verified",
                "text": f"Verified: the {obj} lands in the bin with no derate, slip or collision. {temp}",
                "say": f"The {obj} lands in the bin cleanly."}
    if ok:
        return {"kind": "warn", "label": "Marginal",
                "text": f"Marginal: the {obj} ends up in the bin, but " + "; ".join(issues) + ".",
                "say": f"The {obj} ends up in the bin, but " + " and ".join(say_issues) + "."}
    reason = m["outcome"].replace("FAIL: ", "")
    return {"kind": "err", "label": "Failed",
            "text": f"Still failing: {reason}" + (f"; {'; '.join(issues)}" if issues else "") + ".",
            "say": f"That did not work. The {reason.replace('object', obj)}."}


def measurements(m: dict) -> str:
    return (f"outcome={m['outcome']}; peak driver temperature {m['max_driver_temp_C']} C (derates at "
            f"{m.get('driver_limit_C', 80)}); derate at {m.get('derate_ms')} ms; slip detected at "
            f"{m.get('slip_ms')} ms; max obstacle contact {m.get('max_obstacle_contact_N')} N; "
            f"first obstacle contact at {m.get('first_contact_ms')} ms")


# ------------------------------------------------------------------ the loop

def ev(type_: str, **kw) -> dict:
    return {"type": type_, **kw}


def investigate(ep: Episode, question: str, current_ms: int, history: list[dict]) -> Iterator[dict]:
    if os.environ.get("POINTER_MOCK") == "1":
        yield from mock_investigate(ep, question)
        return
    t_start = time.time()
    model = meta_api.REASONING_MODEL
    yield ev("start", model=model, brief={"signals": len(ep.signals), "log_lines": len(ep.log)})
    base = [{"type": "input_text", "text": brief(ep)}]
    if history:
        convo = "\n".join(f"Engineer: {h['q']}\nPointer: {h['a']}" for h in history[-3:])
        base.append({"type": "input_text", "text": f"Earlier in this session:\n{convo}"})
    q = {"type": "input_text", "text": (
        f"The engineer's video is paused at t={current_ms} ms.\nEngineer asks: {question}")}

    steps: list[dict] = []
    for n in range(1, MAX_TOOL_STEPS + 2):
        last = n > MAX_TOOL_STEPS
        nudge = {"type": "input_text", "text": (
            "You have used all your tool calls; finish now." if last else "Decide your next step.")}
        t0 = time.time()
        call = meta_api.spark_json(STEP_SYSTEM, base + evidence_parts(ep, steps) + [q, nudge],
                                   step_schema(ep), "next_step")
        call["tool"] = "finish" if last else call.get("tool", "finish")
        result = run_tool(ep, call)
        steps.append({"call": call, "result": result})
        yield ev("step", n=n, tool=call["tool"], thought=str(call.get("thought", "")).strip(),
                 label=result["label"], summary=result["summary"], ui=result["ui"],
                 model_ms=int((time.time() - t0) * 1000))
        if call["tool"] == "finish":
            break

    t0 = time.time()
    raw = meta_api.spark_json(ANSWER_SYSTEM, base + evidence_parts(ep, steps) + [q],
                              answer_schema(list(ep.patchable)), "pointer_answer")
    ans = validate(ep, raw)
    ans |= {"model": model, "model_ms": int((time.time() - t0) * 1000),
            "total_ms": int((time.time() - t_start) * 1000), "steps": len(steps)}
    yield ev("answer", answer=ans)

    yield from verify_loop(ep, ans, base)
    yield ev("done", total_ms=int((time.time() - t_start) * 1000))


def verify_loop(ep: Episode, ans: dict, base: list, revise=None) -> Iterator[dict]:
    patch = ans["takeaway"]["sim_patch"]
    if not patch.get("changed"):
        yield ev("fix_done", status="skipped", reason="No simulator change was proposed.")
        return
    keys = list(ep.patchable)
    attempts = []
    thought = patch.get("why", "")
    for n in range(1, MAX_FIX_ATTEMPTS + 1):
        params = {k: patch[k] for k in keys}
        yield ev("fix_attempt", n=n, params=params, defaults={k: ep.patchable[k]["default"] for k in keys},
                 thought=thought)
        t0 = time.time()
        try:
            run = run_fix(ep, params)
        except Exception as e:
            yield ev("fix_done", status="error", reason=str(e))
            return
        v = verdict(run.meta)
        payload = run.payload() | {"verdict": v}
        yield ev("fix_result", n=n, verdict=v, run=payload, sim_ms=int((time.time() - t0) * 1000))
        attempts.append({"params": params, "meas": measurements(run.meta), "verdict": v["label"]})
        if v["kind"] == "ok":
            yield ev("fix_done", status="verified", n=n, run_id=run.id)
            return
        if n == MAX_FIX_ATTEMPTS:
            break
        if revise is not None:
            nxt = revise(n)
        else:
            hist = "\n".join(f"Attempt {i}: settings {json.dumps(a['params'])} -> {a['verdict']}. {a['meas']}"
                             for i, a in enumerate(attempts, 1))
            nxt = meta_api.spark_json(
                REVISE_SYSTEM,
                base + [{"type": "input_text", "text": (
                    f"Root cause you found: {ans['takeaway']['root_cause']}\n"
                    f"Original run measurements: {measurements(ep.meta)}\n{hist}\nChoose the next settings.")}],
                {"type": "object", "properties": {"thought": {"type": "string"}, "sim_patch": patch_schema(keys)},
                 "required": ["thought", "sim_patch"], "additionalProperties": False}, "revision")
        patch = clean_patch(ep, nxt.get("sim_patch") or {})
        thought = str(nxt.get("thought", "")).strip()
        if not patch["changed"] or any(a["params"] == {k: patch[k] for k in keys} for a in attempts):
            yield ev("fix_done", status="not_verified", n=n, reason="No new settings to try.")
            return
    yield ev("fix_done", status="not_verified", n=len(attempts),
             reason=f"No clean fix within {MAX_FIX_ATTEMPTS} attempts.")


# ------------------------------------------------------------------ cache and replay

_cache: dict[str, list] = {}
_lock = threading.Lock()


def cache_key(ep: Episode, q: str) -> str:
    return ep.id + "|" + " ".join(q.lower().replace("?", "").split())


def run_or_replay(ep: Episode, question: str, current_ms: int, history: list[dict]) -> Iterator[dict]:
    """Stream an investigation. A first question asked before is replayed from the earlier run,
    and the events say so."""
    key = cache_key(ep, question)
    if not history:
        with _lock:
            hit = _cache.get(key)
        if hit:
            for e in hit:
                if e["type"] == "start":
                    e = e | {"replayed": True}
                if e["type"] == "answer":
                    e = e | {"answer": e["answer"] | {"cached": True}}
                if e["type"] in ("agent_step", "hypothesis", "challenge", "experiment_start", "revise"):
                    time.sleep(0.35)
                yield e
            return
    from . import board
    script = None
    if os.environ.get("POINTER_MOCK") == "1":
        from .ui_test_fixtures import mock_script
        script = mock_script(ep)
    events = []
    for e in board.run(ep, question, current_ms, history, script=script):
        events.append(e)
        yield e
    if not history and events and events[-1]["type"] == "done":
        with _lock:
            _cache[key] = events


def warm(episode_ids: list[str], get_episode):
    """Run the first suggested question on each episode in the background (POINTER_WARM=1)."""
    from .agent import SUGGESTIONS

    def go():
        for eid in episode_ids:
            try:
                ep = get_episode(eid)
                for _ in run_or_replay(ep, SUGGESTIONS[0], 0, []):
                    pass
                print(f"[warm] ready: {eid}", flush=True)
            except Exception as e:
                print(f"[warm] {eid} failed: {e}", flush=True)

    threading.Thread(target=go, daemon=True).start()


# ------------------------------------------------------------------ mock (UI testing only)

MOCK = {
    "thermal": {"signals": ["finger_L_temp_C", "finger_L_cmd_force_N"], "first": {"hold_force_N": 8},
                "second": {"hold_force_N": 6},
                "cause": "MOCK: the left finger driver overheated under a 12 N grip and derated mid-transport."},
    "slip": {"signals": ["contact_L_normal_N", "cube_slip_mm", "cube_z"], "first": {"hold_force_N": 5.2},
             "second": {"hold_force_N": 9},
             "cause": "MOCK: the 5.1 N grip assumes catalog friction, but the steel block is far more slippery."},
    "collision": {"signals": ["obstacle_contact_N", "cube_z", "gripper_x"], "first": {"transport_height_cm": 16},
                  "second": {"transport_height_cm": 18},
                  "cause": "MOCK: an unmapped fixture sits in the transport path below the carry height."},
}


def mock_investigate(ep: Episode, question: str) -> Iterator[dict]:
    """Scripted agent for UI testing (POINTER_MOCK=1). Tools and MuJoCo runs are real; the model
    is not called. Every thought is prefixed MOCK and the UI shows a 'Mock answers' badge."""
    t_start = time.time()
    sc = MOCK.get(ep.meta.get("scenario", "thermal"), MOCK["thermal"])
    errs = [l for l in ep.log if l["level"] == "ERROR"]
    t_err = errs[0]["t_ms"] if errs else ep.meta["duration_ms"] // 2
    dur = ep.meta["duration_ms"]
    script = [
        {"thought": "MOCK: Starting broad with every warning and error in the log.", "tool": "read_log",
         "level": "warn_error", "t_from_ms": 0, "t_to_ms": dur},
        {"thought": f"MOCK: The first error is at {t_err / 1000:.2f} s. Checking the sensors before it.",
         "tool": "read_telemetry", "signals": sc["signals"], "t_from_ms": max(0, t_err - 1500),
         "t_to_ms": min(dur, t_err + 500), "step_ms": 50},
        {"thought": "MOCK: Looking at the camera around that moment.", "tool": "view_frames",
         "times_ms": [max(0, t_err - 300), t_err, min(dur, t_err + 300)]},
        {"thought": "MOCK: That is enough evidence for the root cause.", "tool": "finish"},
    ]
    yield ev("start", model="MOCK", brief={"signals": len(ep.signals), "log_lines": len(ep.log)})
    for n, call in enumerate(script, 1):
        time.sleep(0.5)
        r = run_tool(ep, call)
        yield ev("step", n=n, tool=call["tool"], thought=call["thought"], label=r["label"],
                 summary=r["summary"], ui=r["ui"], model_ms=0)
    keys = list(ep.patchable)
    segs = [{"say": f"MOCK: {l['src']} reported {l['msg'][:60]}.", "time_ms": l["t_ms"], "objects": ["1", "2"],
             "signal": sc["signals"][min(i, len(sc["signals"]) - 1)], "log_line": l["n"],
             "point": {"x": -1, "y": -1, "label": ""}} for i, l in enumerate(errs[:3])]
    first = {k: ep.patchable[k]["default"] for k in keys} | sc["first"]
    ans = validate(ep, {"segments": segs, "takeaway": {
        "root_cause": sc["cause"], "fixes": ["MOCK: first fix.", "MOCK: second fix."],
        "say": "MOCK takeaway.", "sim_patch": first | {"why": "MOCK: first attempt."}}})
    ans |= {"model": "MOCK", "model_ms": 0, "total_ms": int((time.time() - t_start) * 1000), "steps": len(script)}
    yield ev("answer", answer=ans)
    second = {k: ep.patchable[k]["default"] for k in keys} | sc["second"]
    yield from verify_loop(ep, ans, [], revise=lambda n: {
        "thought": "MOCK: the first attempt was not clean, so I am pushing the same setting further.",
        "sim_patch": second | {"why": "MOCK: second attempt."}})
    yield ev("done", total_ms=int((time.time() - t_start) * 1000))
