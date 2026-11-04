"""Thin client for the Meta Model API (https://api.meta.ai/v1).

Models used:
  - Muse Spark          reasoning over frames + telemetry + logs, structured JSON output
  - SAM 3.1             video object tracking (precompute) and single-frame refinement (live)
  - Muse Voice Transcribe  push-to-talk speech to text

Docs: https://dev.meta.ai/docs/overview
"""

import base64
import json
import os
import time
import uuid

import numpy as np
import requests
from meta_sam_parser import (
    CompletedOutcome,
    decode_mask_to_raster,
    image_segmentation_format,
    video_segmentation_format,
)
from openai import OpenAI

BASE_URL = os.environ.get("META_API_BASE", "https://api.meta.ai/v1")
SPARK_MODEL = os.environ.get("MUSE_SPARK_MODEL", "muse-spark-1.3")
SAM_MODEL = os.environ.get("SAM_MODEL", "sam-3.1")
ASR_MODEL = os.environ.get("ASR_MODEL", "muse-voice-transcribe-1.0")

# Optional stand-in for Muse Spark while testing (any OpenAI-compatible Chat Completions
# endpoint that accepts images). Never use this for the demo or the submission.
TEST_MODEL = os.environ.get("TEST_MODEL", "").strip()
TEST_API_BASE = os.environ.get("TEST_API_BASE", "https://generativelanguage.googleapis.com/v1beta/openai/")
TEST_API_KEY = os.environ.get("TEST_API_KEY", "").strip()
USING_TEST_MODEL = bool(TEST_MODEL and TEST_API_KEY)
REASONING_MODEL = f"{TEST_MODEL} (test stand-in)" if USING_TEST_MODEL else SPARK_MODEL


def api_key() -> str:
    key = os.environ.get("MODEL_API_KEY")
    if not key:
        raise RuntimeError("MODEL_API_KEY is not set. Put it in .env (see .env.example).")
    return key


def client() -> OpenAI:
    return OpenAI(api_key=api_key(), base_url=BASE_URL, timeout=180)


# ---------------------------------------------------------------- Muse Spark

SPARK_EFFORT = os.environ.get("MUSE_SPARK_EFFORT", "low")  # "none" omits the setting

# Running totals, read by eval scripts to report calls and tokens per session.
import threading as _threading
USAGE = {"spark_calls": 0, "input_tokens": 0, "output_tokens": 0, "sam_calls": 0}
_usage_lock = _threading.Lock()


def _add_usage(**kw):
    with _usage_lock:
        for k, v in kw.items():
            USAGE[k] = USAGE.get(k, 0) + (v or 0)


def spark_json(system: str, content: list, schema: dict, schema_name: str = "answer",
               reasoning_effort: str | None = None) -> dict:
    """Call Muse Spark through the Responses API and return parsed JSON.

    `content` is a list of Responses input parts: {"type": "input_text", ...} and
    {"type": "input_image", "image_url": "data:image/jpeg;base64,..."}.
    """
    if USING_TEST_MODEL:
        return _test_json(system, content, schema, schema_name)
    effort = reasoning_effort or SPARK_EFFORT
    kwargs = dict(
        model=SPARK_MODEL,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system}]},
            {"role": "user", "content": content},
        ],
        text={"format": {"type": "json_schema", "name": schema_name, "schema": schema}},
    )
    if effort and effort != "none":
        kwargs["reasoning"] = {"effort": effort}
    n_img = sum(p.get("type") == "input_image" for p in content)
    t0 = time.time()
    resp = None
    for attempt in range(4):
        try:
            resp = client().responses.create(**kwargs)
            break
        except Exception as e:
            msg = str(e).lower()
            if "reasoning" in kwargs and "reasoning" in msg and "rate" not in msg:
                kwargs.pop("reasoning")  # model rejected the effort value; retry with its default
                continue
            if attempt < 3 and ("429" in msg or "rate limit" in msg or "too many requests" in msg):
                time.sleep(2 * (attempt + 1))   # parallel agents can hit rate limits; back off
                continue
            raise
    dt = time.time() - t0
    u = getattr(resp, "usage", None)
    toks = ""
    if u is not None:
        rt = getattr(getattr(u, "output_tokens_details", None), "reasoning_tokens", None)
        toks = f" in={getattr(u, 'input_tokens', '?')} out={getattr(u, 'output_tokens', '?')}" + (
            f" reasoning={rt}" if rt is not None else "")
    _add_usage(spark_calls=1, input_tokens=getattr(u, "input_tokens", 0) if u is not None else 0,
               output_tokens=getattr(u, "output_tokens", 0) if u is not None else 0)
    print(f"[muse-spark] {SPARK_MODEL} effort={kwargs.get('reasoning', {}).get('effort', 'default')} "
          f"images={n_img} {dt:.1f}s{toks}", flush=True)
    text = resp.output_text or ""
    return _loads_lenient(text)


def _test_json(system: str, content: list, schema: dict, schema_name: str) -> dict:
    """Same request through Chat Completions on the test provider."""
    parts = []
    for p in content:
        if p.get("type") == "input_text":
            parts.append({"type": "text", "text": p["text"]})
        elif p.get("type") == "input_image":
            parts.append({"type": "image_url", "image_url": {"url": p["image_url"]}})
    c = OpenAI(api_key=TEST_API_KEY, base_url=TEST_API_BASE, timeout=180)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": parts}]
    t0 = time.time()
    try:
        for attempt in range(4):
            try:
                resp = c.chat.completions.create(
                    model=TEST_MODEL, messages=messages,
                    response_format={"type": "json_schema", "json_schema": {"name": schema_name, "schema": schema}})
                break
            except Exception as e:
                if attempt < 3 and ("429" in str(e) or "rate" in str(e).lower()):
                    time.sleep(3 * (attempt + 1))
                    continue
                raise
    except Exception as e:
        # Some providers reject parts of JSON Schema; fall back to plain JSON mode with the schema in the prompt.
        print(f"[test-model] json_schema rejected ({str(e)[:120]}); retrying in JSON mode", flush=True)
        messages[0]["content"] = system + "\n\nReply with only a JSON object matching this JSON Schema:\n" + json.dumps(schema)
        resp = c.chat.completions.create(model=TEST_MODEL, messages=messages,
                                         response_format={"type": "json_object"})
    n_img = sum(p["type"] == "image_url" for p in parts)
    u = getattr(resp, "usage", None)
    _add_usage(spark_calls=1, input_tokens=getattr(u, "prompt_tokens", 0) if u is not None else 0,
               output_tokens=getattr(u, "completion_tokens", 0) if u is not None else 0)
    print(f"[test-model] {TEST_MODEL} images={n_img} {time.time() - t0:.1f}s", flush=True)
    return _loads_lenient(resp.choices[0].message.content or "")


def _loads_lenient(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def image_part(jpeg_bytes: bytes) -> dict:
    return {"type": "input_image",
            "image_url": "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode()}


# ---------------------------------------------------------------- SAM 3.1

def _sam_stream_text(content: list, metadata: dict | None = None) -> str:
    """Run a streaming SAM 3.1 request and return the full special-token output text."""
    _add_usage(sam_calls=1)
    stream = client().responses.create(
        model=SAM_MODEL,
        input=[{"role": "user", "content": content}],
        stream=True,
        **({"metadata": metadata} if metadata else {}),
    )
    chunks, final = [], None
    for ev in stream:
        t = getattr(ev, "type", "")
        if t == "response.output_text.delta":
            chunks.append(ev.delta)
        elif t == "response.content_part.done":
            part = getattr(ev, "part", None)
            if part is not None and getattr(part, "type", "") == "output_text":
                final = getattr(part, "text", None)
        elif t in ("response.failed", "error"):
            raise RuntimeError(f"SAM request failed: {ev}")
    return final if final is not None else "".join(chunks)


def parse_sam_output(text: str, width: int, height: int, video: bool) -> dict:
    """SAM special-token text -> {frame_idx: {object_id: full-frame binary mask}}."""
    fmt = video_segmentation_format() if video else image_segmentation_format()
    parser = fmt.create_parser()
    parser.push(text if text.endswith("\n") else text + "\n")
    result = parser.finish(CompletedOutcome()).result
    out: dict[int, dict[str, np.ndarray]] = {}
    for rec in result.records:
        if rec.kind != "mask":
            continue
        fi = rec.frame.frame_index if rec.frame is not None else 0
        raster = np.frombuffer(decode_mask_to_raster(rec.mask), np.uint8).reshape(
            rec.mask.height, rec.mask.width)
        b = rec.bounds
        l, t, r, btm = int(round(b.left)), int(round(b.top)), int(round(b.right)), int(round(b.bottom))
        bw, bh = max(1, r - l), max(1, btm - t)
        if raster.shape != (bh, bw):
            import cv2
            raster = cv2.resize(raster, (bw, bh), interpolation=cv2.INTER_NEAREST)
        full = np.zeros((height, width), np.uint8)
        # clip to frame
        x0, y0 = max(0, l), max(0, t)
        x1, y1 = min(width, l + bw), min(height, t + bh)
        if x1 > x0 and y1 > y0:
            full[y0:y1, x0:x1] = raster[y0 - t:y1 - t, x0 - l:x1 - l]
        out.setdefault(fi, {})
        prev = out[fi].get(rec.object_id)
        out[fi][rec.object_id] = full if prev is None else np.maximum(prev, full)
    return out


def box_token(box, w, h) -> str:
    x1, y1, x2, y2 = [int(v) for v in box]
    return f"<|box;x1={x1};y1={y1};x2={x2};y2={y2};w={w};h={h}|>"


def point_token(x, y, w, h) -> str:
    return f"<|point;x={int(x)};y={int(y)};w={w};h={h}|>"


def sam_track_video(video_bytes: bytes, boxes: dict, width: int, height: int,
                    raw_out_path: str | None = None) -> dict:
    """Track objects through a video. `boxes` = {object_id: [x1,y1,x2,y2]} on frame 0."""
    prompt = "<0f>" + ",".join(f"{oid}{box_token(b, width, height)}" for oid, b in boxes.items())
    content = [
        {"type": "input_text", "text": prompt},
        {"type": "input_video",
         "video_url": "data:video/mp4;base64," + base64.b64encode(video_bytes).decode()},
    ]
    text = _sam_stream_text(content, metadata={"propagation_direction": "forward"})
    if raw_out_path:
        with open(raw_out_path, "w") as f:
            f.write(text)
    return parse_sam_output(text, width, height, video=True)


def sam_segment_text(jpeg_bytes: bytes, prompt: str, width: int, height: int) -> list[np.ndarray]:
    """Segment every instance of a short noun phrase on one frame. Returns full-frame binary masks."""
    content = [{"type": "input_text", "text": prompt}, image_part(jpeg_bytes)]
    text = _sam_stream_text(content)
    masks = parse_sam_output(text, width, height, video=False)
    frame = masks.get(0) or (next(iter(masks.values())) if masks else {})
    return [frame[k] for k in sorted(frame, key=lambda k: int(k) if str(k).isdigit() else 0)]


def sam_available() -> bool:
    return bool(os.environ.get("MODEL_API_KEY")) and os.environ.get("POINTER_MOCK") != "1"


def sam_segment_point(jpeg_bytes: bytes, x: int, y: int, width: int, height: int) -> np.ndarray | None:
    """Segment the object under one pixel on a single frame. Returns a binary mask or None."""
    content = [
        {"type": "input_text", "text": "<0f>1" + point_token(x, y, width, height)},
        image_part(jpeg_bytes),
    ]
    text = _sam_stream_text(content)
    masks = parse_sam_output(text, width, height, video=False)
    frame = masks.get(0) or (next(iter(masks.values())) if masks else {})
    if not frame:
        return None
    m = frame.get("1")
    return m if m is not None else next(iter(frame.values()))


# ---------------------------------------------------------------- Muse Voice Transcribe

def transcribe(wav_bytes: bytes) -> str:
    settings = {"mode": "PUSH_TO_TALK", "model": ASR_MODEL, "audioEncoding": "WAV",
                "languageBias": ["English"], "keywords": ["gripper", "finger", "torque", "derate", "cube"]}
    resp = requests.post(
        f"{BASE_URL}/asr/transcribe",
        params={"sessionId": f"pointer-{uuid.uuid4().hex[:12]}"},
        headers={"Authorization": f"Bearer {api_key()}"},
        files={
            "request": (None, json.dumps(settings), "application/json"),
            "audio": ("question.wav", wav_bytes, "audio/wav"),
        },
        timeout=120,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"transcribe failed {resp.status_code}: {resp.text[:300]}")
    return resp.json().get("transcript", "").strip()
