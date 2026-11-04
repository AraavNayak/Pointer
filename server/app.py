"""Pointer web server.  Run:  uvicorn server.app:app --port 8000"""

import json
import os
import random
import re
import subprocess
import sys
import threading
import traceback
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from . import agent, investigator, meta_api  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
app = FastAPI(title="Pointer")
app.mount("/web", StaticFiles(directory=ROOT / "web"), name="web")


@app.on_event("startup")
def _warm():
    # Off by default: each warm-up is a full paid investigation per episode.
    if (os.environ.get("POINTER_WARM") == "1" and os.environ.get("POINTER_MOCK") != "1"
            and (os.environ.get("MODEL_API_KEY") or meta_api.USING_TEST_MODEL)):
        investigator.warm([e["id"] for e in agent.list_episodes()], agent.get_episode)


class InvestigateBody(BaseModel):
    episode_id: str
    question: str
    current_ms: int = 0
    history: list[dict] = []


class RefineBody(BaseModel):
    episode_id: str
    t_ms: int
    x: int
    y: int


def _episode(eid: str):
    try:
        return agent.get_episode(eid), None
    except KeyError as e:
        return None, JSONResponse({"error": str(e)}, status_code=404)


@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "index.html")


@app.get("/api/status")
def status():
    return {
        "mock": os.environ.get("POINTER_MOCK") == "1",
        "has_key": bool(os.environ.get("MODEL_API_KEY")) or meta_api.USING_TEST_MODEL,
        "test_model": meta_api.USING_TEST_MODEL,
        "models": {"reasoning": meta_api.REASONING_MODEL, "segmentation": meta_api.SAM_MODEL,
                   "speech": meta_api.ASR_MODEL},
    }


@app.get("/api/episodes")
def episodes():
    return {"episodes": agent.list_episodes(), "default": agent.DEFAULT_EPISODE}


@app.get("/api/episode")
def episode(id: str = agent.DEFAULT_EPISODE):
    ep, err = _episode(id)
    return err or ep.payload()


@app.get("/episode/{eid}/video.mp4")
def video(eid: str):
    if not re.fullmatch(r"[a-z0-9_]+", eid) or not (agent.EPISODES_DIR / eid / "video.mp4").exists():
        return JSONResponse({"error": "unknown episode"}, status_code=404)
    return FileResponse(agent.EPISODES_DIR / eid / "video.mp4", media_type="video/mp4")


@app.post("/api/investigate")
def investigate(body: InvestigateBody):
    """Server-sent events: start, step..., answer, fix_attempt/fix_result..., fix_done, done."""
    ep, err = _episode(body.episode_id)
    if err:
        return err

    def stream():
        try:
            for e in investigator.run_or_replay(ep, body.question, body.current_ms, body.history):
                yield f"data: {json.dumps(e)}\n\n"
        except Exception as e:  # surface API errors to the UI
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'message': f'{type(e).__name__}: {e}'})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


_mystery_lock = threading.Lock()


@app.post("/api/mystery")
def new_mystery():
    """Generate a fresh mystery episode in MuJoCo: random fault type, parameters and decoys."""
    seed = random.SystemRandom().randint(1000, 9999)
    env = dict(os.environ)
    if sys.platform.startswith("linux"):
        env.setdefault("MUJOCO_GL", "egl")
    with _mystery_lock:
        proc = subprocess.run([sys.executable, str(ROOT / "sim" / "mystery.py"), str(agent.EPISODES_DIR),
                               "--seed", str(seed)], capture_output=True, text=True, env=env, timeout=900)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-4:]
        return JSONResponse({"error": "Could not generate a mystery: " + " | ".join(tail)}, status_code=500)
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    return {"id": out["id"], "episodes": agent.list_episodes()}


@app.get("/api/mystery/{eid}/answer")
def mystery_answer(eid: str):
    """The answer key. The review board never calls this; the browser asks for it only after
    the board has published its decision."""
    ep, err = _episode(eid)
    if err:
        return err
    if not ep.is_mystery:
        return JSONResponse({"error": "not a mystery episode"}, status_code=400)
    return ep.answer_key()


@app.post("/api/refine")
def refine(body: RefineBody):
    ep, err = _episode(body.episode_id)
    if err:
        return err
    try:
        return agent.refine_point(ep, body.t_ms, body.x, body.y)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=502)


@app.post("/api/transcribe")
async def transcribe(request: Request):
    wav = await request.body()
    if os.environ.get("POINTER_MOCK") == "1":
        return {"transcript": "Why did this run fail?", "model": "MOCK"}
    try:
        return {"transcript": meta_api.transcribe(wav), "model": meta_api.ASR_MODEL}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=502)
