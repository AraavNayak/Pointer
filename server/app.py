"""Pointer web server.  Run:  uvicorn server.app:app --port 8000"""

import os
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from . import agent, meta_api  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
app = FastAPI(title="Pointer")
app.mount("/web", StaticFiles(directory=ROOT / "web"), name="web")


class AskBody(BaseModel):
    question: str
    current_ms: int = 0
    history: list[dict] = []


class RefineBody(BaseModel):
    t_ms: int
    x: int
    y: int


def _err(e: Exception, status=502):
    traceback.print_exc()
    return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=status)


@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "index.html")


@app.get("/episode/video.mp4")
def video():
    return FileResponse(agent.EP_DIR / "video.mp4", media_type="video/mp4")


@app.get("/api/status")
def status():
    return {
        "mock": os.environ.get("POINTER_MOCK") == "1",
        "has_key": bool(os.environ.get("MODEL_API_KEY")),
        "mask_source": agent.masks().get("source"),
        "models": {"reasoning": meta_api.SPARK_MODEL, "segmentation": meta_api.SAM_MODEL,
                   "speech": meta_api.ASR_MODEL},
    }


@app.get("/api/episode")
def episode():
    return agent.episode_payload()


@app.post("/api/ask")
def ask(body: AskBody):
    t0 = time.time()
    try:
        out = agent.ask(body.question, body.current_ms, body.history)
    except Exception as e:  # surface API errors to the UI
        return _err(e)
    out["latency_ms"] = int((time.time() - t0) * 1000)
    return out


@app.post("/api/refine")
def refine(body: RefineBody):
    t0 = time.time()
    try:
        out = agent.refine_point(body.t_ms, body.x, body.y)
    except Exception as e:
        return _err(e)
    out["latency_ms"] = int((time.time() - t0) * 1000)
    return out


@app.post("/api/transcribe")
async def transcribe(request: Request):
    wav = await request.body()
    if os.environ.get("POINTER_MOCK") == "1":
        return {"transcript": "Why did the gripper drop the cube?", "model": "MOCK"}
    try:
        return {"transcript": meta_api.transcribe(wav), "model": meta_api.ASR_MODEL}
    except Exception as e:
        return _err(e)
