# Pointer

**An AI co-pilot that points at why your robot failed.**

You replay a failed robot episode and ask, out loud, "Why did the gripper drop the cube?"
Pointer answers by voice and, while it talks, points at the evidence in every view at once:
it outlines the object in the camera video, puts a cursor on the exact moment in the
telemetry, and highlights the controller log line that explains it. A pink thread links
all three, so you see the causal chain instead of reading a paragraph and hunting for
timestamps.

Built for the Meta Global AI Developer Hackathon, *Push the Frontier* track.

## How it uses Meta's models

| Model | What it does in Pointer |
|---|---|
| **Muse Spark** (`muse-spark-1.3`, Meta Model API) | Reads 13 video frames, the 100 Hz telemetry table and the numbered controller log, and answers with a short spoken causal chain. Each sentence comes back as structured JSON that *references* evidence: a timestamp, which tracked objects to outline, which signal to highlight, which log line to point at, and optionally a raw pixel point. |
| **SAM 3.1** (`sam-3.1`, Meta Model API) | Tracks the cube and the gripper through the whole video from one box prompt each on frame 0, so the outline stays locked on the object as it falls. When Muse Spark points at something that isn't tracked (e.g. a single finger pad), SAM 3.1 segments whatever is under that pixel live. |
| **Muse Voice Transcribe** (`muse-voice-transcribe-1.0`) | Push-to-talk speech to text. Pressing the mic mid-answer stops the current answer (barge-in). |

The key design idea is **grounding by reference, not by guessing pixels**. Language
models are poor at returning exact coordinates. So Muse Spark decides *what* matters and
*when*, and SAM 3.1 and the app's own renderers decide *where*. `eval/benchmark.py`
measures the difference on this episode against pixel-exact ground truth.

## The episode

`sim/generate_episode.py` builds a deterministic MuJoCo pick-and-place episode:
a parallel-jaw gripper picks up a red cube and carries it toward a blue bin. Under the
sustained 12 N grip, the left finger's motor driver heats up, warns at 70 °C, and at
80 °C derates its torque limit to 0.6 N. The right finger keeps squeezing, the cube
is pushed against the weakened left jaw, slips out and lands short of the bin.

The thermal fault is an injected fault, which is standard practice for testing failure
analysis. Everything downstream of it (contact forces, slip, the fall, the video) is
real physics from the simulator. Outputs live in `episodes/pick_place_drop_001/`:
`video.mp4`, `telemetry.csv`, `controller.log`, `gt_masks.json` (MuJoCo segmentation
ground truth) and `meta.json`.

## Run it (macOS)

Requires Python 3.10+.

```bash
./run.sh                                  # first run installs into .venv, creates .env, then stops
# open .env and set MODEL_API_KEY=...
.venv/bin/python -m sam.precompute_masks  # SAM 3.1 tracking, run once
./run.sh                                  # http://localhost:8000
```

Open http://localhost:8000 in Chrome. Hold the mic button (or hold the space bar), ask a
question, release. You can also type a question or click a suggestion.

Keyboard: hold **space** to talk, **K** play or pause, **← →** step one frame.

Until `sam/precompute_masks.py` has run, the app outlines objects with the simulator's
ground-truth masks and says so in the video panel ("Object masks from simulator ground
truth"). After it runs, the panel reads "Objects tracked by SAM 3.1".

To try the interface without API calls, set `POINTER_MOCK=1` in `.env`. Answers are then
canned UI fixtures and the header shows a "Mock answers" badge.

## Benchmark

```bash
.venv/bin/python -m eval.benchmark 24
```

On 24 evenly spaced frames, it asks Muse Spark directly for a bounding box of the cube and
the gripper, and compares those raw coordinates with SAM 3.1's tracked boxes. Both are
scored by IoU against MuJoCo's pixel-exact segmentation. It also reports SAM 3.1 mask IoU
over all 180 frames. Results are written to `eval/results.json`.

## Regenerate the episode

```bash
.venv/bin/python sim/generate_episode.py            # macOS
MUJOCO_GL=egl python sim/generate_episode.py        # Linux
```

## Layout

```
sim/generate_episode.py   MuJoCo scene, injected fault, video/telemetry/log/ground truth
sam/precompute_masks.py   SAM 3.1 video tracking -> episodes/<id>/sam_masks.json
server/meta_api.py        Meta Model API client: Muse Spark, SAM 3.1, Muse Voice Transcribe
server/agent.py           context building, answer schema, validation
server/app.py             FastAPI: /api/episode, /api/ask, /api/refine, /api/transcribe
web/                      single-page UI (video + overlay, charts, log, laser thread, voice)
eval/benchmark.py         raw Muse Spark pointing vs SAM 3.1 grounding
```

## Built during the hackathon

All code in this repository, the simulated episode and the UI were written during the
hackathon building period (Oct 27 to Nov 4, 2026). Third-party pieces used as-is:
MuJoCo, FastAPI, the OpenAI Python SDK (pointed at the Meta Model API), and Meta's
`meta-sam-parser` package for decoding SAM 3.1 masks.
