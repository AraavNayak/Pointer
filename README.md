# Pointer

**A review board of AI agents that debates why your robot failed, and lets the physics simulator decide who is right.**

Pick a failed robot episode and ask, out loud, "Why did this run fail?" Three investigator
agents, all running on Muse Spark, look into it at the same time, each from a different
angle. They commit to competing hypotheses, a skeptic agent cross-examines them, and every
hypothesis still standing has its fix tested in MuJoCo. The one whose fix actually works
wins. Pointer then explains the failure by voice while pointing at the evidence in every
view at once: the object outlined in the video (tracked by SAM 3.1), a cursor on the
telemetry, and the log line, linked by a pink thread.

<img width="1512" height="982" alt="Screenshot 2026-11-04 at 12 46 51 PM" src="https://github.com/user-attachments/assets/3e388179-fc8a-4635-8e48-30baf40e5114" />

Built for the Meta Global AI Developer Hackathon (Indie Developer track), answering the
*Push the Frontier* problem statement: multimodal agents that understand a physical
system's failure and prove a fix in physics simulation.

## The review board

Hypotheses compete; physics decides. The bottom-right panel has two tabs: **Review board**
shows the session live, and **Findings** holds the spoken explanation, the root cause and
every fix that was tested.

```
            ┌ Hardware investigator ─────┐
question ──▶├ Grasp investigator ────────┤──▶ hypotheses ──▶ Skeptic ──▶ MuJoCo experiments ──▶ decision ──▶ findings
 (parallel) └ Environment investigator ──┘   + evidence     cross-       (parallel; verdicts    (winner's     (spoken,
                                              + confidence  examines     from measurements;      fix is        pointing)
                                              + experiment               revise if needed)       verified)
```

1. **Investigate, in parallel.** Three investigators (hardware: actuators, drivers,
   temperatures; grasp: contact forces, friction, slip; environment: motion, obstacles,
   the map) start with only a brief of the episode and must ask for data. Each step they
   write a short thought and call one tool: `read_log`, `read_telemetry`, `view_frames`,
   or `segment`, which runs SAM 3.1 on chosen frames (see below).
   As they work, the views show what they are looking at.
2. **Hypotheses.** Each commits to a claim with cited evidence (times, log lines), a
   confidence, a prediction, and an experiment expressed as controller settings.
3. **Cross-examine.** A skeptic agent reads every hypothesis with the data behind it and
   marks each plausible, needs a test, or refuted, mostly by separating causes from
   effects ("that alarm fired after the drop, so it can't be the cause"). A
   code-enforced guardrail backs it up (see below).
4. **Experiments.** Every hypothesis still standing has its fix re-run in MuJoCo, in
   parallel. The verdict is computed from the run, not by a model: did the object reach
   the bin, and did the driver derate, the object slip, or the load hit an obstacle? A fix
   that only gets lucky is reported as **marginal**.
5. **Revise.** If nothing verified, the investigator with the best result sees the new
   measurements, revises its fix, and the board tests again (up to two more rounds).
6. **Decision and findings.** The hypothesis whose fix verified wins. The lead writes the
   spoken, pointing explanation from the winning evidence, then switches the view to the
   verified run, with the original run's telemetry dashed underneath.

Everything streams to the browser as server-sent events (`/api/investigate`).

### Guardrails against a wrong board

- **Cause before effect, enforced in code.** The failure time is taken from the log (the
  supervisor's first error). The skeptic's prompt states it with a hard rule: evidence
  that first appears after the failure cannot be the primary cause. Then
  `temporal_guardrail()` in `server/board.py` checks every hypothesis itself: if all the
  evidence it cites comes after the failure, it is refuted, whatever the skeptic said.
  The board shows a "Guardrail" badge, or "Guardrail overrode" when the code had to
  overrule the skeptic.
- **The skeptic only prunes; physics decides.** If the skeptic lets a wrong hypothesis
  through, it still has to pass its own MuJoCo experiment, and the verdict comes from the
  run's measurements. A wrong cause does not produce a clean fix.
- **One failing agent does not sink the board.** Errors are reported per agent and the
  others continue.

## Mystery episodes: nothing scripted

"New mystery" asks MuJoCo to simulate a failure nobody wrote an answer for
(`sim/mystery.py`):

- The fault type is drawn at random: an overheating driver, a slippery part, or an
  unmapped fixture.
- Its parameters are random too: heating rate and thermal limits; friction and grip
  force; fixture height, width and position.
- The object is a generic "part" in a random color.
- Two decoy warnings, harmless but real-looking, land at random times in the log.
- Every mystery's log opens with identical boilerplate.
- The draw repeats until the run really fails.

The hidden setup is stored with the episode, but the server strips it from everything the
agents and the browser receive. The board gets exactly the same tools and brief as for any
other episode, and its fix experiments reproduce the hidden setup. Only after the board
publishes its decision does the browser fetch the answer key (`/api/mystery/<id>/answer`)
and show whether the board found the planted cause.

The failure physics is deterministic once drawn. What is not scripted is the agents' path:
which log windows, signals and frames to request, which hypothesis to commit to, what the
skeptic rejects, and which fix to try. The prompts describe each investigator's area in
general terms and never name the faults.

`server/ui_test_fixtures.py` contains canned agent outputs for the built-in episodes. They
are used only with `POINTER_MOCK=1` to test the interface without model calls; every line
is prefixed "MOCK", the app shows a "Mock answers" badge, and the board never imports the
file otherwise.

## The three episodes

All three are deterministic MuJoCo pick-and-place runs with a parallel-jaw gripper. Each
fails for a different reason, so the agent has to actually investigate.

| Episode | What goes wrong | Evidence | Fix that verifies |
|---|---|---|---|
| Gripper drops the cube mid-transport | The left finger's motor driver heats up under a sustained 12 N grip, warns at 70 °C and derates to 0.6 N at 80 °C. The right finger squeezes the cube out of the weakened jaw. | Driver temperature, left actuator force collapse, thermal log lines | Hold force 6 N (8 N is marginal: it still derates later) |
| Steel block slides out during the lift | The part is an oily steel block (friction 0.25). The "eco" grip profile assumes the catalog friction (0.7) and grips at 5.1 N, which cannot hold it while it accelerates upward. | Normal contact force looks healthy, but the block slides and falls with no hardware alarm; the video shows a shiny metal part | Hold force about 8 to 9 N |
| Cube is knocked off by a fixture | A fixture that is missing from the planner's map sits between the pick and the bin. The carry height is too low to clear it. | Obstacle contact force spike, x-axis tracking error, planner log says the path is clear | Carry height 18 cm (16 cm is marginal: it still scrapes) |

### How realistic is this?

The parts that are real physics: MuJoCo simulates the rigid bodies, gravity, contact and
friction. The object is held only by friction from the two finger pads, so slipping,
falling, being squeezed out of a weak jaw and being knocked off by a fixture all emerge
from the contact solver, not from a script. Contact forces, positions and the video come
straight from the simulation.

The parts that are simplified or injected: the motor driver's temperature is a simple
first-order model (heats in proportion to force, cools toward ambient) with constants
chosen so the failure happens mid-transport, and the derate is a rule, not physics. Real
drivers do protect themselves thermally like this, but the numbers are not calibrated to
any real product. The gripper is a Cartesian gantry rather than a jointed arm, the camera
is fixed, and the sensors have no noise. The fixture and the friction mismatch are
planted causes, which is how fault injection is normally used to test diagnosis tools.

Why it matters: each episode reproduces a real class of failure. A fault in one place
shows up as a symptom somewhere else, and the evidence is split across video, telemetry
and logs. That is exactly what makes robot debugging slow. The simulator also gives exact
ground truth for the benchmark, and lets the agent test its fix instead of just
suggesting it, which is how robotics teams validate changes before touching hardware.

## How it uses Meta's models

| Model | What it does in Pointer |
|---|---|
| **Muse Spark** (`muse-spark-1.3`, Meta Model API) | Every agent on the board: the three investigators (choosing each step, reading logs, telemetry and frames, committing to hypotheses), the skeptic, the revisions, and the lead who writes the spoken pointing answer as structured JSON. |
| **SAM 3.1** (`sam-3.1`, Meta Model API) | A measuring instrument for the agents: the `segment` tool runs SAM 3.1 with a text prompt ("part", "gripper finger", "gray box") on frames an investigator picks, and returns boxes, centers, areas and the gap between two objects, computed from the masks. The video shows each agent's masks in its color as it works. SAM 3.1 also tracks the object and gripper through each video from one box prompt, so the pointing outlines stay locked on, and segments anything the final answer points at that isn't tracked. |
| **Muse Voice Transcribe** (`muse-voice-transcribe-1.0`) | Push-to-talk speech to text. Pressing the mic mid-answer stops the answer (barge-in). |

The key design idea is **grounding by reference, not by guessing pixels**. Language models
are poor at returning exact coordinates. So Muse Spark decides *what* matters and *when*;
SAM 3.1 and the app's own renderers decide *where*. `eval/benchmark.py` measures the
difference against pixel-exact ground truth.

## Run it (macOS)

Requires Python 3.10+.

```bash
./run.sh                                  # first run installs into .venv, creates .env, then stops
# open .env and set MODEL_API_KEY=...
./run.sh                                  # http://localhost:8000
```

Optional, once per episode: SAM 3.1 tracking (two small API calls each). Until it has run
for an episode, the video panel says "Object masks from simulator ground truth".

```bash
.venv/bin/python -m sam.precompute_masks pick_place_drop_001
.venv/bin/python -m sam.precompute_masks pick_place_slip_002
.venv/bin/python -m sam.precompute_masks pick_place_collision_003
```

Pick an episode from the dropdown. Hold the mic button (or the space bar), ask a question,
release. You can also type, or click a suggestion. Keyboard: hold **space** to talk,
**K** play or pause, **← →** step one frame.

Check that MuJoCo renders on your machine (the agent's fix runs need it):

```bash
.venv/bin/python sim/generate_episode.py /tmp/pointer-test --scenario collision --params '{"transport_height_cm": 18}'
```

## Speed and cost

A board session is about 18 small Muse Spark calls (each around 30 KB, since agents only
pull the data they ask for), three of them at a time, plus two to four MuJoCo runs in
parallel. `python -m eval.latency` times every step. Simulation runs are cached by their
settings. If parallel calls hit rate limits, set `POINTER_PARALLEL=0`.

Before recording a demo, set `POINTER_WARM=1`. At startup, the server runs the first
suggested question on every episode in the background; asking it again replays that run
quickly, and the board says it is a replay. It is off by default because it costs a full
board session per episode.

`POINTER_MOCK=1` runs the interface with scripted agents (no model calls; the tools and
MuJoCo runs are real) and shows a "Mock answers" badge. `TEST_MODEL` and `TEST_API_KEY`
swap in any OpenAI-compatible model as a stand-in while testing (the header then says
"Test model, not Muse Spark"). Remove both for the demo and submission.

## SAM 3.1 as an agent tool

Language models are poor at reading exact positions off an image, so investigators don't
estimate them. They call `segment` with a frame time and a short noun phrase, and SAM 3.1
returns the masks. The tool then measures in code:

```
Frame 114 at t=3800 ms, "part": 1 match
  #1: box (140,102)-(160,126), center (152,114), area 260 px; vs "gripper finger": gap 0 px, center 48 px below
Frame 118 at t=3950 ms, "part": 1 match
  #1: box (154,140)-(174,159), center (164,150), area 325 px; vs "gripper finger": gap 18 px, center 82 px below
```

That is how an agent turns "it looks like it fell" into "the part left the fingers between
3.80 and 3.95 s and is 18 px clear of them". Results are cached per frame and prompt. Without
a Meta key (mock mode, or a test model), the tool falls back to the simulator's segmentation
for the object and gripper, and both its output and the video label say so.

## Mystery benchmark: accuracy on unscripted failures

```bash
.venv/bin/python -m eval.mystery_bench --n 20
```

This generates 20 seeded mystery episodes (fault type, parameters and decoys all random),
runs the full review board on each one blind, exactly as in the app, and only then compares
the decision with the answer key. It reports how often the board found the planted cause
(chance is 1 in 3), how often a fix verified in MuJoCo, and both, broken down by fault type,
with average time, Muse Spark calls, tokens, SAM 3.1 calls and experiments per session.
Results are saved after every episode (`eval/mystery_results.json`, summary in
`eval/mystery_results.md`), and a re-run skips seeds already scored.

## Pointing benchmark

```bash
.venv/bin/python -m eval.benchmark 24
```

On 24 evenly spaced frames, it asks Muse Spark directly for bounding boxes of the object and
the gripper, and compares those raw coordinates with SAM 3.1's tracked boxes. Both are
scored by IoU against MuJoCo's pixel-exact segmentation. Results go to `eval/results.json`.

## Layout

```
sim/generate_episode.py   MuJoCo scenes, three failure scenarios, patchable controller settings
server/board.py           the review board: parallel investigators, hypotheses, skeptic + guardrail, experiments, decision
server/ui_test_fixtures.py  canned agent outputs for POINTER_MOCK=1 only
sim/mystery.py            random hidden failures with decoys, for blind diagnosis
server/investigator.py    agent tools (log, telemetry, frames), MuJoCo runner, verdicts, replay cache
server/agent.py           episodes, answer prompt and schema, validation
server/meta_api.py        Meta Model API client: Muse Spark, SAM 3.1, Muse Voice Transcribe
server/app.py             FastAPI: /api/episodes, /api/episode, /api/investigate (SSE), /api/mystery, /api/refine, /api/transcribe
sam/precompute_masks.py   SAM 3.1 video tracking -> episodes/<id>/sam_masks.json
eval/mystery_bench.py     board accuracy on N random mysteries, scored against hidden answer keys
web/                      single-page UI (video + overlay, charts, log, agent trace, voice)
eval/benchmark.py         raw Muse Spark pointing vs SAM 3.1 grounding
eval/latency.py           per-step timing of one board session
```

## Built during the hackathon

All code in this repository, the simulated episodes and the UI were written during the
hackathon building period (Oct 27 to Nov 4, 2026). Third-party pieces used as-is: MuJoCo,
FastAPI, the OpenAI Python SDK (pointed at the Meta Model API), and Meta's
`meta-sam-parser` package for decoding SAM 3.1 masks.
