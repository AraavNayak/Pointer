"""Generate a mystery episode: a failure nobody scripted an answer for.

The fault type (overheating driver, slippery part, unmapped fixture) and its parameters are
drawn at random from a seed, two decoy warnings are sprinkled into the controller log, and
the object is renamed "part" with a random color. The run is re-drawn until it really fails.
The hidden parameters are stored in meta.json under "mystery"; the server strips them from
everything the agents and the browser see, and only reveals them through the answer-key
endpoint after the review board has decided.

Run:  python sim/mystery.py [out_root] [--seed N]      prints {"id": ..., "attempts": ...}
"""

import argparse
import contextlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_episode as sim  # noqa: E402

COLORS = ["0.86 0.16 0.18 1", "0.95 0.55 0.15 1", "0.20 0.65 0.55 1", "0.92 0.80 0.20 1", "0.55 0.35 0.80 1"]
DECOYS = [
    ("WARN", "perception", "exposure auto-adjust  gain=+1.5dB (glare on table)"),
    ("WARN", "net", "telemetry packet delayed 42ms (resent)"),
    ("WARN", "finger_R", "encoder jitter 0.3mm (within tolerance)"),
    ("WARN", "planner", "replan request from operator UI ignored (no change)"),
    ("WARN", "controller", "z-axis effort 18% above model during lift"),
    ("WARN", "power", "bus voltage dip 23.4V (nominal 24V)"),
]


def draw(rng: random.Random, force: str | None = None) -> tuple[str, dict]:
    scenario = rng.choice(["thermal", "slip", "collision"])
    scenario = force or scenario
    h: dict = {"object": "part", "rgba": rng.choice(COLORS), "uniform_log": True,
               "decoys": [[round(rng.uniform(0.4, 5.6), 2), *d] for d in rng.sample(DECOYS, 2)]}
    if scenario == "thermal":
        h["heat_gain"] = round(rng.uniform(2.3, 3.2), 2)
        h["temp_hard"] = round(rng.uniform(76, 86), 1)
        h["temp_soft"] = round(h["temp_hard"] - rng.uniform(8, 12), 1)
    elif scenario == "slip":
        mu = round(rng.uniform(0.18, 0.30), 3)
        ratio = rng.uniform(0.90, 1.03)       # grip capacity vs. the part's weight
        h["friction"] = mu
        h["grip_force_N"] = round(min(9.0, max(2.5, ratio * 0.25 * 9.81 / (2 * mu))), 1)
    else:
        top = round(rng.uniform(0.150, 0.185), 3)
        h["fixture"] = {"x": round(rng.uniform(0.30, 0.41), 3), "half_x": round(rng.uniform(0.025, 0.04), 3),
                        "half_y": 0.13, "top": top,
                        "rgba": rng.choice(["0.42 0.47 0.40 1", "0.45 0.42 0.38 1", "0.36 0.42 0.50 1"])}
    return scenario, h


def generate(out_root: Path, seed: int, max_attempts: int = 8, force: str | None = None) -> dict:
    rng = random.Random(seed)
    episode_id = f"mystery_{seed}"
    for attempt in range(1, max_attempts + 1):
        scenario, hidden = draw(rng, force)
        hidden |= {"seed": seed, "scenario": scenario, "title": f"Mystery episode {seed}"}
        with contextlib.redirect_stdout(sys.stderr):   # keep stdout for the one-line result
            sim.main(out_root, episode_id, {}, scenario, hidden)
        meta = json.loads((out_root / episode_id / "meta.json").read_text())
        if meta["outcome"].startswith("FAIL"):
            return {"id": episode_id, "attempts": attempt}
    raise RuntimeError("could not draw a failing mystery episode")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_root", nargs="?", default=str(Path(__file__).resolve().parent.parent / "episodes"))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--scenario", default=None, choices=["thermal", "slip", "collision"], help="testing only")
    a = ap.parse_args()
    seed = a.seed if a.seed is not None else random.SystemRandom().randint(1000, 9999)
    print(json.dumps(generate(Path(a.out_root), seed, force=a.scenario)))
