"""Mystery benchmark: how often does the review board find a cause nobody scripted?

For each seed, MuJoCo generates a mystery episode (random fault type, random parameters,
decoy warnings; see sim/mystery.py). The full review board then runs on it blind, with real
Muse Spark (and SAM 3.1 if MODEL_API_KEY is set), exactly as in the app. Only afterwards is the
board's decision compared with the answer key.

  found     the winning hypothesis came from the investigator whose area holds the planted fault
  verified  a proposed fix re-ran cleanly in MuJoCo (no drop, derate, slip or collision)
  both      found and verified

Chance level for "found" is 1 in 3 (three investigators).

Run:  python -m eval.mystery_bench --n 20            seeds 1..20
      python -m eval.mystery_bench --seeds 7 12 40   specific seeds
Results are saved after every episode to eval/mystery_results.json (re-running skips seeds
already scored; --fresh starts over) and summarised in eval/mystery_results.md.
Each session makes about 15 to 20 Muse Spark calls; the totals are reported so you can check cost.
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sim"))
load_dotenv(ROOT / ".env")

import mystery  # noqa: E402

from server import board, meta_api  # noqa: E402
from server.agent import Episode  # noqa: E402

OUT_DIR = ROOT / "eval" / "mystery_episodes"      # kept apart from the app's episode list
RESULTS = ROOT / "eval" / "mystery_results.json"
REPORT = ROOT / "eval" / "mystery_results.md"
QUESTION = "Why did this run fail?"
FAULT = {"thermal": "overheating driver", "slip": "slippery part", "collision": "unmapped fixture"}


def run_one(seed: int) -> dict:
    ep_dir = OUT_DIR / f"mystery_{seed}"
    if not (ep_dir / "meta.json").exists():
        mystery.generate(OUT_DIR, seed)
    ep = Episode(ep_dir)
    key = ep.answer_key()
    before = dict(meta_api.USAGE)
    t0 = time.time()
    decision, answer, errors, n_exp = None, None, [], 0
    for e in board.run(ep, QUESTION, 0, []):
        t = e["type"]
        if t == "decision":
            decision = e
        elif t == "answer":
            answer = e["answer"]
        elif t == "experiment_result":
            n_exp += 1
        elif t in ("agent_error", "experiment_error"):
            errors.append(e.get("message", ""))
    used = {k: meta_api.USAGE.get(k, 0) - before.get(k, 0) for k in meta_api.USAGE}
    found = bool(decision) and decision["winner"] == key["expected_agent"]
    verified = bool(decision) and decision["status"] == "verified"
    return {
        "seed": seed, "fault": key["scenario"], "planted": key["planted"], "expected": key["expected_agent"],
        "winner": decision["winner"] if decision else None, "status": decision["status"] if decision else "error",
        "found": found, "verified": verified, "both": found and verified,
        "root_cause": answer["takeaway"]["root_cause"] if answer else "",
        "experiments": n_exp, "seconds": round(time.time() - t0, 1), "errors": errors, **used,
    }


def summarize(rows: list[dict]) -> str:
    n = len(rows)
    if not n:
        return "No results yet."

    def pct(k, rs):
        return f"{sum(r[k] for r in rs)}/{len(rs)} ({100 * sum(r[k] for r in rs) / len(rs):.0f}%)"

    lines = [
        "# Mystery benchmark",
        "",
        f"{n} mystery episodes, each with a random hidden fault, random parameters and two decoy warnings. "
        f"Reasoning model: {meta_api.REASONING_MODEL}. Segmentation: "
        f"{meta_api.SAM_MODEL if meta_api.sam_available() else 'simulator fallback (no MODEL_API_KEY)'}.",
        "",
        "| | Found the planted cause | Verified a fix | Both |",
        "|---|---|---|---|",
        f"| **All** | {pct('found', rows)} | {pct('verified', rows)} | {pct('both', rows)} |",
    ]
    for f in ("thermal", "slip", "collision"):
        rs = [r for r in rows if r["fault"] == f]
        if rs:
            lines.append(f"| {FAULT[f]} | {pct('found', rs)} | {pct('verified', rs)} | {pct('both', rs)} |")
    calls = [r.get("spark_calls", 0) for r in rows]
    toks = [r.get("input_tokens", 0) + r.get("output_tokens", 0) for r in rows]
    secs = [r["seconds"] for r in rows]
    lines += [
        "",
        "Chance level for finding the cause is 33% (one of three investigators).",
        "",
        f"Per session: {sum(secs) / n:.0f} s on average, {sum(calls) / n:.1f} Muse Spark calls, "
        f"{sum(toks) / n / 1000:.0f}k tokens, {sum(r.get('sam_calls', 0) for r in rows) / n:.1f} SAM 3.1 calls, "
        f"{sum(r['experiments'] for r in rows) / n:.1f} MuJoCo experiments.",
        "",
        "| Seed | Planted fault | Board's winner | Fix | Found | Board's root cause |",
        "|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: r["seed"]):
        lines.append(f"| {r['seed']} | {FAULT[r['fault']]} | {r['winner'] or 'error'} | {r['status']} | "
                     f"{'yes' if r['found'] else 'no'} | {r['root_cause'].replace('|', '/')[:140]} |")
    confusions = Counter((r["expected"], r["winner"]) for r in rows if not r["found"])
    if confusions:
        lines += ["", "Misses (planted area → board's pick): " +
                  ", ".join(f"{a} → {b}: {c}" for (a, b), c in confusions.most_common())]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20, help="run seeds 1..N")
    ap.add_argument("--seeds", type=int, nargs="*", help="run these seeds instead")
    ap.add_argument("--fresh", action="store_true", help="ignore saved results")
    a = ap.parse_args()
    if os.environ.get("POINTER_MOCK") == "1":
        sys.exit("POINTER_MOCK=1 uses scripted agents; unset it to benchmark the real board.")
    if not (os.environ.get("MODEL_API_KEY") or meta_api.USING_TEST_MODEL):
        sys.exit("Set MODEL_API_KEY in .env.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [] if a.fresh or not RESULTS.exists() else json.loads(RESULTS.read_text())
    done = {r["seed"] for r in rows}
    seeds = a.seeds or list(range(1, a.n + 1))
    for seed in seeds:
        if seed in done:
            continue
        try:
            r = run_one(seed)
        except Exception as e:  # keep going; record the failure
            r = {"seed": seed, "fault": "?", "planted": "", "expected": "", "winner": None, "status": "error",
                 "found": False, "verified": False, "both": False, "root_cause": f"{type(e).__name__}: {e}",
                 "experiments": 0, "seconds": 0, "errors": [str(e)]}
        rows.append(r)
        RESULTS.write_text(json.dumps(rows, indent=2))
        REPORT.write_text(summarize([x for x in rows if x["fault"] != "?"]))
        print(f"seed {seed:>4}  {FAULT.get(r['fault'], '?'):<19} winner {str(r['winner']):<12} "
              f"{r['status']:<10} found={'yes' if r['found'] else 'no ':<3}  {r['seconds']:>5.0f}s  "
              f"{r.get('spark_calls', 0)} calls", flush=True)
    print()
    print(summarize([x for x in rows if x["fault"] != "?"]))


if __name__ == "__main__":
    main()
