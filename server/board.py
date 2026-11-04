"""The review board: several agents investigate in parallel, argue, and the simulator decides.

Phases (each streamed to the browser as events):

1. Investigate   Three investigators, each with a different lens (hardware, grasp physics,
                 environment and planning), run the tool loop from investigator.py at the
                 same time. They see only the episode brief until they ask for data.
2. Hypotheses    Each commits to a claim, cites evidence (times, log lines), states a
                 confidence, a prediction, and an experiment as simulator settings.
3. Cross-examine A skeptic reads every hypothesis with the evidence behind it and marks each
                 plausible, needs a test, or refuted (usually: an effect mistaken for a cause).
4. Experiments   Every hypothesis still standing is tested in MuJoCo, in parallel. Verdicts
                 come from the run's measurements, not from any model.
5. Revise        If nothing verified, the owner of the best result revises with the new
                 measurements and the board tests again (up to 2 more rounds).
6. Decision      The hypothesis whose fix verified wins. The lead writes the spoken, pointing
                 answer from the winning evidence.
"""

import json
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator

from . import meta_api
from .agent import ANSWER_SYSTEM, Episode, answer_schema, clean_patch, patch_schema, validate
from .investigator import (REVISE_SYSTEM, STEP_SYSTEM, brief, evidence_parts, ev, measurements,
                           run_fix, run_tool, step_schema, verdict)

MAX_STEPS = int(os.environ.get("POINTER_BOARD_STEPS", "4"))
REVISION_ROUNDS = 2
PARALLEL = os.environ.get("POINTER_PARALLEL", "1") != "0"

AGENTS = [
    {"id": "hardware", "name": "Hardware", "short": "HW",
     "lens": "actuators, motor drivers, temperatures, torque limits and electrical faults"},
    {"id": "grasp", "name": "Grasp", "short": "GR",
     "lens": "contact physics: grip force, friction, slip and the object's properties"},
    {"id": "environment", "name": "Environment", "short": "EN",
     "lens": "motion and planning: trajectories, speeds, obstacles, collisions and the workspace map"},
]
SKEPTIC = {"id": "skeptic", "name": "Skeptic", "short": "SK"}
LEAD = {"id": "lead", "name": "Pointer", "short": "PT"}


def lens_system(a: dict) -> str:
    return STEP_SYSTEM + (
        f"\nYou are the {a['name']} investigator on a review board of three. Your lens is {a['lens']}. "
        "The others investigate other lenses at the same time. Look for a cause in your area, but be "
        "honest: if the evidence shows your area is fine, say so. You have at most "
        f"{MAX_STEPS} tool calls.\n")


HYPOTHESIS_SYSTEM = """You are an investigator on a robot failure review board. Based only on the evidence \
you gathered, commit to a hypothesis from your lens.
- claim: one sentence: the first thing that went wrong and how it led to the failure. If your evidence \
shows your area is not at fault, say that and set confidence below 0.2.
- evidence: one to three facts you actually observed, each with the episode time (ms) and the log line \
number it came from (-1 if none).
- confidence: 0 to 1.
- prediction: one sentence: what the experiment should show if your claim is right.
- sim_patch: the controller settings that would fix the failure if your claim is right. The board will \
re-run the episode in simulation with them. Keep settings you would not change at their current value. \
"why": one sentence.
"""

SKEPTIC_SYSTEM = """You are the skeptic on a robot failure review board. Three investigators each proposed \
a hypothesis with evidence. Cross-examine each one against ALL the evidence the board gathered. Your main \
job is to separate causes from effects. Hard rule: evidence that first appears AFTER the failure time \
given below cannot be the primary cause; it can only be a consequence. A cause must come before its \
effects. Also flag claims that cite evidence the board never observed.
For each hypothesis give:
- status: "refuted" if the evidence contradicts it, "needs_test" if it is plausible but unproven, \
"plausible" if the evidence strongly supports it.
- challenge: one or two sentences, addressed to that investigator, with the specific reason. No markdown.
note: one sentence on where the board stands.
Only hypotheses that are not refuted will be tested in the simulator, so do not refute without a reason.
"""


def hypothesis_schema(keys: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "claim": {"type": "string"},
            "evidence": {"type": "array", "items": {
                "type": "object",
                "properties": {"text": {"type": "string"}, "time_ms": {"type": "integer"},
                               "log_line": {"type": "integer"}},
                "required": ["text", "time_ms", "log_line"], "additionalProperties": False}},
            "confidence": {"type": "number"},
            "prediction": {"type": "string"},
            "sim_patch": patch_schema(keys),
        },
        "required": ["claim", "evidence", "confidence", "prediction", "sim_patch"],
        "additionalProperties": False,
    }


SKEPTIC_SCHEMA = {
    "type": "object",
    "properties": {
        "reviews": {"type": "array", "items": {
            "type": "object",
            "properties": {"agent": {"type": "string", "enum": [a["id"] for a in AGENTS]},
                           "status": {"type": "string", "enum": ["plausible", "needs_test", "refuted"]},
                           "challenge": {"type": "string"}},
            "required": ["agent", "status", "challenge"], "additionalProperties": False}},
        "note": {"type": "string"},
    },
    "required": ["reviews", "note"],
    "additionalProperties": False,
}

_run_locks: dict[str, threading.Lock] = {}
_run_locks_guard = threading.Lock()


def run_fix_once(ep: Episode, params: dict) -> Episode:
    """run_fix, but two agents proposing the same settings share one simulation."""
    key = ep.id + json.dumps(params, sort_keys=True)
    with _run_locks_guard:
        lock = _run_locks.setdefault(key, threading.Lock())
    with lock:
        return run_fix(ep, params)


def failure_time(ep: Episode) -> dict | None:
    """When the episode visibly failed: the first supervisor error (object dropped / not in bin)."""
    for l in ep.log:
        if l["src"] == "supervisor" and l["level"] == "ERROR" and l["t_ms"] is not None:
            return {"t_ms": l["t_ms"], "line": l["n"], "text": l["msg"]}
    return None


def temporal_guardrail(h: dict, fail: dict | None) -> str | None:
    """Code-enforced cause-before-effect rule, applied after the skeptic whatever it decided.
    If every piece of evidence a hypothesis cites comes after the failure, it is refuted."""
    if not fail or not h["evidence"]:
        return None
    first = min(e["time_ms"] for e in h["evidence"])
    if first > fail["t_ms"]:
        return (f"Guardrail: all the evidence cited starts at {first / 1000:.2f} s, after the failure at "
                f"{fail['t_ms'] / 1000:.2f} s (log line {fail['line']}), so it can only be an effect.")
    return None


def clean_hypothesis(ep: Episode, h: dict) -> dict:
    n_log = len(ep.log)
    ev_ = []
    for e in (h.get("evidence") or [])[:3]:
        ll = int(e.get("log_line", -1))
        ev_.append({"text": str(e.get("text", "")).strip(),
                    "time_ms": max(0, min(int(e.get("time_ms", 0)), ep.meta["duration_ms"])),
                    "log_line": ll if 1 <= ll <= n_log else -1})
    try:
        conf = float(h.get("confidence", 0))
    except (TypeError, ValueError):
        conf = 0.0
    return {"claim": str(h.get("claim", "")).strip(), "evidence": ev_,
            "confidence": round(max(0.0, min(1.0, conf)), 2),
            "prediction": str(h.get("prediction", "")).strip(),
            "patch": clean_patch(ep, h.get("sim_patch") or {})}


# ------------------------------------------------------------------ the board

def run(ep: Episode, question: str, current_ms: int, history: list[dict],
        script: dict | None = None) -> Iterator[dict]:
    """Run the review board. `script` replaces model calls with fixed outputs (mock mode)."""
    t_start = time.time()
    keys = list(ep.patchable)
    defaults = {k: ep.patchable[k]["default"] for k in keys}
    model = "MOCK" if script else meta_api.REASONING_MODEL
    yield ev("start", model=model, agents=AGENTS + [SKEPTIC, LEAD], parallel=PARALLEL)

    base = [{"type": "input_text", "text": brief(ep)}]
    if history:
        convo = "\n".join(f"Engineer: {h['q']}\nPointer: {h['a']}" for h in history[-3:])
        base.append({"type": "input_text", "text": f"Earlier in this session:\n{convo}"})
    q = {"type": "input_text", "text": f"The engineer's video is paused at t={current_ms} ms.\nEngineer asks: {question}"}

    # ---- 1 + 2: parallel investigations, each ending in a hypothesis
    yield ev("phase", phase="investigate")
    out: queue.Queue = queue.Queue()
    steps_by: dict[str, list] = {a["id"]: [] for a in AGENTS}
    hyps: dict[str, dict] = {}

    def investigator(a: dict):
        try:
            steps = steps_by[a["id"]]
            for n in range(1, MAX_STEPS + 2):
                last = n > MAX_STEPS
                t0 = time.time()
                if script:
                    time.sleep(0.35 + 0.25 * AGENTS.index(a))
                    call = script["steps"][a["id"]][min(n - 1, len(script["steps"][a["id"]]) - 1)]
                else:
                    nudge = {"type": "input_text", "text": (
                        "You have used all your tool calls; finish now." if last else "Decide your next step.")}
                    call = meta_api.spark_json(lens_system(a), base + evidence_parts(ep, steps) + [q, nudge],
                                               step_schema(ep), "next_step")
                call["tool"] = "finish" if last else call.get("tool", "finish")
                result = run_tool(ep, call)
                steps.append({"call": call, "result": result})
                out.put(ev("agent_step", agent=a["id"], n=n, tool=call["tool"],
                           thought=str(call.get("thought", "")).strip(), label=result["label"],
                           summary=result["summary"], ui=result["ui"], model_ms=int((time.time() - t0) * 1000)))
                if call["tool"] == "finish":
                    break
            if script:
                raw = script["hypotheses"][a["id"]]
            else:
                raw = meta_api.spark_json(HYPOTHESIS_SYSTEM + f"\nYou are the {a['name']} investigator; your lens is "
                                          f"{a['lens']}.\n", base + evidence_parts(ep, steps) + [q],
                                          hypothesis_schema(keys), "hypothesis")
            h = clean_hypothesis(ep, raw)
            hyps[a["id"]] = h
            out.put(ev("hypothesis", agent=a["id"], **h, defaults=defaults))
        except Exception as e:  # one agent failing should not sink the board
            out.put(ev("agent_error", agent=a["id"], message=f"{type(e).__name__}: {e}"))
        finally:
            out.put(ev("_agent_done", agent=a["id"]))

    workers = [threading.Thread(target=investigator, args=(a,), daemon=True) for a in AGENTS]
    if PARALLEL:
        for w in workers:
            w.start()
    else:
        threading.Thread(target=lambda: [w.start() or w.join() for w in workers], daemon=True).start()
    remaining = len(AGENTS)
    while remaining:
        e = out.get()
        if e["type"] == "_agent_done":
            remaining -= 1
            continue
        yield e
    if not hyps:
        raise RuntimeError("No investigator produced a hypothesis.")

    # ---- 3: skeptic
    yield ev("phase", phase="cross_examine")
    fail = failure_time(ep)
    t0 = time.time()
    if script:
        time.sleep(0.6)
        sk = script["skeptic"]
    else:
        dossier = []
        for a in AGENTS:
            if a["id"] not in hyps:
                continue
            h = hyps[a["id"]]
            ev_text = "\n".join(f"  - {e['text']} (t={e['time_ms']} ms, log line {e['log_line']})" for e in h["evidence"])
            gathered = "\n".join(f"  [{s['result']['label']}]\n{s['result']['text']}" for s in steps_by[a["id"]]
                                 if s["call"]["tool"] in ("read_log", "read_telemetry"))
            dossier.append(f"=== {a['name']} investigator (lens: {a['lens']})\nClaim: {h['claim']}\n"
                           f"Confidence: {h['confidence']}\nCited evidence:\n{ev_text}\nPrediction: {h['prediction']}\n"
                           f"Proposed settings: {json.dumps({k: h['patch'][k] for k in keys})}\n"
                           f"Data this investigator read:\n{gathered}\n")
        fail_txt = (f"Failure time: {fail['t_ms']} ms (log line {fail['line']}: {fail['text']}).\n"
                    if fail else "The log has no supervisor failure line.\n")
        sk = meta_api.spark_json(SKEPTIC_SYSTEM, base + [q, {"type": "input_text", "text": fail_txt + "\n".join(dossier)}],
                                 SKEPTIC_SCHEMA, "review")
    reviews = {r["agent"]: r for r in sk.get("reviews", []) if r.get("agent") in hyps}
    for a in AGENTS:
        if a["id"] in hyps:
            r = reviews.get(a["id"], {"status": "needs_test", "challenge": ""})
            status, challenge = r["status"], str(r.get("challenge", "")).strip()
            guard = temporal_guardrail(hyps[a["id"]], fail)
            overrode = bool(guard) and status != "refuted"
            if guard:
                status = "refuted"
                challenge = (challenge + " " if challenge and not overrode else "") + guard
            hyps[a["id"]]["status"] = status
            hyps[a["id"]]["guardrail"] = bool(guard)
            yield ev("challenge", agent=SKEPTIC["id"], target=a["id"], status=status, challenge=challenge,
                     guardrail=bool(guard), overrode=overrode)
    yield ev("skeptic_note", agent=SKEPTIC["id"], note=str(sk.get("note", "")).strip(),
             model_ms=int((time.time() - t0) * 1000))

    # ---- 4: experiments in parallel
    yield ev("phase", phase="experiments")
    candidates = [a["id"] for a in AGENTS if a["id"] in hyps and hyps[a["id"]]["status"] != "refuted"
                  and hyps[a["id"]]["patch"]["changed"] and hyps[a["id"]]["confidence"] >= 0.2]
    if not candidates:  # the skeptic refuted everything: still test the most confident proposal
        ranked = sorted((aid for aid in hyps if hyps[aid]["patch"]["changed"]),
                        key=lambda aid: -hyps[aid]["confidence"])
        candidates = ranked[:1]
    for aid in hyps:
        if aid not in candidates:
            why = ("refuted by the cause-before-effect guardrail" if hyps[aid].get("guardrail") else
                   "refuted by the skeptic" if hyps[aid]["status"] == "refuted" else
                   "no change proposed" if not hyps[aid]["patch"]["changed"] else "too unsure to test")
            yield ev("experiment_skipped", agent=aid, reason=why)

    results: dict[str, dict] = {}

    def experiment(aid: str, params: dict, attempt: int) -> dict:
        t0 = time.time()
        run_ep = run_fix_once(ep, params)
        v = verdict(run_ep.meta)
        return {"agent": aid, "attempt": attempt, "params": params, "verdict": v,
                "run": run_ep.payload() | {"verdict": v}, "meas": measurements(run_ep.meta),
                "sim_ms": int((time.time() - t0) * 1000)}

    def run_batch(jobs: list[tuple[str, dict, int]]) -> Iterator[dict]:
        for aid, params, attempt in jobs:
            yield ev("experiment_start", agent=aid, attempt=attempt, params=params, defaults=defaults)
        with ThreadPoolExecutor(max_workers=3 if PARALLEL else 1) as pool:
            futs = [pool.submit(experiment, *j) for j in jobs]
            for f in futs:
                try:
                    r = f.result()
                except Exception as e:
                    yield ev("experiment_error", message=str(e))
                    continue
                results[f"{r['agent']}#{r['attempt']}"] = r
                yield ev("experiment_result", agent=r["agent"], attempt=r["attempt"], verdict=r["verdict"],
                         run=r["run"], sim_ms=r["sim_ms"])

    yield from run_batch([(aid, {k: hyps[aid]["patch"][k] for k in keys}, 1) for aid in candidates])

    # ---- 5: revise the most promising hypothesis if nothing verified
    rank = {"ok": 0, "warn": 1, "err": 2}
    for rnd in range(REVISION_ROUNDS):
        if any(r["verdict"]["kind"] == "ok" for r in results.values()) or not results:
            break
        best = min(results.values(), key=lambda r: (rank[r["verdict"]["kind"]], -hyps[r["agent"]]["confidence"]))
        aid = best["agent"]
        tried = [r for r in results.values() if r["agent"] == aid]
        if script:
            nxt = script["revisions"][aid][min(rnd, len(script["revisions"][aid]) - 1)]
        else:
            hist = "\n".join(f"Attempt {r['attempt']}: settings {json.dumps(r['params'])} -> "
                             f"{r['verdict']['label']}. {r['meas']}" for r in tried)
            nxt = meta_api.spark_json(
                REVISE_SYSTEM,
                base + evidence_parts(ep, steps_by[aid]) + [{"type": "input_text", "text": (
                    f"Your hypothesis: {hyps[aid]['claim']}\nOriginal run: {measurements(ep.meta)}\n{hist}\n"
                    "Choose the next settings.")}],
                {"type": "object", "properties": {"thought": {"type": "string"}, "sim_patch": patch_schema(keys)},
                 "required": ["thought", "sim_patch"], "additionalProperties": False}, "revision")
        p = clean_patch(ep, nxt.get("sim_patch") or {})
        params = {k: p[k] for k in keys}
        if not p["changed"] or any(r["params"] == params for r in results.values()):
            break
        yield ev("revise", agent=aid, thought=str(nxt.get("thought", "")).strip(), params=params, defaults=defaults)
        yield from run_batch([(aid, params, len(tried) + 1)])

    # ---- 6: decision and answer
    yield ev("phase", phase="decision")
    verified = [r for r in results.values() if r["verdict"]["kind"] == "ok"]
    if verified:
        win = max(verified, key=lambda r: hyps[r["agent"]]["confidence"])
    elif results:
        win = min(results.values(), key=lambda r: (rank[r["verdict"]["kind"]], -hyps[r["agent"]]["confidence"]))
    else:
        win = None
    winner = win["agent"] if win else max(hyps, key=lambda aid: hyps[aid]["confidence"])
    status = "verified" if verified else ("partial" if win and win["verdict"]["kind"] == "warn" else "unverified")
    outcomes = {}
    for aid in hyps:
        mine = [r for r in results.values() if r["agent"] == aid]
        if aid == winner and status == "verified":
            outcomes[aid] = "supported"
        elif hyps[aid].get("guardrail") and not mine:
            outcomes[aid] = "refuted by the guardrail"
        elif hyps[aid].get("status") == "refuted" and not mine:
            outcomes[aid] = "refuted by the skeptic"
        elif mine and all(r["verdict"]["kind"] == "err" for r in mine):
            outcomes[aid] = "refuted by simulation"
        elif mine:
            outcomes[aid] = "only partly fixed it" if aid != winner or status != "verified" else "supported"
        else:
            outcomes[aid] = "not tested"
    name = {a["id"]: a["name"] for a in AGENTS}
    yield ev("decision", agent=LEAD["id"], winner=winner, status=status, outcomes=outcomes,
             summary=(f"{name[winner]} investigator's hypothesis is " +
                      ("supported: its fix verified in simulation." if status == "verified" else
                       "the strongest, but no fix verified cleanly.")))

    t0 = time.time()
    others = "\n".join(f"- {name[aid]}: {hyps[aid]['claim']} -> {outcomes[aid]}" for aid in hyps if aid != winner)
    board_text = (
        f"REVIEW BOARD OUTCOME\nWinning hypothesis ({name[winner]} investigator): {hyps[winner]['claim']}\n"
        f"Its evidence: " + "; ".join(f"{e['text']} (t={e['time_ms']} ms, log line {e['log_line']})"
                                      for e in hyps[winner]["evidence"]) + "\n"
        + (f"Simulation test of its fix: {win['verdict']['text']} Settings {json.dumps(win['params'])}\n" if win else "")
        + f"Other hypotheses:\n{others}\n"
        "In the takeaway, the sim_patch must be exactly the tested settings above. The engineer has the "
        "review board in another tab, so explain the failure itself; mention the simulation result in the "
        "takeaway say. Write times in seconds (for example 3.6 seconds), not milliseconds.")
    if script:
        time.sleep(0.4)
        raw = script["answer"]
    else:
        evidence = evidence_parts(ep, steps_by[winner]) + [
            p for aid in hyps if aid != winner for p in evidence_parts(ep, steps_by[aid]) if p["type"] == "input_text"]
        raw = meta_api.spark_json(ANSWER_SYSTEM, base + evidence + [{"type": "input_text", "text": board_text}, q],
                                  answer_schema(keys), "pointer_answer")
    ans = validate(ep, raw)
    if win:
        ans["takeaway"]["sim_patch"] = clean_patch(ep, win["params"] | {"why": ans["takeaway"]["sim_patch"].get("why", "")})
    ans |= {"model": model, "model_ms": int((time.time() - t0) * 1000),
            "total_ms": int((time.time() - t_start) * 1000), "winner": winner, "status": status,
            "experiments": [{"agent": r["agent"], "attempt": r["attempt"], "params": r["params"],
                             "verdict": r["verdict"], "run_id": r["run"]["meta"]["episode_id"]}
                            for r in sorted(results.values(), key=lambda r: (r["attempt"], r["agent"]))],
            "final_run_id": win["run"]["meta"]["episode_id"] if win else None}
    yield ev("answer", answer=ans)
    yield ev("done", total_ms=int((time.time() - t_start) * 1000))
