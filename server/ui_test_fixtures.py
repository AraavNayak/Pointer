"""UI test fixtures for POINTER_MOCK=1 only.

These scripted agent outputs let the interface be exercised without model calls. They know
each built-in scenario's answer, which is exactly why they live here, apart from the real
agents: the review board (board.py) never imports this file unless mock mode is on, every
line it produces is prefixed "MOCK", and the app shows a "Mock answers" badge. Tools and
MuJoCo runs stay real even in mock mode.
"""

from .agent import Episode

def _step(thought, tool, **kw):
    base = {"level": "warn_error", "t_from_ms": 0, "t_to_ms": 6000, "signals": [], "step_ms": 100, "times_ms": [],
            "prompt": "", "relative_to": ""}
    return base | kw | {"thought": "MOCK: " + thought, "tool": tool}


def mock_script(ep: Episode) -> dict:
    """Scripted outputs so the board can be tested without model calls. Tools and MuJoCo runs
    are real. Every line is prefixed MOCK and the UI shows a 'Mock answers' badge."""
    sc = ep.meta.get("scenario", "thermal")
    errs = [l for l in ep.log if l["level"] == "ERROR"]
    t_err = errs[0]["t_ms"] if errs else 3000
    keys = list(ep.patchable)
    d = {k: ep.patchable[k]["default"] for k in keys}

    fail = next((l["t_ms"] for l in ep.log if l["src"] == "supervisor" and l["level"] == "ERROR"), t_err)

    def hyp(claim, conf, patch, ll=-1, t=None):
        return {"claim": "MOCK: " + claim, "confidence": conf, "prediction": "MOCK: prediction.",
                "evidence": [{"text": "MOCK: evidence.", "time_ms": t_err if t is None else t, "log_line": ll}],
                "sim_patch": d | patch | {"why": "MOCK"}}

    steps = {
        "hardware": [_step("Checking driver warnings first.", "read_log"),
                     _step("Looking at driver temperature and force.", "read_telemetry",
                           signals=["finger_L_temp_C", "finger_L_cmd_force_N"], t_from_ms=1500, t_to_ms=t_err + 500, step_ms=50),
                     _step("Done.", "finish")],
        "grasp": [_step("Measuring where the part sits relative to the fingers.", "segment",
                        prompt="part", relative_to="gripper", times_ms=[max(0, t_err - 400), t_err, t_err + 250]),
                  _step("Reading the contact forces through the lift.", "read_telemetry",
                        signals=["contact_L_normal_N", "contact_R_normal_N", "cube_slip_mm"],
                        t_from_ms=2000, t_to_ms=t_err + 500, step_ms=50),
                  _step("Looking at the object as it slips.", "view_frames", times_ms=[t_err - 200, t_err, t_err + 300]),
                  _step("Done.", "finish")],
        "environment": [_step("Checking the planner and controller lines.", "read_log", level="all", t_from_ms=0, t_to_ms=t_err + 400),
                        _step("Checking for contact with anything else.", "read_telemetry",
                              signals=["obstacle_contact_N", "gripper_x"], t_from_ms=2500, t_to_ms=t_err + 500, step_ms=50),
                        _step("Done.", "finish")],
    }
    S = {
        "thermal": {"hyps": {"hardware": hyp("left driver overheated and derated", 0.8, {"hold_force_N": 8}),
                             "grasp": hyp("grip force was too low for the cube", 0.4, {"hold_force_N": 12}),
                             "environment": hyp("transport was too fast", 0.3, {"speed_scale": 0.8}, t=fail + 700)},
                    "skeptic": {"hardware": "needs_test", "grasp": "refuted", "environment": "needs_test"},
                    "rev": {"hardware": {"hold_force_N": 6}}, "win": "hardware"},
        "slip": {"hyps": {"hardware": hyp("a finger driver lost force", 0.2, {"hold_force_N": 8}),
                          "grasp": hyp("the steel block is more slippery than the grip assumes", 0.8, {"hold_force_N": 5.2}),
                          "environment": hyp("the lift accelerated too hard", 0.5, {"speed_scale": 0.75})},
                 "skeptic": {"hardware": "refuted", "grasp": "plausible", "environment": "needs_test"},
                 "rev": {"grasp": {"hold_force_N": 9}}, "win": "grasp"},
        "collision": {"hyps": {"hardware": hyp("the x-axis motor stalled", 0.4, {"speed_scale": 0.75}),
                               "grasp": hyp("the cube slipped from the grip", 0.3, {"hold_force_N": 12}),
                               "environment": hyp("an unmapped fixture blocks the transport path", 0.8,
                                                  {"transport_height_cm": 16})},
                      "skeptic": {"hardware": "needs_test", "grasp": "refuted", "environment": "plausible"},
                      "rev": {"environment": {"transport_height_cm": 18}}, "win": "environment"},
    }[sc]
    segs = [{"say": f"MOCK: {l['src']} reported {l['msg'][:50]}.", "time_ms": l["t_ms"], "objects": ["1", "2"],
             "signal": "none", "log_line": l["n"], "point": {"x": -1, "y": -1, "label": ""}} for l in errs[:3]]
    return {
        "steps": steps,
        "hypotheses": S["hyps"],
        "skeptic": {"reviews": [{"agent": k, "status": v, "challenge": f"MOCK: challenge to {k}."}
                                for k, v in S["skeptic"].items()], "note": "MOCK: two hypotheses go to the simulator."},
        "revisions": {aid: [{"thought": "MOCK: pushing the same setting further.", "sim_patch": d | S["rev"].get(aid, {}) | {"why": "MOCK"}}]
                      for aid in ("hardware", "grasp", "environment")},
        "answer": {"segments": segs, "takeaway": {
            "root_cause": "MOCK: root cause from the winning hypothesis.", "fixes": ["MOCK: fix one.", "MOCK: fix two."],
            "say": "MOCK takeaway.", "sim_patch": d | {"why": "MOCK"}}},
    }
