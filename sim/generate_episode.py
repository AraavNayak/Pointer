"""Generate deterministic pick-and-place failure episodes in MuJoCo.

A parallel-jaw gripper picks up an object and carries it toward a bin. Three
scenarios, each failing for a different reason:

  thermal    the left finger driver overheats under a 12 N grip and derates
             mid-transport; the cube is squeezed out of the weakened jaw
  slip       the part is an oily steel block (friction 0.25), but the grip
             profile assumes the catalog value (0.7) and uses a light 5 N grip
  collision  a fixture that is missing from the planner's map sits between the
             pick and the bin; the transport height is too low to clear it

Outputs (in episodes/<episode_id>/):
  video.mp4        640x480 @ 30 fps, H.264 (what the user watches, what SAM 3.1 sees)
  telemetry.csv    100 Hz sensor log
  controller.log   timestamped controller stdout
  gt_masks.json    per-frame ground-truth masks from MuJoCo's segmentation render
  meta.json        episode metadata + SAM point prompts for frame 0

Run:  python sim/generate_episode.py [out_root] --scenario thermal|slip|collision
      [--id ID] [--params '{"hold_force_N": 6, "speed_scale": 1, "transport_height_cm": 12.5}']
      (on Linux, set MUJOCO_GL=egl or osmesa)
"""

import json
import os
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

EPISODE_ID = "pick_place_drop_001"
FPS = 30
W, H = 640, 480
DURATION_S = 6.0
TELEMETRY_HZ = 100

XML = """
<mujoco model="pointer_pick_place">
  <option timestep="0.002" gravity="0 0 -9.81" cone="elliptic" impratio="10"/>
  <visual>
    <global offwidth="640" offheight="480"/>
    <quality shadowsize="2048"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.95 0.96 0.98" rgb2="0.70 0.74 0.80" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.86 0.87 0.89" rgb2="0.80 0.81 0.84" width="512" height="512"/>
    <material name="table" texture="grid" texrepeat="6 6" reflectance="0.05"/>
    <material name="metal" rgba="0.55 0.58 0.62 1" specular="0.6" shininess="0.6"/>
    <material name="pad" rgba="0.18 0.19 0.21 1"/>
    <material name="cube" rgba="0.86 0.16 0.18 1" specular="0.3"/>
    <material name="bin" rgba="0.20 0.42 0.78 1"/>
  </asset>
  <worldbody>
    <light pos="0.3 -0.6 1.6" dir="-0.2 0.4 -1" diffuse="0.7 0.7 0.7" castshadow="true"/>
    <geom name="table" type="plane" size="1 1 0.01" material="table"/>
    <body name="look" pos="0.33 0.02 0.12"/>
    <camera name="front" pos="0.40 -0.60 0.40" mode="targetbody" target="look" fovy="52"/>

    <!-- target bin -->
    <body name="bin" pos="0.55 0.0 0">
      <geom type="box" size="0.09 0.09 0.004" pos="0 0 0.004" material="bin"/>
      <geom type="box" size="0.09 0.004 0.04" pos="0 0.09 0.04" material="bin"/>
      <geom type="box" size="0.09 0.004 0.04" pos="0 -0.09 0.04" material="bin"/>
      <geom type="box" size="0.004 0.09 0.04" pos="0.09 0 0.04" material="bin"/>
      <geom type="box" size="0.004 0.09 0.04" pos="-0.09 0 0.04" material="bin"/>
    </body>

    <body name="cube" pos="0.15 0 0.025">
      <freejoint name="cube_free"/>
      <geom name="cube" type="box" size="0.025 0.025 0.025" mass="0.25" material="cube" friction="0.7 0.01 0.001"/>
    </body>

    <body name="gripper" pos="0 0 0.30" gravcomp="1">
      <joint name="gx" type="slide" axis="1 0 0" range="-0.2 0.8" damping="40"/>
      <joint name="gy" type="slide" axis="0 1 0" range="-0.4 0.4" damping="40"/>
      <joint name="gz" type="slide" axis="0 0 1" range="-0.29 0.2" damping="40"/>
      <geom name="mast" type="cylinder" size="0.012 0.15" pos="0 0 0.18" material="metal" contype="0" conaffinity="0"/>
      <geom name="palm" type="box" size="0.03 0.07 0.012" material="metal" mass="0.4"/>
      <body name="finger_L" pos="0 0.05 -0.05" gravcomp="1">
        <joint name="fL" type="slide" axis="0 -1 0" range="-0.035 0.05" damping="2"/>
        <geom name="finger_L" type="box" size="0.012 0.006 0.045" material="metal" mass="0.05"/>
        <geom name="pad_L" type="box" size="0.011 0.002 0.022" pos="0 -0.007 -0.02" material="pad" friction="0.7 0.01 0.001" mass="0.01"/>
      </body>
      <body name="finger_R" pos="0 -0.05 -0.05" gravcomp="1">
        <joint name="fR" type="slide" axis="0 1 0" range="0 0.05" damping="2"/>
        <geom name="finger_R" type="box" size="0.012 0.006 0.045" material="metal" mass="0.05"/>
        <geom name="pad_R" type="box" size="0.011 0.002 0.022" pos="0 0.007 -0.02" material="pad" friction="0.7 0.01 0.001" mass="0.01"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="ax" joint="gx" kp="2500" forcerange="-200 200"/>
    <position name="ay" joint="gy" kp="2500" forcerange="-200 200"/>
    <position name="az" joint="gz" kp="2500" forcerange="-200 200"/>
    <position name="aL" joint="fL" kp="600" forcerange="-12 12"/>
    <position name="aR" joint="fR" kp="600" forcerange="-12 12"/>
  </actuator>
</mujoco>
"""

SCENARIOS = {
    "thermal": {
        "episode_id": "pick_place_drop_001",
        "title": "Gripper drops the cube mid-transport",
        "object": "red cube",
        "grip_force_N": 12.0,
        "heat_gain": None,          # None = module default (2.5)
    },
    "slip": {
        "episode_id": "pick_place_slip_002",
        "title": "Steel block slides out during the lift",
        "object": "steel block",
        "grip_force_N": 5.1,
        "heat_gain": 0.9,
        "xml": [
            ('<material name="cube" rgba="0.86 0.16 0.18 1" specular="0.3"/>',
             '<material name="cube" rgba="0.70 0.72 0.76 1" specular="0.9" shininess="0.9" reflectance="0.25"/>'),
            ('mass="0.25" material="cube" friction="0.7 0.01 0.001"/>',
             'mass="0.25" material="cube" friction="0.25 0.005 0.0005" priority="1"/>'),
        ],
        "log": [
            ("INFO", "controller", "grip profile=eco  grip_force=5.1N  (object_db: block_A mu=0.70 mass=0.25kg)"),
            ("INFO", "perception", "object detected  class=block_A  surface=metal  sheen=high"),
        ],
    },
    "collision": {
        "episode_id": "pick_place_collision_003",
        "title": "Cube is knocked off by a fixture during transport",
        "object": "red cube",
        "grip_force_N": 12.0,
        "heat_gain": 0.9,
        "xml": [('    <body name="cube" pos="0.15 0 0.025">', '    <body name="fixture" pos="0.36 0.0 0.0">\n      <geom name="fixture" type="box" size="0.035 0.13 0.08" pos="0 0 0.08" rgba="0.42 0.47 0.40 1"/>\n    </body>\n\n    <body name="cube" pos="0.15 0 0.025">')],
        "log": [
            ("INFO", "planner", "workspace map v12 (2026-10-21)  obstacles between pick and place: none"),
            ("INFO", "planner", "transport height 12.5cm (object bottom)  clearance 5.0cm over mapped geometry"),
        ],
    },
}

# Waypoints for the gripper base (relative to its home pose) and finger command.
# (time_s, x, y, z_offset, finger_cmd)   finger_cmd: 0 = open, 0.05 = fully closed
BASE_WAYPOINTS = [
    (0.00, 0.00, 0.0, 0.00, 0.0),
    (1.00, 0.15, 0.0, 0.00, 0.0),   # move above cube
    (1.80, 0.15, 0.0, -0.20, 0.0),  # descend
    (2.20, 0.15, 0.0, -0.20, 0.05), # close
    (2.50, 0.15, 0.0, -0.20, 0.05), # settle grasp
    (3.20, 0.15, 0.0, -0.08, 0.05), # lift
    (4.60, 0.55, 0.0, -0.08, 0.05), # transport toward bin
    (5.20, 0.55, 0.0, -0.14, 0.05), # lower into bin
    (6.00, 0.55, 0.0, -0.14, 0.0),  # release
]

GRASP_DONE_T = 2.50     # waypoints after this are scaled by speed_scale

# Parameters a proposed fix may change, with their defaults and allowed ranges.
# Settings a proposed fix may change: (allowed min, allowed max, description).
# Defaults come from the scenario.
PATCHABLE = {
    "hold_force_N": (2.0, 12.0, "grip force each finger applies after the grasp is confirmed"),
    "speed_scale": (0.75, 1.5, "speed multiplier for lift, transport and place (>1 is faster)"),
    "transport_height_cm": (8.0, 20.0, "height of the object's bottom above the table while it is carried"),
}
DEFAULT_TRANSPORT_HEIGHT_CM = 12.5   # matches the -0.08 m lift offset below


def patch_defaults(scenario: str) -> dict:
    return {"hold_force_N": SCENARIOS[scenario]["grip_force_N"], "speed_scale": 1.0,
            "transport_height_cm": DEFAULT_TRANSPORT_HEIGHT_CM}


def build_waypoints(speed_scale=1.0, transport_height_cm=DEFAULT_TRANSPORT_HEIGHT_CM):
    wps = [(t if t <= GRASP_DONE_T else GRASP_DONE_T + (t - GRASP_DONE_T) / speed_scale, *rest)
           for t, *rest in BASE_WAYPOINTS]
    if transport_height_cm != DEFAULT_TRANSPORT_HEIGHT_CM:
        z = transport_height_cm / 100 - 0.205   # object bottom = gripper z - 0.095 m
        wps = [(t, x, y, z if zo == -0.08 else zo, f) for t, x, y, zo, f in wps]
    return wps


WAYPOINTS = build_waypoints()
BIN_CENTER, BIN_HALF = (0.55, 0.0), 0.085

TEMP_SOFT_LIMIT = 70.0  # C, left finger driver warns
TEMP_HARD_LIMIT = 80.0  # C, driver derates its torque limit
DERATED_FORCE = 0.6     # N, from 12 N
HEAT_GAIN = float(os.environ.get("HEAT_GAIN", "2.5"))  # C/s per N of finger force
COOLING = 0.3                                            # 1/s toward ambient
AMBIENT_C = 41.2


def smoothstep(a):
    return a * a * (3 - 2 * a)


def target_at(t):
    for (t0, *p0), (t1, *p1) in zip(WAYPOINTS, WAYPOINTS[1:]):
        if t0 <= t <= t1:
            a = smoothstep((t - t0) / (t1 - t0))
            return [v0 + (v1 - v0) * a for v0, v1 in zip(p0, p1)]
    return WAYPOINTS[-1][1:]


def phase_at(t):
    names = ["approach", "descend", "close", "settle", "lift", "transport", "place", "release"]
    for (t0, *_), (t1, *_), n in zip(WAYPOINTS, WAYPOINTS[1:], names):
        if t0 <= t < t1:
            return n
    return "release"


def contact_normals(m, d, cube_gid, finger_geoms):
    """Sum of normal contact force between the cube and each finger (N)."""
    out = {k: 0.0 for k in finger_geoms}
    f6 = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        pair = {c.geom1, c.geom2}
        if cube_gid not in pair:
            continue
        for k, gids in finger_geoms.items():
            if pair & gids:
                mujoco.mj_contactForce(m, d, i, f6)
                out[k] += abs(f6[0])
    return out


def apply_hidden(scenario: str, hidden: dict | None) -> dict:
    """Scenario config with hidden (mystery) parameters applied. Without `hidden`, unchanged."""
    sc = dict(SCENARIOS[scenario])
    if not hidden:
        return sc
    sc["xml"] = list(sc.get("xml", []))
    sc["log"] = list(sc.get("log", []))
    if "heat_gain" in hidden:
        sc["heat_gain"] = hidden["heat_gain"]
    if "grip_force_N" in hidden:
        sc["grip_force_N"] = hidden["grip_force_N"]
        sc["log"] = [(lv, src, msg.replace("grip_force=5.1N", f"grip_force={hidden['grip_force_N']:.1f}N"))
                     for lv, src, msg in sc["log"]]
    if "friction" in hidden:
        sc["xml"] = [(a, b.replace('friction="0.25 0.005 0.0005"', f'friction="{hidden["friction"]:.3f} 0.005 0.0005"'))
                     for a, b in sc["xml"]]
    if "fixture" in hidden:
        f = hidden["fixture"]
        geom = (f'    <body name="fixture" pos="{f["x"]:.3f} 0.0 0.0">\n'
                f'      <geom name="fixture" type="box" size="{f["half_x"]:.3f} {f["half_y"]:.3f} {f["top"] / 2:.3f}" '
                f'pos="0 0 {f["top"] / 2:.3f}" rgba="{f["rgba"]}"/>\n    </body>\n\n'
                '    <body name="cube" pos="0.15 0 0.025">')
        sc["xml"] = [('    <body name="cube" pos="0.15 0 0.025">', geom)]
    if "rgba" in hidden and scenario != "slip":
        sc["xml"].append(('<material name="cube" rgba="0.86 0.16 0.18 1" specular="0.3"/>',
                          f'<material name="cube" rgba="{hidden["rgba"]}" specular="0.3"/>'))
    if "object" in hidden:
        sc["object"] = hidden["object"]
    if "title" in hidden:
        sc["title"] = hidden["title"]
    if hidden.get("uniform_log"):
        # every mystery opens with the same boilerplate, so the log's first lines don't give the fault away
        sc["log"] = [
            ("INFO", "controller", f"grip profile=standard  grip_force={sc['grip_force_N']:.1f}N  (object_db: part mu=0.70 mass=0.25kg)"),
            ("INFO", "perception", "object detected  class=part"),
            ("INFO", "planner", "workspace map v12 (2026-10-21)  obstacles between pick and place: none"),
            ("INFO", "planner", "transport height 12.5cm (object bottom)  clearance 5.0cm over mapped geometry"),
        ]
    sc["temp_soft"] = hidden.get("temp_soft", TEMP_SOFT_LIMIT)
    sc["temp_hard"] = hidden.get("temp_hard", TEMP_HARD_LIMIT)
    sc["log"] += [tuple(h[1:]) + (h[0],) for h in hidden.get("decoys", [])]
    return sc


def main(out_root: Path, episode_id: str | None = None, params: dict | None = None,
         scenario: str = "thermal", hidden: dict | None = None):
    global WAYPOINTS
    sc = apply_hidden(scenario, hidden)
    episode_id = episode_id or sc["episode_id"]
    defaults = patch_defaults(scenario) | {"hold_force_N": sc["grip_force_N"]}
    temp_soft = sc.get("temp_soft", TEMP_SOFT_LIMIT)
    temp_hard = sc.get("temp_hard", TEMP_HARD_LIMIT)
    params = {k: float(np.clip(float((params or {}).get(k, defaults[k])), lo, hi))
              for k, (lo, hi, _) in PATCHABLE.items()}
    patched = any(params[k] != defaults[k] for k in PATCHABLE)
    WAYPOINTS = build_waypoints(params["speed_scale"], params["transport_height_cm"])
    duration_s = max(DURATION_S, WAYPOINTS[-1][0] + 0.3) if patched else DURATION_S
    heat_gain = sc["heat_gain"] if sc["heat_gain"] is not None else HEAT_GAIN
    xml = XML
    for a, b in sc.get("xml", []):
        assert a in xml, a
        xml = xml.replace(a, b)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
    aid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    cube_gid = gid("cube")
    fingers = {"L": {gid("finger_L"), gid("pad_L")}, "R": {gid("finger_R"), gid("pad_R")}}
    gripper_geoms = {gid(n) for n in ["palm", "mast", "finger_L", "pad_L", "finger_R", "pad_R"]}
    fixture_gid = gid("fixture")  # -1 when the scenario has no fixture
    if sc["grip_force_N"] != 12.0:
        for a in ("aL", "aR"):
            m.actuator_forcerange[aid(a)] = [-sc["grip_force_N"], sc["grip_force_N"]]
    cube_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")
    grip_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    aL, aR = aid("aL"), aid("aR")
    jL = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "fL")]
    jR = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "fR")]

    renderer = mujoco.Renderer(m, H, W)
    seg_renderer = mujoco.Renderer(m, H, W)
    seg_renderer.enable_segmentation_rendering()
    cam = "front"

    ep_dir = out_root / episode_id
    ep_dir.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(ep_dir / "video.mp4", fps=FPS, codec="libx264",
                                quality=8, pixelformat="yuv420p", macro_block_size=8)

    log_lines = []

    def log(t, level, src, msg):
        log_lines.append(f"[{int(round(t*1000)):>6}ms] {level:<5} {src:<14} {msg}")

    telemetry = []
    gt_masks = {"source": "mujoco_segmentation", "width": W, "height": H, "fps": FPS,
                "objects": {"1": sc["object"], "2": "gripper"}, "frames": {}}

    n_steps = int(duration_s / m.opt.timestep)
    steps_per_frame = int(round(1 / FPS / m.opt.timestep))
    steps_per_tel = int(round(1 / TELEMETRY_HZ / m.opt.timestep))
    frame_idx = 0
    last_phase = None
    warned = derated = slip_logged = drop_logged = False
    holding_logged = False
    last_hb = -1

    log(0, "INFO", "supervisor", f"episode {episode_id} start  task=pick_place  object={sc['object'].replace(' ', '_')}  target=bin_A")
    if patched:
        log(0, "INFO", "controller", "patched config  " + "  ".join(f"{k}={v:g}" for k, v in params.items()))
    log(0, "INFO", "finger_L", "driver online  fw=2.4.1  force_limit=12.0N  temp=41.2C")
    log(0, "INFO", "finger_R", "driver online  fw=2.4.1  force_limit=12.0N  temp=39.8C")
    log(0, "INFO", "grasp_mon", "monitor armed  slip_thresh=8mm  min_normal=1.0N")
    for entry in sc.get("log", []):
        level, src, msg = entry[:3]
        log(entry[3] if len(entry) > 3 else 0, level, src, msg)

    temp_L = AMBIENT_C
    grasp_rel = None
    derate_t = None
    track_logged = False
    hit_t = None
    slip_t = None
    for step in range(n_steps + 1):
        t = step * m.opt.timestep
        x, y, z, fcmd = target_at(t)
        d.ctrl[0], d.ctrl[1], d.ctrl[2] = x, y, z
        d.ctrl[aL] = fcmd
        d.ctrl[aR] = fcmd

        # thermal model for the left finger driver (simple first-order rise under load)
        temp_L += (heat_gain * abs(d.actuator_force[aL]) - COOLING * (temp_L - AMBIENT_C)) * m.opt.timestep
        # --- injected fault ---
        if temp_L >= temp_soft and not warned:
            warned = True
            log(t, "WARN", "finger_L", f"driver temp {temp_L:.1f}C exceeds soft limit {temp_soft:.1f}C")
        if temp_L >= temp_hard and not derated:
            derated = True
            derate_t = t
            prev_limit = m.actuator_forcerange[aL][1]
            m.actuator_forcerange[aL] = [-DERATED_FORCE, DERATED_FORCE]
            log(t, "ERROR", "finger_L", f"thermal limit {temp_hard:.0f}C reached at |tau|={abs(d.actuator_force[aL]):.2f}N; torque limit derated {prev_limit:.2f}N -> {DERATED_FORCE:.2f}N")
            log(t, "WARN", "controller", "finger_L saturated; continuing trajectory (no grasp re-plan configured)")

        phase = phase_at(t)
        if phase != last_phase:
            log(t, "INFO", "controller", f"phase -> {phase}")
            last_phase = phase

        mujoco.mj_step(m, d)

        fx = 0.0
        if fixture_gid >= 0:
            f6 = np.zeros(6)
            for i in range(d.ncon):
                c = d.contact[i]
                if fixture_gid in (c.geom1, c.geom2):
                    other = c.geom2 if c.geom1 == fixture_gid else c.geom1
                    if other == cube_gid or other in gripper_geoms:
                        mujoco.mj_contactForce(m, d, i, f6)
                        fx += abs(f6[0])
            if fx > 1.0 and hit_t is None:
                hit_t = t
                log(t, "WARN", "controller", f"x-axis effort spike  {fx:.1f}N contact on carried object")
            if hit_t is not None and not track_logged and t > hit_t + 0.12:
                track_logged = True
                err_mm = abs(d.ctrl[0] - d.qpos[m.jnt_qposadr[0]]) * 1000
                log(t, "ERROR", "controller", f"x-axis tracking error {err_mm:.0f}mm exceeds 20mm  (unexpected resistance)")

        cube_pos = d.xpos[cube_bid].copy()
        grip_pos = d.xpos[grip_bid].copy()
        pad_mid_z = grip_pos[2] - 0.07
        rel = cube_pos - np.array([grip_pos[0], grip_pos[1], pad_mid_z])
        slip_mm = float(np.linalg.norm(rel - grasp_rel) * 1000) if grasp_rel is not None else 0.0

        if step % steps_per_tel == 0:
            cn = contact_normals(m, d, cube_gid, fingers)
            telemetry.append({
                "timestamp_ms": int(round(t * 1000)),
                "gripper_x": round(float(grip_pos[0]), 4),
                "gripper_z": round(float(grip_pos[2]), 4),
                "finger_L_cmd_force_N": round(float(d.actuator_force[aL]), 3),
                "finger_R_cmd_force_N": round(float(d.actuator_force[aR]), 3),
                "contact_L_normal_N": round(cn["L"], 3),
                "contact_R_normal_N": round(cn["R"], 3),
                "finger_L_pos_mm": round(float(d.qpos[jL]) * 1000, 2),
                "finger_R_pos_mm": round(float(d.qpos[jR]) * 1000, 2),
                "finger_L_temp_C": round(temp_L, 2),
                "cube_z": round(float(cube_pos[2]), 4),
                "cube_slip_mm": round(slip_mm, 2),
                "obstacle_contact_N": round(fx, 2),
            })
            in_hand_phase = phase in ("lift", "transport", "place")
            if t > 2.2 and not holding_logged and cn["L"] > 1 and cn["R"] > 1:
                holding_logged = True
                grasp_rel = rel.copy()
                log(t, "INFO", "grasp_mon", f"grasp confirmed  N_L={cn['L']:.1f}N  N_R={cn['R']:.1f}N")
                if params["hold_force_N"] != sc["grip_force_N"]:
                    hf = params["hold_force_N"]
                    m.actuator_forcerange[aL] = [-hf, hf]
                    m.actuator_forcerange[aR] = [-hf, hf]
                    log(t, "INFO", "controller", f"hold force -> {hf:.1f}N for transport")
            if in_hand_phase and holding_logged and not slip_logged and slip_mm > 8:
                slip_logged = True
                slip_t = t
                log(t, "ERROR", "grasp_mon", f"object slip detected  slip={slip_mm:.1f}mm  N_L={cn['L']:.2f}N (min 1.0N)  N_R={cn['R']:.1f}N")
            if slip_logged and not drop_logged and cube_pos[2] < 0.03 and cube_pos[0] < 0.46:
                drop_logged = True
                log(t, "ERROR", "supervisor", f"object dropped  cube=({cube_pos[0]:.3f},{cube_pos[1]:.3f},{cube_pos[2]:.3f})  outside bin_A")
            hb = int(t * 2)  # heartbeat every 0.5 s
            if hb != last_hb:
                last_hb = hb
                log(t, "DEBUG", "finger_L", f"tau={d.actuator_force[aL]:6.2f}N  pos={d.qpos[jL]*1000:5.1f}mm  temp={temp_L:5.1f}C")
                log(t, "DEBUG", "finger_R", f"tau={d.actuator_force[aR]:6.2f}N  pos={d.qpos[jR]*1000:5.1f}mm")

        if t + 1e-9 >= frame_idx / FPS and frame_idx < int(round(duration_s * FPS)):
            renderer.update_scene(d, camera=cam)
            rgb = renderer.render()
            writer.append_data(rgb)
            seg_renderer.update_scene(d, camera=cam)
            seg = seg_renderer.render()  # (H, W, 2): [objid, objtype]
            is_geom = seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
            frame_masks = {}
            for oid, gset in (("1", {cube_gid}), ("2", gripper_geoms)):
                mask = (is_geom & np.isin(seg[..., 0], list(gset))).astype(np.uint8)
                frame_masks[oid] = mask_to_record(mask)
            gt_masks["frames"][str(frame_idx)] = frame_masks
            frame_idx += 1

    writer.close()
    in_bin = (abs(cube_pos[0] - BIN_CENTER[0]) < BIN_HALF and abs(cube_pos[1] - BIN_CENTER[1]) < BIN_HALF
              and cube_pos[2] < 0.08)
    if drop_logged:
        outcome = "FAIL: object dropped"
    elif in_bin:
        outcome = "OK: placed in bin"
        log(duration_s, "INFO", "supervisor", f"object placed in bin_A  cube=({cube_pos[0]:.3f},{cube_pos[1]:.3f},{cube_pos[2]:.3f})")
    else:
        outcome = "FAIL: object not in bin"
        log(duration_s, "ERROR", "supervisor", f"object not in bin_A  cube=({cube_pos[0]:.3f},{cube_pos[1]:.3f},{cube_pos[2]:.3f})")
    log(duration_s, "INFO", "supervisor", f"episode end  result={'OK' if outcome.startswith('OK') else 'FAIL'}")
    log_lines.sort(key=lambda s: int(s[1:7]))

    (ep_dir / "controller.log").write_text("\n".join(log_lines) + "\n")
    import csv
    with open(ep_dir / "telemetry.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(telemetry[0].keys()))
        w.writeheader()
        w.writerows(telemetry)
    (ep_dir / "gt_masks.json").write_text(json.dumps(gt_masks))

    f0 = gt_masks["frames"]["0"]
    meta = {
        "episode_id": episode_id,
        "task": f"Pick the {sc['object']} and place it in the blue bin.",
        "scenario": scenario,
        "title": sc["title"],
        "fps": FPS, "width": W, "height": H,
        "duration_ms": int(round(duration_s * 1000)),
        "n_frames": frame_idx,
        "objects": {"1": sc["object"], "2": "gripper"},
        "sam_prompt_points": {k: v["centroid"] for k, v in f0.items() if v},
        "outcome": outcome,
        "params": params,
        **({"mystery": hidden} if hidden else {}),
        "patchable": {k: {"min": lo, "max": hi, "desc": desc, "default": defaults[k]}
                      for k, (lo, hi, desc) in PATCHABLE.items()},
        "slip_ms": int(round(slip_t * 1000)) if slip_t is not None else None,
        "first_contact_ms": int(round(hit_t * 1000)) if hit_t is not None else None,
        "max_obstacle_contact_N": round(max(r["obstacle_contact_N"] for r in telemetry), 1),
        "max_driver_temp_C": round(max(r["finger_L_temp_C"] for r in telemetry), 1),
        "driver_limit_C": round(temp_hard, 1),
        "driver_soft_C": round(temp_soft, 1),
        "derate_ms": int(round(derate_t * 1000)) if derate_t is not None else None,
    }
    (ep_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f"wrote {ep_dir}  ({len(log_lines)} log lines, {len(telemetry)} telemetry rows)")


def mask_to_record(mask):
    """Binary mask -> {bbox, centroid, polygons} in pixel coords (empty dict if absent)."""
    if mask.sum() < 4:
        return {}
    ys, xs = np.nonzero(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) < 6:
            continue
        c = cv2.approxPolyDP(c, 1.0, True)
        polys.append(c.reshape(-1, 2).tolist())
    return {
        "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "centroid": [int(xs.mean()), int(ys.mean())],
        "area": int(mask.sum()),
        "polygons": polys,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("out_root", nargs="?", default=str(Path(__file__).resolve().parent.parent / "episodes"))
    ap.add_argument("--id", default=None)
    ap.add_argument("--scenario", default="thermal", choices=list(SCENARIOS))
    ap.add_argument("--params", default="{}", help='JSON, e.g. {"hold_force_N": 6}')
    ap.add_argument("--hidden", default="", help="JSON of hidden mystery parameters (see sim/mystery.py)")
    a = ap.parse_args()
    main(Path(a.out_root), a.id, json.loads(a.params), a.scenario, json.loads(a.hidden) if a.hidden else None)
