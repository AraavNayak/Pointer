"""Generate a deterministic pick-and-place failure episode in MuJoCo.

A parallel-jaw gripper picks up a red cube and carries it toward a bin.
Mid-transport, the left finger actuator hits its thermal torque limit and is
derated (an injected fault). The right finger keeps squeezing, the cube is
pushed against the weakened left jaw, slips out and falls.

Outputs (in episodes/<episode_id>/):
  video.mp4        640x480 @ 30 fps, H.264 (what the user watches, what SAM 3.1 sees)
  telemetry.csv    100 Hz sensor log
  controller.log   timestamped controller stdout
  gt_masks.json    per-frame ground-truth masks from MuJoCo's segmentation render
  meta.json        episode metadata + SAM point prompts for frame 0

Run:  MUJOCO_GL=egl python sim/generate_episode.py      (Linux)
      python sim/generate_episode.py                     (macOS)
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

# Waypoints for the gripper base (relative to its home pose) and finger command.
# (time_s, x, y, z_offset, finger_cmd)   finger_cmd: 0 = open, 0.05 = fully closed
WAYPOINTS = [
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


def main(out_root: Path):
    m = mujoco.MjModel.from_xml_string(XML)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
    aid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    cube_gid = gid("cube")
    fingers = {"L": {gid("finger_L"), gid("pad_L")}, "R": {gid("finger_R"), gid("pad_R")}}
    gripper_geoms = {gid(n) for n in ["palm", "mast", "finger_L", "pad_L", "finger_R", "pad_R"]}
    cube_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")
    grip_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    aL, aR = aid("aL"), aid("aR")
    jL = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "fL")]
    jR = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "fR")]

    renderer = mujoco.Renderer(m, H, W)
    seg_renderer = mujoco.Renderer(m, H, W)
    seg_renderer.enable_segmentation_rendering()
    cam = "front"

    ep_dir = out_root / EPISODE_ID
    ep_dir.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(ep_dir / "video.mp4", fps=FPS, codec="libx264",
                                quality=8, pixelformat="yuv420p", macro_block_size=8)

    log_lines = []

    def log(t, level, src, msg):
        log_lines.append(f"[{int(round(t*1000)):>6}ms] {level:<5} {src:<14} {msg}")

    telemetry = []
    gt_masks = {"source": "mujoco_segmentation", "width": W, "height": H, "fps": FPS,
                "objects": {"1": "red cube", "2": "gripper"}, "frames": {}}

    n_steps = int(DURATION_S / m.opt.timestep)
    steps_per_frame = int(round(1 / FPS / m.opt.timestep))
    steps_per_tel = int(round(1 / TELEMETRY_HZ / m.opt.timestep))
    frame_idx = 0
    last_phase = None
    warned = derated = slip_logged = drop_logged = False
    holding_logged = False
    last_hb = -1

    log(0, "INFO", "supervisor", f"episode {EPISODE_ID} start  task=pick_place  object=red_cube  target=bin_A")
    log(0, "INFO", "finger_L", "driver online  fw=2.4.1  force_limit=12.0N  temp=41.2C")
    log(0, "INFO", "finger_R", "driver online  fw=2.4.1  force_limit=12.0N  temp=39.8C")
    log(0, "INFO", "grasp_mon", "monitor armed  slip_thresh=8mm  min_normal=1.0N")

    temp_L = AMBIENT_C
    grasp_rel = None
    for step in range(n_steps + 1):
        t = step * m.opt.timestep
        x, y, z, fcmd = target_at(t)
        d.ctrl[0], d.ctrl[1], d.ctrl[2] = x, y, z
        d.ctrl[aL] = fcmd
        d.ctrl[aR] = fcmd

        # thermal model for the left finger driver (simple first-order rise under load)
        temp_L += (HEAT_GAIN * abs(d.actuator_force[aL]) - COOLING * (temp_L - AMBIENT_C)) * m.opt.timestep
        # --- injected fault ---
        if temp_L >= TEMP_SOFT_LIMIT and not warned:
            warned = True
            log(t, "WARN", "finger_L", f"driver temp {temp_L:.1f}C exceeds soft limit {TEMP_SOFT_LIMIT:.1f}C")
        if temp_L >= TEMP_HARD_LIMIT and not derated:
            derated = True
            m.actuator_forcerange[aL] = [-DERATED_FORCE, DERATED_FORCE]
            log(t, "ERROR", "finger_L", f"thermal limit {TEMP_HARD_LIMIT:.0f}C reached at |tau|={abs(d.actuator_force[aL]):.2f}N; torque limit derated 12.00N -> {DERATED_FORCE:.2f}N")
            log(t, "WARN", "controller", "finger_L saturated; continuing trajectory (no grasp re-plan configured)")

        phase = phase_at(t)
        if phase != last_phase:
            log(t, "INFO", "controller", f"phase -> {phase}")
            last_phase = phase

        mujoco.mj_step(m, d)

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
            })
            in_hand_phase = phase in ("lift", "transport", "place")
            if t > 2.2 and not holding_logged and cn["L"] > 1 and cn["R"] > 1:
                holding_logged = True
                grasp_rel = rel.copy()
                log(t, "INFO", "grasp_mon", f"grasp confirmed  N_L={cn['L']:.1f}N  N_R={cn['R']:.1f}N")
            if in_hand_phase and holding_logged and not slip_logged and slip_mm > 8:
                slip_logged = True
                log(t, "ERROR", "grasp_mon", f"object slip detected  slip={slip_mm:.1f}mm  N_L={cn['L']:.2f}N (min 1.0N)  N_R={cn['R']:.1f}N")
            if slip_logged and not drop_logged and cube_pos[2] < 0.03 and cube_pos[0] < 0.46:
                drop_logged = True
                log(t, "ERROR", "supervisor", f"object dropped  cube=({cube_pos[0]:.3f},{cube_pos[1]:.3f},{cube_pos[2]:.3f})  outside bin_A")
            hb = int(t * 2)  # heartbeat every 0.5 s
            if hb != last_hb:
                last_hb = hb
                log(t, "DEBUG", "finger_L", f"tau={d.actuator_force[aL]:6.2f}N  pos={d.qpos[jL]*1000:5.1f}mm  temp={temp_L:5.1f}C")
                log(t, "DEBUG", "finger_R", f"tau={d.actuator_force[aR]:6.2f}N  pos={d.qpos[jR]*1000:5.1f}mm")

        if t + 1e-9 >= frame_idx / FPS and frame_idx < int(DURATION_S * FPS):
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
    log(DURATION_S, "INFO", "supervisor", f"episode end  result={'FAIL' if drop_logged else 'OK'}")
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
        "episode_id": EPISODE_ID,
        "task": "Pick the red cube and place it in the blue bin.",
        "fps": FPS, "width": W, "height": H,
        "duration_ms": int(DURATION_S * 1000),
        "n_frames": frame_idx,
        "objects": {"1": "red cube", "2": "gripper"},
        "sam_prompt_points": {k: v["centroid"] for k, v in f0.items() if v},
        "outcome": "FAIL: object dropped" if drop_logged else "OK",
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
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "episodes"
    main(root)
