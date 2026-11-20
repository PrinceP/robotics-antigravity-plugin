# /// script
# requires-python = ">=3.10"
# dependencies = ["mujoco>=3.2", "numpy", "opencv-python-headless"]
# ///
"""Bimanual handover in MuJoCo: the left arm picks a sponge from the desk, hands it to the right arm, the right arm places it.

    MUJOCO_GL=egl uv run scripts/xlerobot_handover_sim.py --out handover.mp4

Scene (measured on the real setup): desk top 0.75 m, front edge flush with the cart's top basket; sponge 10 x 6 x 2 cm lying with
its long side pointing at the robot, near edge 8 cm in from the desk edge, centre 10 cm left of the robot's centre line; it is
placed back the same distance in, 10 cm right of the centre line.

Why the grasps look the way they do: 11 cm into the desk the arms cannot point straight down (the wrist would need to be 0.31 m
from the shoulder; upper arm + forearm are 0.25 m), so the gripper tilts ~50 deg forward-down and is rolled so the jaws close
sideways across the sponge's 6 cm width. For the handover the left arm rolls back, which swings the sponge's free end sideways
toward the right arm; the right arm grabs that end, and rolls again to set the sponge down flat.

Pipeline (the same steps run on the real robot):
1. the head tilts down; a yellow-blob detector finds the sponge in head_camera; the pixel is cast onto the desk plane -> xy
2. inverse kinematics (damped least squares on the 5 arm joints) for each pose: tool point, gripper pointing direction, and
   jaw closing direction
3. pick (left) -> handover (left presents, right grabs the free end, left releases) -> place (right)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import mujoco
import numpy as np

SCENE = Path(__file__).resolve().parent.parent / "assets" / "makermods_xlerobot" / "scene.xml"
DESK_Z, DESK_FRONT_X = 0.75, 0.182          # desk top height; front edge (flush with the top basket's front, x 0.182)
SPONGE = np.array([0.10, 0.06, 0.02])        # length (along x, pointing at the robot), width (along y), height
SPONGE_XY = np.array([DESK_FRONT_X + 0.08 + SPONGE[0] / 2, 0.10])   # true centre (perception must find it)
PLACE_XY = np.array([SPONGE_XY[0], -0.10])
ARM_JOINTS = {"right": ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"],
              "left": ["Rotation_2", "Pitch_2", "Elbow_2", "Wrist_Pitch_2", "Wrist_Roll_2"]}
JAW = {"right": "Jaw", "left": "Jaw_2"}
TIPS = {"right": ("Fixed_Jaw_tip", "Moving_Jaw_tip"), "left": ("Fixed_Jaw_tip_2", "Moving_Jaw_tip_2")}
ARM_ACT = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
FOLDED = np.array([0.0, 2.6, 2.6, 0.0, 0.0])  # arms raised and folded out of the way
JAW_OPEN, JAW_CLOSED = 1.6, 0.2               # rad: ~75 mm between the pads open; close target just past contact (~0.35 rad on the
                                              # 6 cm sponge), so the load-limited gripper force holds it
TCP_BACK = 0.015                              # tool point: this far back from the jaw tips
# The object centre sits this far from the fixed jaw tip toward the moving jaw: 6 mm from the tip frame to the pad's inner face,
# 5 mm clearance, half the sponge width. (Measured from the jaw pads at the grasp pose.)
GRASP_HALF = 0.006 + 0.005 + SPONGE[1] / 2
TILT = 60.0                                   # gripper tilt below horizontal for desk grasps (deg): the steepest reachable here
EDGE_IN = 0.025                               # grip this far in from the sponge's ends


def tilted(deg: float, heading: np.ndarray = np.array([1.0, 0, 0])) -> np.ndarray:
  h = heading / np.linalg.norm(heading)
  return np.array([h[0] * np.cos(np.radians(deg)), h[1] * np.cos(np.radians(deg)), -np.sin(np.radians(deg))])


def build() -> tuple[mujoco.MjModel, mujoco.MjData]:
  spec = mujoco.MjSpec.from_file(str(SCENE))
  desk = spec.worldbody.add_body(name="desk", pos=[DESK_FRONT_X + 0.25, 0, DESK_Z - 0.015])
  desk.add_geom(name="desk_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.25, 0.55, 0.015], rgba=[0.45, 0.28, 0.15, 1],
                friction=[0.8, 0.01, 0.001])
  sp = spec.worldbody.add_body(name="sponge", pos=[SPONGE_XY[0], SPONGE_XY[1], DESK_Z + SPONGE[2] / 2 + 0.001])
  sp.add_freejoint()
  # soft, grippy sponge against soft TPU jaws: high friction, compliant contact
  sp.add_geom(name="sponge", type=mujoco.mjtGeom.mjGEOM_BOX, size=SPONGE / 2, rgba=[0.95, 0.85, 0.15, 1], mass=0.015,
              friction=[1.6, 0.02, 0.002], condim=4, solref=[0.02, 1], solimp=[0.9, 0.95, 0.005, 0.5, 2])
  model = spec.compile()
  return model, mujoco.MjData(model)


class Arms:
  """Kinematics on a scratch MjData, so IK never disturbs the simulation."""

  def __init__(self, model: mujoco.MjModel):
    self.m, self.d = model, mujoco.MjData(model)
    self.qadr = {a: [model.jnt_qposadr[model.joint(j).id] for j in js] for a, js in ARM_JOINTS.items()}
    self.dadr = {a: [model.jnt_dofadr[model.joint(j).id] for j in js] for a, js in ARM_JOINTS.items()}
    self.rng = {a: np.array([model.jnt_range[model.joint(j).id] for j in js]) for a, js in ARM_JOINTS.items()}

  def tool(self, arm: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """tool point, gripper pointing axis, closing direction (fixed jaw -> moving jaw), body carrying the tool point"""
    m, d = self.m, self.d
    f, mv = (d.body(t).xpos.copy() for t in TIPS[arm])
    roll = m.joint(ARM_JOINTS[arm][4]).id
    axis = d.xaxis[roll] / np.linalg.norm(d.xaxis[roll])
    hinge = d.xaxis[m.joint(JAW[arm]).id].copy()
    n = np.cross(axis, hinge); n /= np.linalg.norm(n)
    if (mv - f) @ n < 0:
      n = -n
    # anchored on the fixed jaw: the moving jaw pushes the object against it, so the object centre ends up GRASP_HALF from it
    return f + GRASP_HALF * n - TCP_BACK * axis, axis, n, m.body(TIPS[arm][0]).id

  def solve(self, arm: str, q_full: np.ndarray, target: np.ndarray, axis_des: np.ndarray, close_des: np.ndarray,
            w_axis: float = 0.15, w_close: float = 0.08, iters: int = 600) -> tuple[np.ndarray, float, float, float]:
    """Damped least squares on the 5 arm joints: tool point -> target, gripper axis -> axis_des, closing direction -> +-close_des."""
    m, d = self.m, self.d
    d.qpos[:] = q_full
    qa, da = self.qadr[arm], self.dadr[arm]
    axis_des = axis_des / np.linalg.norm(axis_des)
    close_des = close_des / np.linalg.norm(close_des)
    for _ in range(iters):
      mujoco.mj_forward(m, d)
      p, ax, n, bid = self.tool(arm)
      s = 1.0 if n @ close_des >= 0 else -1.0       # either jaw may take either side
      err = np.concatenate([target - p, w_axis * (axis_des - ax), w_close * (s * close_des - n)])
      if np.linalg.norm(err) < 1e-4:
        break
      jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
      mujoco.mj_jac(m, d, jp, jr, p, bid)
      wr = jr[:, da].T                                # (5, 3): gripper angular velocity per arm joint
      j = np.vstack([jp[:, da], w_axis * np.cross(wr, ax).T, w_close * np.cross(wr, n).T])
      dq = j.T @ np.linalg.solve(j @ j.T + 2e-4 * np.eye(len(err)), err)
      d.qpos[qa] = np.clip(d.qpos[qa] + 0.6 * dq, self.rng[arm][:, 0], self.rng[arm][:, 1])
    mujoco.mj_forward(m, d)
    p, ax, n, _ = self.tool(arm)
    return (d.qpos[qa].copy(), float(np.linalg.norm(target - p)), float(np.degrees(np.arccos(np.clip(ax @ axis_des, -1, 1)))),
            float(np.degrees(np.arccos(np.clip(abs(n @ close_des), -1, 1)))))


def find_sponge(model: mujoco.MjModel, data: mujoco.MjData, renderer: mujoco.Renderer) -> np.ndarray:
  """Yellow blob in head_camera -> ray through its centre -> sponge top plane. Returns sponge xy in world coordinates."""
  renderer.update_scene(data, "head_camera")
  rgb = renderer.render()
  mask = cv2.inRange(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV), (20, 120, 120), (35, 255, 255))
  ys, xs = np.nonzero(mask)
  if len(xs) < 20:
    raise RuntimeError("sponge not visible in head_camera")
  cam = model.camera("head_camera").id
  h, w = rgb.shape[:2]
  f = 0.5 * h / np.tan(np.radians(model.cam_fovy[cam]) / 2)
  R, o = data.cam_xmat[cam].reshape(3, 3), data.cam_xpos[cam]

  def on_plane(u: float, v: float, z: float) -> np.ndarray:
    ray = R @ np.array([(u - w / 2) / f, -(v - h / 2) / f, -1.0])
    return o + (z - o[2]) / ray[2] * ray

  # Looking down from behind, the blob shows the top face and the near side face. Its far edge (top of the blob in the image)
  # is the top face's far edge; its near edge (bottom of the blob) is where the near side meets the desk.
  v_far, v_near = np.percentile(ys, 2), np.percentile(ys, 98)
  u_mid = xs.mean()
  far = on_plane(u_mid, v_far, DESK_Z + SPONGE[2])
  near = on_plane(u_mid, v_near, DESK_Z)
  return 0.5 * (far + near)[:2]


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", default="handover.mp4")
  ap.add_argument("--fps", type=int, default=30)
  ap.add_argument("--truth", action="store_true", help="use the true sponge position instead of the head-camera estimate")
  a = ap.parse_args()
  model, data = build()
  arms = Arms(model)
  act = {(arm, n): model.actuator(f"{arm}_arm_{n}").id for arm in ARM_JOINTS for n in ARM_ACT + ["gripper"]}
  w, h = 480, 360
  renderer = mujoco.Renderer(model, h, w)
  writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (2 * w, 2 * h))
  overview = mujoco.MjvCamera()
  overview.lookat[:] = [0.2, 0, 0.82]; overview.distance, overview.azimuth, overview.elevation = 1.15, 150, -30
  spf = int(round(1 / (a.fps * model.opt.timestep)))
  log: list[str] = []

  def frame() -> None:
    tiles = []
    for cam in (overview, "head_camera", "wrist_camera_left", "wrist_camera_right"):
      renderer.update_scene(data, cam)
      img = renderer.render().copy()
      cv2.putText(img, cam if isinstance(cam, str) else "overview", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
      tiles.append(img)
    writer.write(cv2.cvtColor(np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]), cv2.COLOR_RGB2BGR))

  def run(seconds: float) -> None:
    for _ in range(int(seconds * a.fps)):
      for _ in range(spf):
        mujoco.mj_step(model, data)
      frame()

  def move(arm: str, q_goal: np.ndarray, seconds: float) -> None:
    q0 = np.array([data.ctrl[act[(arm, k)]] for k in ARM_ACT])
    steps = int(seconds * a.fps)
    for i in range(1, steps + 1):
      s = 0.5 - 0.5 * np.cos(np.pi * i / steps)
      for k, nm in enumerate(ARM_ACT):
        data.ctrl[act[(arm, nm)]] = q0[k] + s * (q_goal[k] - q0[k])
      for _ in range(spf):
        mujoco.mj_step(model, data)
      frame()

  def jaw(arm: str, value: float, seconds: float = 0.6) -> None:
    data.ctrl[act[(arm, "gripper")]] = value
    run(seconds)

  def ik(arm: str, xyz, axis_des, close_des, label: str) -> np.ndarray:
    """Best of several seeds: the commanded pose, and generic reaching poses with the wrist rolled either way."""
    current = [data.ctrl[act[(arm, k)]] for k in ARM_ACT]
    seeds = [current] + [[0.0, p, e, w, r] for (p, e, w) in ((0.9, 1.4, 0.4), (0.7, 1.0, 0.8), (1.3, 1.7, 0.3))
                         for r in (-1.57, 0.0, 1.57)]
    best = None
    for sq in seeds:
      seed = data.qpos.copy(); seed[arms.qadr[arm]] = sq
      res = arms.solve(arm, seed, np.asarray(xyz, float), np.asarray(axis_des, float), np.asarray(close_des, float))
      # prefer accuracy, then staying close to the current pose (no needless wrist flips)
      score = res[1] + 0.002 * (res[2] + res[3]) + 0.003 * np.abs(res[0] - np.array(current)).sum()
      if best is None or score < best[0]:
        best = (score, res)
    q, perr, aerr, cerr = best[1]
    log.append(f"{arm:5s} {label:9s} -> {np.round(xyz, 3)}: pos err {perr * 1000:4.1f} mm, axis {aerr:4.1f} deg, jaws {cerr:4.1f} deg")
    return q

  def sponge_state() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """sponge centre, long axis, width axis (world)"""
    R = data.body("sponge").xmat.reshape(3, 3)
    return data.body("sponge").xpos.copy(), R[:, 0].copy(), R[:, 1].copy()

  def heading(arm: str, xyz) -> np.ndarray:
    """Horizontal direction from the arm's shoulder (pan axis) to the target: the gripper can only point within this plane."""
    sh = data.xanchor[model.joint(ARM_JOINTS[arm][0]).id]
    v = np.asarray(xyz, float)[:2] - sh[:2]
    return np.array([v[0], v[1], 0.0]) / np.linalg.norm(v)

  def grasp_dirs(arm: str, xyz, tilt: float, sideways: bool) -> tuple[np.ndarray, np.ndarray]:
    """Gripper axis tilted down along the arm's heading; jaws closing sideways (horizontal) or in the arm's vertical plane."""
    hd = heading(arm, xyz)
    axis = tilted(tilt, hd)
    across = np.array([-hd[1], hd[0], 0.0])
    return axis, (across if sideways else np.cross(axis, across))

  def tool_now(arm: str) -> np.ndarray:
    arms.d.qpos[:] = data.qpos
    mujoco.mj_forward(model, arms.d)
    return arms.tool(arm)[0]

  # start: arms folded, jaws closed, head looking down at the desk
  for arm in ARM_JOINTS:
    for k, v in zip(ARM_ACT, FOLDED):
      data.ctrl[act[(arm, k)]] = v
  data.ctrl[model.actuator("head_motor_2").id] = 0.95
  for _ in range(1500):
    mujoco.mj_step(model, data)
  run(0.5)

  xy = find_sponge(model, data, renderer)
  if a.truth:
    xy = SPONGE_XY.copy()
  log.append(f"head camera: sponge at {xy.round(3)} (true {SPONGE_XY.round(3)}), error {np.linalg.norm(xy - SPONGE_XY) * 1000:.0f} mm")
  zg = DESK_Z + SPONGE[2] / 2

  # 1. left picks the sponge near its end closest to the robot
  pick = np.array([xy[0] - SPONGE[0] / 2 + EDGE_IN, xy[1], zg])
  jaw("left", JAW_OPEN)
  down, side = grasp_dirs("left", pick, TILT, sideways=True)   # jaws close across the sponge's width
  move("left", ik("left", pick + [0, 0, 0.07], down, side, "hover"), 1.8)
  move("left", ik("left", pick, down, side, "grasp"), 1.0)
  jaw("left", JAW_CLOSED, 0.8)
  move("left", ik("left", pick + [0, 0, 0.07], down, side, "lift"), 1.0)
  log.append(f"after lift: sponge at {data.body('sponge').xpos.round(3)} (lifted if z > {zg + 0.03:.2f})")

  # 2. present: roll back so the jaws close in the vertical plane; the sponge's free end swings toward the right arm
  hand = np.array([0.25, 0.04, DESK_Z + 0.10])
  axis_h, close_h = grasp_dirs("left", hand, 40.0, sideways=False)   # rolled back: jaws close in the arm's vertical plane
  move("left", ik("left", hand, axis_h, close_h, "present"), 2.0)
  run(0.5)
  c, long_ax, width_ax = sponge_state()
  ends = [c + long_ax * (SPONGE[0] / 2 - EDGE_IN), c - long_ax * (SPONGE[0] / 2 - EDGE_IN)]
  lt = tool_now("left")
  take = max(ends, key=lambda e: np.linalg.norm(e - lt))
  log.append(f"presented: sponge at {c.round(3)}, long axis {long_ax.round(2)}, free end {take.round(3)}")

  # 3. right grabs the free end across the width, left lets go and backs off
  axis_r, _ = grasp_dirs("right", take, 40.0, sideways=False)
  jaw("right", JAW_OPEN)
  move("right", ik("right", take + [0, -0.04, 0.05], axis_r, width_ax, "approach"), 1.8)
  move("right", ik("right", take, axis_r, width_ax, "take"), 1.0)
  jaw("right", JAW_CLOSED, 0.8)
  jaw("left", JAW_OPEN, 0.6)
  move("left", ik("left", hand + [-0.05, 0.05, 0.06], axis_h, close_h, "release"), 1.0)
  move("left", FOLDED, 1.4)
  log.append(f"after handover: sponge at {data.body('sponge').xpos.round(3)} (held by right if still in the air)")

  # 4. right rolls so the sponge lies flat again and sets it down
  place = np.array([PLACE_XY[0] - SPONGE[0] / 2 + EDGE_IN, PLACE_XY[1], zg + 0.004])
  down_r, side_r = grasp_dirs("right", place, TILT, sideways=True)
  move("right", ik("right", place + [0, 0, 0.07], down_r, side_r, "carry"), 1.8)
  move("right", ik("right", place, down_r, side_r, "set down"), 1.0)
  jaw("right", JAW_OPEN, 0.6)
  move("right", ik("right", place + [0, 0, 0.08], down_r, side_r, "back off"), 1.0)
  move("right", FOLDED, 1.4)
  run(0.6)
  writer.release()

  c, long_ax, _ = sponge_state()
  for line in log:
    print(line)
  print(f"sponge final {c.round(3)} (target {PLACE_XY.round(3)}, z {DESK_Z + SPONGE[2] / 2:.3f}): off by "
        f"{np.linalg.norm(c[:2] - PLACE_XY) * 1000:.0f} mm, long axis {long_ax.round(2)}")
  print(f"wrote {a.out}")


if __name__ == "__main__":
  main()
