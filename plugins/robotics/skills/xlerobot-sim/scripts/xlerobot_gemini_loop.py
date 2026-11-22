"""Closed-loop XLeRobot control with gemini-4-hydrogen: one checked skill per turn, re-observing after every move.

    python scripts/xlerobot_gemini_loop.py --task "hand the sponge from the left arm to the right arm"   # on the robot computer
    python scripts/xlerobot_gemini_loop.py --dry       # no motor writes: one model turn, print the chosen action

Each turn the model gets the head and both wrist camera images and the robot state (tool positions from forward kinematics of the
measured joints, gripper opening and load, a holding flag) and picks ONE skill from a fixed menu. The harness turns it into joint
targets with the xlerobot-sim MuJoCo model (--arm-raise for our arm mounts), refuses anything unsafe, executes it slowly, observes
again, and reports the result back. Refusals ("IK failure", "would collide with the desk", ...) go back to the model as feedback.

Checks before any motion: IK accuracy, workspace box, both jaw tips above the desk, no new MuJoCo contacts between the arms, an arm
and the desk, or an arm and the robot body (cart, head) along the whole straight-line move, and every joint inside the calibrated
range. During the run: motor temperature stop, turn and time limits. Ctrl+C stops; torque stays on (hold the arms before releasing).

Records head, left wrist and right wrist video (one file each, plus a 3-up video) and a per-turn log (images, state, model reply,
result) in --logdir. The API key comes from GEMINI_API_KEY or ~/.config/gemini/api_key (never stored in the repo).
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import xlerobot_handover_sim as sim  # noqa: E402  (MuJoCo model, IK)

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
SIGN = np.array([1, -1, 1, 1, -1.0])
OFFSET = np.array([0, np.pi / 2, np.pi / 2, 0, 0])
GRIPPER_SPAN_RAD = 2.25
FOLDED = np.array([0.0, 2.6, 2.6, 0.0, 0.0])
BYPATH = "/dev/v4l/by-path/platform-3610000.usb-usb-0:{}:1.0-video-index0"
ARMS = {"left": {"port": "/dev/ttyACM0", "id": "xlerobot_left", "cam": BYPATH.format("2.4")},
        "right": {"port": "/dev/ttyACM1", "id": "xlerobot_right", "cam": BYPATH.format("2.2")}}
HEAD_CAM = BYPATH.format("2.1")
# three cameras share one USB 2.0 hub: the right wrist camera only streams alongside the others at 640x360
CAM_SIZE = {"head": (640, 480), "left": (640, 480), "right": (640, 360)}
MODEL = "gemini-4-hydrogen"
DESK_Z = sim.DESK_Z
WORKSPACE = {"x": (0.15, 0.50), "y": (-0.32, 0.32), "z": (DESK_Z + 0.0, DESK_Z + 0.30)}


def sim_to_real_deg(q_sim) -> np.ndarray:
  return np.degrees((np.asarray(q_sim, float) - OFFSET) / SIGN)


def real_deg_to_sim(q_real) -> np.ndarray:
  return OFFSET + SIGN * np.radians(np.asarray(q_real, float))


# ----------------------------------------------------------------------------------------------------------------------- cameras
class Cameras:
  """Head + wrist cameras in background threads; one video per camera, a 3-up video, and snapshots for the model."""

  def __init__(self, logdir: Path, fps: int = 15):
    self.caps, self.latest, self.stop = {}, {}, False
    for name, dev in (("head", HEAD_CAM), ("left", ARMS["left"]["cam"]), ("right", ARMS["right"]["cam"])):
      cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
      cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
      cap.set(3, CAM_SIZE[name][0]); cap.set(4, CAM_SIZE[name][1]); cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
      self.caps[name] = cap
    for name in self.caps:
      threading.Thread(target=self._grab, args=(name,), daemon=True).start()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    self.writers = {n: cv2.VideoWriter(str(logdir / f"{n}.mp4"), fourcc, fps, CAM_SIZE[n]) for n in self.caps}
    self.writers["tile"] = cv2.VideoWriter(str(logdir / "cameras_3up.mp4"), fourcc, fps, (960, 240))
    self.paths = {n: logdir / (f"{n}.mp4" if n != "tile" else "cameras_3up.mp4") for n in self.writers}
    self.overlay = ""
    threading.Thread(target=self._record, args=(fps,), daemon=True).start()

  def _grab(self, name):
    while not self.stop:
      ok, f = self.caps[name].read()
      if ok and f is not None:
        self.latest[name] = f

  def frame(self, name) -> np.ndarray:
    f = self.latest.get(name)
    return f.copy() if f is not None else np.zeros((CAM_SIZE[name][1], CAM_SIZE[name][0], 3), np.uint8)

  def _record(self, fps):
    while not self.stop:
      t = time.time()
      tiles = []
      for n in ("head", "left", "right"):
        f = self.frame(n)
        self.writers[n].write(f)
        f = cv2.resize(f, (320, 240))
        cv2.putText(f, {"head": "head", "left": "left wrist", "right": "right wrist"}[n], (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 2)
        tiles.append(f)
      tile = np.hstack(tiles)
      if self.overlay:
        cv2.putText(tile, self.overlay[:110], (6, 232), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
      self.writers["tile"].write(tile)
      time.sleep(max(0.0, 1 / fps - (time.time() - t)))

  def wait_ready(self, timeout=6.0) -> list[str]:
    t0 = time.time()
    while time.time() - t0 < timeout and len(self.latest) < 3:
      time.sleep(0.1)
    return [n for n in ("head", "left", "right") if n not in self.latest]

  def close(self):
    self.stop = True; time.sleep(0.4)
    for w in self.writers.values():
      w.release()
    for c in self.caps.values():
      c.release()


# ------------------------------------------------------------------------------------------------------------------- kinematics
class Kinematics:
  """xlerobot-sim model for FK, IK and collision checks (scratch data only; nothing is simulated)."""

  def __init__(self, arm_raise: float):
    sim.ARM_RAISE = arm_raise
    self.model, _ = sim.build()
    self.arms = sim.Arms(self.model)
    m = self.model
    self.cd = mujoco.MjData(m)
    self.sponge_geom = m.geom("sponge").id

    def subtree(root: int) -> set[int]:
      return {b for b in range(m.nbody) if self._is_desc(b, root)}
    self.group = {}
    for b in subtree(m.body("Base").id):
      self.group[b] = "right arm"
    for b in subtree(m.body("Base_2").id):
      self.group[b] = "left arm"
    self.group[m.body("desk").id] = "desk"
    self.group[m.body("sponge").id] = "sponge"

  def _is_desc(self, b: int, root: int) -> bool:
    while b > 0:
      if b == root:
        return True
      b = self.model.body_parentid[b]
    return False

  def grp(self, b: int) -> str:
    return self.group.get(b, "robot body" if b > 0 else "floor")

  def qfull(self, q: dict[str, np.ndarray]) -> np.ndarray:
    full = self.arms.d.qpos.copy() * 0 + self.model.qpos0
    for arm, qa in q.items():
      full[self.arms.qadr[arm]] = qa
    return full

  def tool(self, q: dict[str, np.ndarray], arm: str):
    self.arms.d.qpos[:] = self.qfull(q)
    mujoco.mj_forward(self.model, self.arms.d)
    p, ax, n, _ = self.arms.tool(arm)
    tips = [self.arms.d.body(t).xpos.copy() for t in sim.TIPS[arm]]
    return p, ax, n, tips

  def contacts(self, q: dict[str, np.ndarray]) -> set[tuple[str, str]]:
    d = self.cd
    d.qpos[:] = self.qfull(q)
    mujoco.mj_forward(self.model, d)
    out = set()
    for c in d.contact[:d.ncon]:
      if self.sponge_geom in (c.geom1, c.geom2) or c.dist > -0.002:
        continue
      g1, g2 = self.grp(self.model.geom_bodyid[c.geom1]), self.grp(self.model.geom_bodyid[c.geom2])
      if g1 != g2 and ("arm" in g1 or "arm" in g2):
        out.add(tuple(sorted((g1, g2))))
    return out

  def heading(self, arm: str, xyz) -> np.ndarray:
    self.arms.d.qpos[:] = self.model.qpos0
    mujoco.mj_forward(self.model, self.arms.d)
    sh = self.arms.d.xanchor[self.model.joint(sim.ARM_JOINTS[arm][0]).id]
    v = np.asarray(xyz, float)[:2] - sh[:2]
    return np.array([v[0], v[1], 0.0]) / np.linalg.norm(v)

  def ik(self, arm, q_now: dict, target, axis_des, close_des, extra_seeds=True):
    seeds = [q_now[arm]]
    if extra_seeds:
      seeds += [np.array([0.0, p, e, w, r]) for (p, e, w) in ((0.9, 1.4, 0.4), (0.7, 1.0, 0.8), (1.3, 1.7, 0.3))
                for r in (-1.57, 0.0, 1.57)]
    best = None
    for s in seeds:
      qd = dict(q_now); qd[arm] = s
      q, perr, aerr, cerr = self.arms.solve(arm, self.qfull(qd), np.asarray(target, float), np.asarray(axis_des, float),
                                            np.asarray(close_des, float))
      score = perr + 0.002 * (aerr + cerr) + 0.003 * np.abs(q - q_now[arm]).sum()
      if best is None or score < best[0]:
        best = (score, q, perr, aerr, cerr)
    return best[1:]


# ------------------------------------------------------------------------------------------------------------------------ robot
class Robot:
  def __init__(self, args, dry: bool):
    self.a, self.dry = args, dry
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.robots.so_follower.so_follower import SOFollower
    self.r = {arm: SOFollower(SOFollowerRobotConfig(port=p["port"], id=p["id"], disable_torque_on_disconnect=False,
                                                   position_p_coefficient=args.kp)) for arm, p in ARMS.items()}
    self.limits, self.goal = {}, {}
    for arm, r in self.r.items():
      r.bus.connect()
      cal = r.bus.calibration
      self.limits[arm] = np.array([[(cal[j].range_min - (cal[j].range_min + cal[j].range_max) / 2) * 360 / 4096,
                                    (cal[j].range_max - (cal[j].range_min + cal[j].range_max) / 2) * 360 / 4096] for j in JOINTS])

  def read(self, arm) -> np.ndarray:
    pos = self.r[arm].bus.sync_read("Present_Position")
    return np.array([pos[j] for j in JOINTS])

  def gripper(self, arm) -> dict:
    b = self.r[arm].bus
    pct = float(b.sync_read("Present_Position")["gripper"])
    load = (b.read("Present_Load", "gripper", normalize=False) & 0x3FF) / 10
    return {"opening_pct": round(pct, 1), "load_pct": round(load, 1)}

  def torque_on(self):
    for arm, r in self.r.items():
      q0 = self.read(arm)
      g = r.bus.sync_read("Present_Position")["gripper"]
      if r.bus.read("Torque_Enable", "gripper", normalize=False):
        g = r.bus.read("Goal_Position", "gripper")
      r.bus.sync_write("Goal_Position", {**{j: float(v) for j, v in zip(JOINTS, q0)}, "gripper": float(g)})
      r.configure()
      self.goal[arm] = q0.copy()

  def in_range(self, arm, q_real) -> list[str]:
    lim = self.limits[arm]
    return [JOINTS[k] for k in range(5) if not (lim[k, 0] + 2 <= q_real[k] <= lim[k, 1] - 2)]

  def hot(self) -> str:
    for arm, r in self.r.items():
      t = r.bus.sync_read("Present_Temperature", normalize=False)
      if max(t.values()) >= self.a.max_temp:
        return f"{arm} arm motor too hot: {t}"
    return ""

  def move(self, arm, target, seconds, settle=True) -> float:
    r = self.r[arm]
    start = self.goal[arm].copy()
    dur = max(seconds, np.abs(target - start).max() / self.a.speed)
    n = max(1, int(dur * 30))
    for k in range(1, n + 1):
      s = 0.5 - 0.5 * np.cos(np.pi * k / n)
      r.bus.sync_write("Goal_Position", {j: float(v) for j, v in zip(JOINTS, start + s * (target - start))})
      time.sleep(1 / 30)
    if settle:  # sag compensation: nudge the command until the readings sit on the target
      cmd = target.copy()
      for _ in range(25):
        q = self.read(arm)
        cmd = target + np.clip(cmd - target + 0.3 * (target - q), -8, 8)
        r.bus.sync_write("Goal_Position", {j: float(v) for j, v in zip(JOINTS, cmd)})
        time.sleep(1 / 30)
    self.goal[arm] = target.copy()
    return float(np.abs(self.read(arm) - target).max())

  def open(self, arm, pct):
    """Open to pct (max 65 %), then back off while the servo pushes on its end stop: on 2026-11-22 an open to 75 % read 99 % load
    for minutes (stalled at full torque)."""
    b = self.r[arm].bus
    b.write("Torque_Limit", "gripper", 1000)
    pct = float(np.clip(pct, 5, 65))
    b.write("Goal_Position", "gripper", pct)
    time.sleep(0.9)
    for _ in range(8):
      if (b.read("Present_Load", "gripper", normalize=False) & 0x3FF) < 500:
        break
      pct -= 3
      b.write("Goal_Position", "gripper", pct)
      time.sleep(0.3)
    return pct

  def close(self, arm) -> str:
    """Squeeze by load: close in small steps; contact = load high AND the jaw has stopped moving."""
    b = self.r[arm].bus
    b.write("Torque_Limit", "gripper", 300)
    raw = b.read("Present_Position", "gripper", normalize=False)
    rmin = b.calibration["gripper"].range_min
    load = lambda: b.read("Present_Load", "gripper", normalize=False) & 0x3FF  # noqa: E731
    sq, prev, stalls, contact = 0, raw, 0, False
    while raw - sq > rmin + 5:
      sq += 6
      b.write("Goal_Position", "gripper", int(raw - sq), normalize=False)
      time.sleep(0.08)
      pos = b.read("Present_Position", "gripper", normalize=False)
      stalls = stalls + 1 if (load() >= self.a.grip_load and prev - pos <= 2) else 0
      prev = pos
      if stalls >= 2:
        sq = raw - pos + 6
        b.write("Goal_Position", "gripper", int(raw - sq), normalize=False)
        contact = True
        break
    time.sleep(0.3)
    pos = b.read("Present_Position", "gripper", normalize=False)
    return (f"jaw stopped on something {(pos - rmin) * 360 / 4096:.0f} deg before fully closed (load {load() / 10:.0f} %)"
            if contact else f"jaw closed fully without contact (empty, load {load() / 10:.0f} %)")

  def disconnect(self):
    for r in self.r.values():
      r.bus.disconnect(False)


# ------------------------------------------------------------------------------------------------------------------------- model
SYSTEM = """You control a real two-armed robot (XLeRobot: two SO-101 arms with 5 joints + a parallel gripper each, a head camera).
TASK: {task}

You get three images every turn: HEAD camera (looks down at the desk from above and behind the arms; the robot's LEFT arm and
left side appear on the LEFT of this image), LEFT WRIST camera and RIGHT WRIST camera (each looks along its gripper, jaws visible at
the bottom; wrist images may be rotated with the wrist roll). You also get the measured robot state.

Coordinates (metres, robot frame): x forward from the robot, y to the robot's LEFT, z up; "h" = height above the desk top.
The desk front edge is at x = 0.18. A tape measure put the yellow sponge (10 cm long along x, 6 cm wide, 2 cm tall) with its centre at
about x = 0.31, y = +0.10; this is only a prior and may be several cm off - correct it from the images. The place target for the
sponge is about x = 0.31, y = -0.10. A good handover point (reachable by both arms) is about x = 0.25, y = 0.03, h = 0.10.

Each gripper can only point along its own arm's heading (5 joints): it tilts down by "tilt_deg" (60 = steepest that reaches the desk
here). jaws="across" means the jaws close horizontally across the arm's heading (use this to grasp the sponge across its 6 cm
width from the desk); jaws="vertical" means the jaws close in the arm's vertical plane. The tool point is where the middle of a
grasped 6 cm object would be: to grasp the sponge on the desk use tilt_deg 50-55, jaws "across", open the gripper, hover about 6 cm
above a point about 2.5 cm in from the sponge's end closest to the robot, then descend to h = 0.015 (lower is refused: the jaws would
touch the desk), then close. The gripper opening is in %: open to at most 65 % (about 7 cm between the jaws); 75 % stalled on the end stop.

SKILLS (pick exactly ONE per turn):
  {{"skill": "move_to", "arm": "left"|"right", "x": m, "y": m, "h": m, "tilt_deg": 30..70, "jaws": "across"|"vertical"}}
      moves the tool point there (straight line when within 10 cm, otherwise a joint move from a fresh IK solution).
  {{"skill": "nudge", "arm": ..., "dx": m, "dy": m, "dh": m}}   small correction, each |d| <= 0.03, keeps the gripper orientation.
  {{"skill": "roll", "arm": ..., "deg": -45..45}}               rotate the wrist roll joint.
  {{"skill": "open", "arm": ..., "pct": 10..65}}
  {{"skill": "close", "arm": ...}}                               squeeze until the jaws stop on an object (load-limited, safe).
  {{"skill": "fold", "arm": ...}}                                move the arm back to its folded rest pose.
  {{"skill": "look"}}                                            do nothing, observe again.
  {{"skill": "done", "success": true|false, "reason": "..."}}  end the run.

RULES: Re-check the images after every move; do not assume a move worked. Before closing on the sponge, confirm in the wrist and head
images that it is between the open jaws. After a close, check "holding" in the state and in the images. Handover protocol: the
receiving arm closes first, then you observe, then the giving arm opens alone, then confirm the receiver still holds it. If a skill is
refused (IK failure, collision, out of range) choose a different, reachable target. Move only one arm per turn.

Reply with JSON only:
{{"observation": "what you see now (sponge position relative to the jaws, contact, grip)", "sponge_head_point": [y, x] (0-1000 in the
HEAD image, or null), "plan": "the next few steps", "action": {{...one skill...}}}}"""


def gemini(key: str, system: str, parts: list, tries: int = 6) -> tuple[dict, float]:
  body = {"system_instruction": {"parts": [{"text": system}]}, "contents": [{"role": "user", "parts": parts}],
          "generationConfig": {"responseMimeType": "application/json", "thinkingConfig": {"thinkingLevel": "low"}}}
  url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
  t0 = time.time()
  for i in range(tries):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "x-goog-api-key": key})
    try:
      r = json.load(urllib.request.urlopen(req, timeout=150))
      text = "".join(p.get("text", "") for p in r["candidates"][0]["content"]["parts"] if not p.get("thought"))
      return json.loads(text), time.time() - t0
    except urllib.error.HTTPError as e:
      print(f"   model: HTTP {e.code}, retrying", flush=True)
    except (TimeoutError, urllib.error.URLError, json.JSONDecodeError, KeyError) as e:
      print(f"   model: {type(e).__name__}, retrying", flush=True)
    time.sleep(min(30, 4 * (i + 1)))
  raise RuntimeError("model unavailable")


def jpeg_part(img: np.ndarray) -> dict:
  ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
  return {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(buf.tobytes()).decode()}}


# -------------------------------------------------------------------------------------------------------------------------- main
def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", default="Pick up the yellow sponge from the desk with the LEFT arm, hand it over to the RIGHT arm, "
                                    "and have the RIGHT arm put it down on the desk at the place target. Fold both arms at the end.")
  ap.add_argument("--dry", action="store_true", help="one model turn, no motor writes")
  ap.add_argument("--arm-raise", type=float, default=0.116)
  ap.add_argument("--speed", type=float, default=25.0, help="max joint speed, deg/s")
  ap.add_argument("--kp", type=int, default=32)
  ap.add_argument("--grip-load", type=int, default=120)
  ap.add_argument("--max-temp", type=int, default=65)
  ap.add_argument("--max-turns", type=int, default=45)
  ap.add_argument("--max-minutes", type=float, default=30)
  ap.add_argument("--logdir", default="logs")
  a = ap.parse_args()

  key = os.environ.get("GEMINI_API_KEY") or Path("~/.config/gemini/api_key").expanduser().read_text().strip()
  stamp = time.strftime("%Y%m%d_%H%M%S")
  logdir = Path(a.logdir) / f"gemini_{stamp}"; logdir.mkdir(parents=True)
  kin = Kinematics(a.arm_raise)
  robot = Robot(a, a.dry)
  cams = Cameras(logdir)
  missing = cams.wait_ready()
  if missing:
    print(f"WARNING: no frames from {missing}", flush=True)
  history: list[str] = []
  system = SYSTEM.format(task=a.task)
  t_start = time.time()
  holding = {"left": False, "right": False}
  pose = {"left": {"tilt_deg": 60.0, "jaws": "across"}, "right": {"tilt_deg": 60.0, "jaws": "across"}}

  def q_now() -> dict[str, np.ndarray]:
    return {arm: real_deg_to_sim(robot.read(arm)) for arm in ARMS}

  def state_text(q) -> str:
    lines = []
    for arm in ("left", "right"):
      p, ax, _, tips = kin.tool(q, arm)
      g = robot.gripper(arm)
      lines.append(f"{arm} arm: tool point x={p[0]:.3f} y={p[1]:+.3f} h={p[2] - DESK_Z:+.3f}; gripper pointing "
                   f"{np.degrees(np.arcsin(-ax[2])):.0f} deg below horizontal; lowest jaw tip h={min(t[2] for t in tips) - DESK_Z:+.3f}; "
                   f"gripper opening {g['opening_pct']} %, load {g['load_pct']} %; holding={holding[arm]}")
    return "\n".join(lines)

  def check(arm, q, qa, base_contacts) -> str:
    cur = dict(q); cur[arm] = qa
    _, _, _, tips = kin.tool(cur, arm)
    if min(t[2] for t in tips) < DESK_Z + 0.002:
      return "refused: a jaw tip would go into the desk"
    new = kin.contacts(cur) - base_contacts
    if new:
      return f"refused: collision {sorted(new)} along the move"
    bad = robot.in_range(arm, sim_to_real_deg(qa))
    return f"refused: joints {bad} outside the calibrated range" if bad else ""

  def plan_line(arm, q, end, axis, close, n) -> tuple[list[np.ndarray] | None, str]:
    """IK along a straight tool line (each point seeded from the last), every point checked; joint targets in real deg."""
    start, _, _, _ = kin.tool(q, arm)
    base_contacts = kin.contacts(q)
    cur, out = dict(q), []
    for k in range(1, n + 1):
      pt = start + (end - start) * k / n
      qa, perr, aerr, _ = kin.ik(arm, cur, pt, axis, close, extra_seeds=False)
      if perr > 0.008 or aerr > 15:
        return None, f"IK failure: cannot reach {np.round(end - [0, 0, DESK_Z], 3).tolist()} (x, y, h) on a straight line with that orientation"
      why = check(arm, q, qa, base_contacts)
      if why:
        return None, why
      cur = dict(cur); cur[arm] = qa
      out.append(sim_to_real_deg(qa))
    return out, ""

  def plan_joint(arm, q, end, axis, close) -> tuple[list[np.ndarray] | None, str]:
    """IK for the end pose (several seeds), then a joint-space move; 12 points along it are checked."""
    qa, perr, aerr, _ = kin.ik(arm, q, end, axis, close, extra_seeds=True)
    if perr > 0.008 or aerr > 15:
      return None, (f"IK failure: cannot reach {np.round(end - [0, 0, DESK_Z], 3).tolist()} (x, y, h) with that orientation "
                    f"(best is {perr * 1000:.0f} mm, {aerr:.0f} deg off)")
    base_contacts = kin.contacts(q)
    for k in range(1, 13):
      why = check(arm, q, q[arm] + (qa - q[arm]) * k / 12, base_contacts)
      if why:
        return None, why
    return [sim_to_real_deg(qa)], ""

  def execute(act: dict, q) -> str:
    sk, arm = act.get("skill"), act.get("arm")
    if sk in ("move_to", "nudge", "roll", "open", "close", "fold") and arm not in ARMS:
      return "refused: arm must be 'left' or 'right'"
    if sk == "look":
      time.sleep(0.5); return "observed"
    if sk == "open":
      got = robot.open(arm, float(act.get("pct", 60))); holding[arm] = False
      return f"{arm} gripper opened to {got:.0f} %"
    if sk == "close":
      res = robot.close(arm); holding[arm] = res.startswith("jaw stopped")
      return f"{arm} {res}"
    if sk == "fold":
      tgt = sim_to_real_deg(FOLDED)
      if robot.in_range(arm, tgt):
        return "refused: fold pose outside the calibrated range"
      return f"{arm} folded (off by {robot.move(arm, tgt, 2.5):.1f} deg)"
    if sk == "roll":
      d = float(np.clip(act.get("deg", 0), -45, 45))
      qa = q[arm].copy(); qa[4] += np.radians(d) * 1.0
      qr = sim_to_real_deg(qa)
      if robot.in_range(arm, qr):
        return "refused: wrist roll outside the calibrated range"
      cur = dict(q); cur[arm] = qa
      if kin.contacts(cur) - kin.contacts(q):
        return "refused: the roll would collide"
      return f"{arm} wrist rolled {d:+.0f} deg (off by {robot.move(arm, qr, 1.0):.1f} deg)"
    if sk in ("move_to", "nudge"):
      p, ax, n_cl, _ = kin.tool(q, arm)
      if sk == "move_to":
        end = np.array([float(act["x"]), float(act["y"]), DESK_Z + float(act["h"])])
        tilt = float(np.clip(act.get("tilt_deg", 60), 30, 70)); jaws = act.get("jaws", "across")
        hd = kin.heading(arm, end)
        axis = sim.tilted(tilt, hd); across = np.array([-hd[1], hd[0], 0.0])
        close = across if jaws == "across" else np.cross(axis, across)
        pose[arm] = {"tilt_deg": tilt, "jaws": jaws}
      else:
        d = np.clip([float(act.get("dx", 0)), float(act.get("dy", 0)), float(act.get("dh", 0))], -0.03, 0.03)
        end, axis, close = p + d, ax, n_cl
      for c, (lo, hi) in zip(end, WORKSPACE.values()):
        if not lo <= c <= hi:
          return f"refused: target {np.round(end - [0, 0, DESK_Z], 3).tolist()} (x, y, h) is outside the workspace"
      dist = float(np.linalg.norm(end - p))
      targets, why = None, "IK failure"
      if dist <= 0.10:   # short moves (descend, lift, corrections) go in a straight line
        targets, why = plan_line(arm, q, end, axis, close, max(1, min(8, int(np.ceil(dist / 0.012)))))
      if targets is None and sk == "move_to":   # long moves, or a line that would change the IK branch
        targets, why = plan_joint(arm, q, end, axis, close)
      if targets is None:
        return why
      secs = max(1.0, dist / 0.04)
      for i, t in enumerate(targets):
        err = robot.move(arm, t, secs / len(targets), settle=(i == len(targets) - 1))
      pn, _, _, _ = kin.tool(q_now(), arm)
      return (f"{arm} moved; tool point now x={pn[0]:.3f} y={pn[1]:+.3f} h={pn[2] - DESK_Z:+.3f} "
              f"(target off by {np.linalg.norm(pn - end) * 1000:.0f} mm by the joint readings, joints off by {err:.1f} deg)")
    if sk == "done":
      return "done"
    return f"refused: unknown skill {sk!r}"

  try:
    if not a.dry:
      robot.torque_on()
    for turn in range(1, a.max_turns + 1):
      if (time.time() - t_start) / 60 > a.max_minutes:
        print("time limit reached"); break
      hot = robot.hot()
      if hot:
        print("STOP:", hot); break
      q = q_now()
      st = state_text(q)
      imgs = {n: cams.frame(n) for n in ("head", "left", "right")}
      tdir = logdir / f"turn_{turn:02d}"; tdir.mkdir()
      for n, im in imgs.items():
        cv2.imwrite(str(tdir / f"{n}.jpg"), im)
      parts = [{"text": "HEAD camera:"}, jpeg_part(imgs["head"]), {"text": "LEFT WRIST camera:"}, jpeg_part(imgs["left"]),
               {"text": "RIGHT WRIST camera:"}, jpeg_part(imgs["right"]),
               {"text": f"Turn {turn}. Robot state:\n{st}\n\nHistory (oldest first):\n" + ("\n".join(history[-14:]) or "(start)")
                        + "\n\nChoose the next single skill."}]
      cams.overlay = f"turn {turn}: thinking..."
      reply, dt = gemini(key, system, parts)
      act = reply.get("action", {}) if isinstance(reply, dict) else {}
      print(f"\n[turn {turn}] model {dt:.0f}s\n  sees: {reply.get('observation', '')}\n  plan: {reply.get('plan', '')}\n  action: {act}",
            flush=True)
      cams.overlay = f"turn {turn}: {json.dumps(act)}"
      if a.dry:
        result = "dry run: not executed"
      else:
        try:
          result = execute(act, q)
        except (KeyError, TypeError, ValueError) as e:
          result = f"refused: bad arguments ({e})"
      print(f"  result: {result}", flush=True)
      (tdir / "turn.json").write_text(json.dumps({"state": st, "reply": reply, "result": result, "model_seconds": dt}, indent=1))
      history.append(f"turn {turn}: {json.dumps(act)} -> {result}")
      if a.dry or act.get("skill") == "done":
        break
  except KeyboardInterrupt:
    print("\nstopped (Ctrl+C): torque stays on, arms hold")
  finally:
    cams.close()
    robot.disconnect()
    (logdir / "history.txt").write_text("\n".join(history) + "\n")
    print(f"\nlogs and videos: {logdir.resolve()}")


if __name__ == "__main__":
  main()
