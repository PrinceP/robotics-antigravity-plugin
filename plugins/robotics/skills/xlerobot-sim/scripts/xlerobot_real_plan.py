"""Run a motion plan exported by xlerobot_handover_sim.py on the real XLeRobot, step by step.

    python scripts/xlerobot_real_plan.py plan.json --dry          # convert + check every target, move nothing
    python scripts/xlerobot_real_plan.py plan.json                # run; pauses at checkpoints (Enter = continue, q = stop)

Runs on the robot computer with LeRobot installed (SOFollower per arm, calibration ids xlerobot_left / xlerobot_right: split the
XLeRobot calibration first). Stop the XLeRobot host/bridge first: this needs both arms' serial ports.

Sim -> real joint mapping (derived by matching the MakerMods URDF to LeRobot's so101_new_calib model, which LeRobot drives in
calibrated degrees; link directions match to 0.00 deg across random poses):
    real_deg = degrees((sim_rad - OFFSET) / SIGN),  SIGN = [+1, -1, +1, +1, -1],  OFFSET = [0, pi/2, pi/2, 0, 0]
    joints: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll. Same for both arms.
Gripper: "open" -> percent of the calibrated range (sim rad / 2.25 rad); "close" -> squeeze by load (never commands "closed").

Safety: moves are slow (--speed deg/s) and interpolated; the run pauses at checkpoints before contact so you can check alignment;
it stops if a motor reaches --max-temp, and refuses targets outside the calibrated joint range. Torque stays on at the end (hold the
arms before releasing them).
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
SIGN = np.array([1, -1, 1, 1, -1.0])
OFFSET = np.array([0, np.pi / 2, np.pi / 2, 0, 0])
GRIPPER_SPAN_RAD = 2.25            # calibrated gripper travel (~129 deg) -> 0..100 %
ARMS = {"left": {"port": "/dev/ttyACM0", "id": "xlerobot_left", "cam": "/dev/v4l/by-path/platform-3610000.usb-usb-0:2.4:1.0-video-index0"},
        "right": {"port": "/dev/ttyACM1", "id": "xlerobot_right", "cam": "/dev/v4l/by-path/platform-3610000.usb-usb-0:2.2:1.0-video-index0"}}
HEAD_CAM = "/dev/v4l/by-path/platform-3610000.usb-usb-0:2.1:1.0-video-index0"
CHECKPOINTS = ("ready", "hover", "present", "approach", "carry")


def sim_to_real_deg(q_sim) -> np.ndarray:
  return np.degrees((np.asarray(q_sim, float) - OFFSET) / SIGN)


class Cameras:
  """Head + both wrist cameras in background threads; snapshots and a 3-up recording."""

  def __init__(self, out: Path, fps: int = 10):
    self.caps = {}
    for name, dev in (("head", HEAD_CAM), ("left", ARMS["left"]["cam"]), ("right", ARMS["right"]["cam"])):
      cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
      cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")); cap.set(3, 640); cap.set(4, 480); cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
      self.caps[name] = cap
    self.latest: dict[str, np.ndarray] = {}
    self.stop = False
    self.out = out
    for name in self.caps:
      threading.Thread(target=self._grab, args=(name,), daemon=True).start()
    self.writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (960, 240))
    threading.Thread(target=self._record, args=(fps,), daemon=True).start()

  def _grab(self, name):
    while not self.stop:
      ok, f = self.caps[name].read()
      if ok:
        self.latest[name] = f

  def tile(self) -> np.ndarray:
    tiles = []
    for name in ("head", "left", "right"):
      f = self.latest.get(name)
      f = cv2.resize(f, (320, 240)) if f is not None else np.zeros((240, 320, 3), np.uint8)
      cv2.putText(f, name, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
      tiles.append(f)
    return np.hstack(tiles)

  def _record(self, fps):
    while not self.stop:
      self.writer.write(self.tile()); time.sleep(1 / fps)

  def snapshot(self, path: Path):
    cv2.imwrite(str(path), self.tile())

  def close(self):
    self.stop = True; time.sleep(0.3); self.writer.release()
    for c in self.caps.values():
      c.release()


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("plan")
  ap.add_argument("--dry", action="store_true", help="convert and check only; no motor writes")
  ap.add_argument("--speed", type=float, default=30.0, help="max joint speed, deg/s")
  ap.add_argument("--kp", type=int, default=32)
  ap.add_argument("--grip-load", type=int, default=120, help="squeeze until the gripper load reaches this (0.1 %% units)")
  ap.add_argument("--max-temp", type=int, default=65)
  ap.add_argument("--no-pause", action="store_true", help="do not stop at checkpoints")
  ap.add_argument("--logdir", default="logs")
  a = ap.parse_args()
  plan = json.loads(Path(a.plan).read_text())
  steps = plan["steps"]
  ready = {"move": list(sim_to_real_deg([0.0, 2.6, 2.6, 0.0, 0.0])), "label": "ready"}

  from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
  from lerobot.robots.so_follower.so_follower import SOFollower
  robots = {arm: SOFollower(SOFollowerRobotConfig(port=p["port"], id=p["id"], disable_torque_on_disconnect=False,
                                                 position_p_coefficient=a.kp)) for arm, p in ARMS.items()}
  limits = {}
  for arm, r in robots.items():
    r.bus.connect()
    cal = r.bus.calibration
    limits[arm] = np.array([[(cal[j].range_min - (cal[j].range_min + cal[j].range_max) / 2) * 360 / 4096,
                             (cal[j].range_max - (cal[j].range_min + cal[j].range_max) / 2) * 360 / 4096] for j in JOINTS])

  # check every target against the calibrated range
  bad = []
  print(f"{'step':5s} {'arm':5s} {'label':16s} real degrees (pan, lift, elbow, wrist_flex, wrist_roll) / gripper")
  for i, s in enumerate(steps):
    if "move" in s:
      q = sim_to_real_deg(s["move"])
      out = [JOINTS[k] for k in range(5) if not (limits[s["arm"]][k, 0] + 2 <= q[k] <= limits[s["arm"]][k, 1] - 2)]
      bad += [(i, s["arm"], s["label"], out)] if out else []
      print(f"{i:5d} {s['arm']:5s} {s['label']:16s} {np.round(q, 1)}" + (f"   OUT OF RANGE: {out}" if out else ""))
    else:
      print(f"{i:5d} {s['arm']:5s} {'jaw ' + s['jaw']:16s} " + (f"{100 * s['sim_rad'] / GRIPPER_SPAN_RAD:.0f} %" if s["jaw"] == "open" else "squeeze by load"))
  if bad:
    print("targets outside the calibrated range:", bad)
  if a.dry or bad:
    for r in robots.values():
      r.bus.disconnect(False)
    print("dry run: nothing moved" if a.dry else "not running: fix the out-of-range targets first")
    return

  logdir = Path(a.logdir); logdir.mkdir(exist_ok=True)
  stamp = time.strftime("%H%M%S")
  cams = Cameras(logdir / f"handover_real_{stamp}.mp4")
  read = {arm: (lambda r=r: np.array([r.bus.sync_read("Present_Position")[j] for j in JOINTS])) for arm, r in robots.items()}
  goal = {}
  for arm, r in robots.items():  # torque on holding the present pose (keep any existing gripper goal)
    q0 = read[arm]()
    g = r.bus.sync_read("Present_Position")["gripper"]
    if r.bus.read("Torque_Enable", "gripper", normalize=False):
      g = r.bus.read("Goal_Position", "gripper")
    r.bus.sync_write("Goal_Position", {**{j: float(v) for j, v in zip(JOINTS, q0)}, "gripper": float(g)})
    r.configure()
    goal[arm] = q0.copy()

  def temps_ok() -> bool:
    for arm, r in robots.items():
      t = r.bus.sync_read("Present_Temperature", normalize=False)
      if max(t.values()) >= a.max_temp:
        print(f"STOP: {arm} arm motor too hot {t}"); return False
    return True

  def move(arm: str, target: np.ndarray, seconds: float):
    r = robots[arm]
    start = goal[arm].copy()
    dur = max(seconds, np.abs(target - start).max() / a.speed)
    n = max(1, int(dur * 30))
    for k in range(1, n + 1):
      s = 0.5 - 0.5 * np.cos(np.pi * k / n)
      cmd = start + s * (target - start)
      r.bus.sync_write("Goal_Position", {j: float(v) for j, v in zip(JOINTS, cmd)})
      time.sleep(1 / 30)
    cmd = target.copy()  # sag compensation: nudge the command until the readings sit on the target
    for _ in range(25):
      q = read[arm]()
      cmd = target + np.clip(cmd - target + 0.3 * (target - q), -8, 8)
      r.bus.sync_write("Goal_Position", {j: float(v) for j, v in zip(JOINTS, cmd)})
      time.sleep(1 / 30)
    goal[arm] = target.copy()
    return np.abs(read[arm]() - target).max()

  def jaw(arm: str, action: str, sim_rad: float):
    b = robots[arm].bus
    if action == "open":
      b.write("Torque_Limit", "gripper", 1000)
      b.write("Goal_Position", "gripper", float(min(80.0, 100 * sim_rad / GRIPPER_SPAN_RAD)))
      time.sleep(0.8)
      return
    # squeeze by load: close in small steps until the load reaches --grip-load (commanding "closed" would stall the servo)
    b.write("Torque_Limit", "gripper", 300)
    raw = b.read("Present_Position", "gripper", normalize=False)
    rmin = b.calibration["gripper"].range_min
    load = lambda: b.read("Present_Load", "gripper", normalize=False) & 0x3FF
    sq = 0
    while raw - sq > rmin + 5:
      sq += 6
      b.write("Goal_Position", "gripper", int(raw - sq), normalize=False)
      time.sleep(0.06)
      if load() >= a.grip_load:
        break
    time.sleep(0.3)
    print(f"   {arm} gripper: closed {sq * 360 / 4096:.0f} deg, load {load() / 10:.0f} %")

  def pause(label: str, arm: str):
    snap = logdir / f"handover_real_{stamp}_{arm}_{label.split()[0]}.jpg"
    time.sleep(0.4); cams.snapshot(snap)
    if a.no_pause:
      return True
    ans = input(f"CHECKPOINT {arm} {label}: snapshot {snap}. Enter = continue, q = stop here (torque stays on): ").strip().lower()
    return ans != "q"

  try:
    for arm in ("left", "right"):
      err = move(arm, np.array(ready["move"]), 2.0)
      print(f"{arm} -> ready (off by {err:.1f} deg)")
    if not pause("ready", "both"):
      return
    for i, s in enumerate(steps):
      if not temps_ok():
        break
      arm = s["arm"]
      if "move" in s:
        err = move(arm, sim_to_real_deg(s["move"]), s["seconds"])
        print(f"{i:3d} {arm:5s} {s['label']:16s} done (off by {err:.1f} deg)", flush=True)
        if s["label"].split()[0] in CHECKPOINTS and not s["label"].endswith(("1/8", "2/8", "3/8", "4/8", "5/8", "6/8", "7/8", "1/4", "2/4", "3/4")):
          if not pause(s["label"], arm):
            break
      else:
        print(f"{i:3d} {arm:5s} jaw {s['jaw']}", flush=True)
        jaw(arm, s["jaw"], s["sim_rad"])
  except KeyboardInterrupt:
    print("\nstopped (Ctrl+C): torque stays on, arms hold")
  finally:
    cams.close()
    for r in robots.values():
      r.bus.disconnect(False)
    print(f"video: {cams.out}")


if __name__ == "__main__":
  main()
