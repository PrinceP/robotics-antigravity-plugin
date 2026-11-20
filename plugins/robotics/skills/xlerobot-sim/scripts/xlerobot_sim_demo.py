# /// script
# requires-python = ">=3.10"
# dependencies = ["mujoco>=3.2", "numpy", "opencv-python-headless"]
# ///
"""Drive the MakerMods XLeRobot in MuJoCo and record what it and its cameras see.

    MUJOCO_GL=egl uv run scripts/xlerobot_sim_demo.py --out xlerobot_demo.mp4        # headless video
    uv run scripts/xlerobot_sim_demo.py --viewer                                      # interactive viewer

Video layout: overview | head_camera on top, wrist_camera_left | wrist_camera_right below. A cube is dropped into the top basket.
The base is commanded in the robot frame (forward, left, yaw rate) and converted to the world-frame planar joints.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import mujoco
import numpy as np

SCENE = Path(__file__).resolve().parent.parent / "assets" / "makermods_xlerobot" / "scene.xml"
ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def load(with_cube: bool = True) -> tuple[mujoco.MjModel, mujoco.MjData]:
  spec = mujoco.MjSpec.from_file(str(SCENE))
  if with_cube:
    body = spec.worldbody.add_body(name="cube", pos=[0.05, 0.0, 0.72])
    body.add_freejoint()
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.025] * 3, rgba=[0.95, 0.4, 0.1, 1], mass=0.05)
  model = spec.compile()
  return model, mujoco.MjData(model)


def drive_base(model: mujoco.MjModel, data: mujoco.MjData, forward: float, left: float, yaw_rate: float) -> None:
  """Robot-frame base velocity (m/s, m/s, rad/s) -> world-frame x/y slide velocities + yaw rate."""
  yaw = data.qpos[model.jnt_qposadr[model.joint("root_z_rotation_joint").id]]
  c, s = np.cos(yaw), np.sin(yaw)
  data.ctrl[model.actuator("base_x").id] = c * forward - s * left
  data.ctrl[model.actuator("base_y").id] = s * forward + c * left
  data.ctrl[model.actuator("base_yaw").id] = yaw_rate


def set_arm(model: mujoco.MjModel, data: mujoco.MjData, side: str, q: dict[str, float]) -> None:
  for name, value in q.items():
    a = model.actuator(f"{side}_arm_{name}").id
    data.ctrl[a] = np.clip(value, *model.actuator_ctrlrange[a])


def script(t: float, model: mujoco.MjModel, data: mujoco.MjData) -> None:
  """A short routine: raise the arms, look down into the basket, drive forward, turn, wave."""
  ready = {"shoulder_pan": 0.0, "shoulder_lift": 1.0, "elbow_flex": 1.3, "wrist_flex": 0.4, "wrist_roll": 0.0, "gripper": 0.6}
  wave = 0.5 * np.sin(2 * np.pi * 0.6 * t) if t > 6 else 0.0
  set_arm(model, data, "right", {**ready, "shoulder_pan": wave})
  set_arm(model, data, "left", {**ready, "shoulder_pan": -wave})
  data.ctrl[model.actuator("head_motor_2").id] = 0.8 if t < 4 else 0.3
  data.ctrl[model.actuator("head_motor_1").id] = 0.4 * np.sin(2 * np.pi * 0.2 * t) if t > 4 else 0.0
  if 2 < t < 4:
    drive_base(model, data, 0.2, 0.0, 0.0)
  elif 4 < t < 6:
    drive_base(model, data, 0.0, 0.0, 0.6)
  else:
    drive_base(model, data, 0.0, 0.0, 0.0)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", default="xlerobot_demo.mp4")
  ap.add_argument("--seconds", type=float, default=9.0)
  ap.add_argument("--fps", type=int, default=30)
  ap.add_argument("--viewer", action="store_true", help="open the interactive viewer instead of recording")
  a = ap.parse_args()
  model, data = load()
  steps_per_frame = int(round(1 / (a.fps * model.opt.timestep)))

  if a.viewer:
    from mujoco import viewer as mj_viewer  # local name: `import mujoco.viewer` here would shadow the module in main()
    with mj_viewer.launch_passive(model, data) as v:
      while v.is_running() and data.time < 600:
        script(data.time, model, data)
        mujoco.mj_step(model, data)
        v.sync()
    return

  w, h = 480, 360
  renderer = mujoco.Renderer(model, h, w)
  overview = mujoco.MjvCamera()
  overview.distance, overview.azimuth, overview.elevation = 2.2, 145, -18
  writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (2 * w, 2 * h))
  for _ in range(int(a.seconds * a.fps)):
    for _ in range(steps_per_frame):
      script(data.time, model, data)
      mujoco.mj_step(model, data)
    overview.lookat[:] = data.body("base_link").xpos + [0.1, 0, 0.7]
    tiles = []
    for cam in (overview, "head_camera", "wrist_camera_left", "wrist_camera_right"):
      renderer.update_scene(data, cam)
      img = renderer.render().copy()
      cv2.putText(img, cam if isinstance(cam, str) else "overview", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
      tiles.append(img)
    frame = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
  writer.release()
  print(f"wrote {a.out} ({a.seconds:.0f} s); cube at {data.body('cube').xpos.round(3)}, base at {data.body('base_link').xpos.round(3)}")


if __name__ == "__main__":
  main()
