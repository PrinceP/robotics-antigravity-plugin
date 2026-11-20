# /// script
# requires-python = ">=3.10"
# dependencies = ["mujoco>=3.2"]
# ///
"""Build the MuJoCo model of the MakerMods XLeRobot from MakerMods' URDF package.

    uv run scripts/build_xlerobot_mjcf.py            # writes assets/makermods_xlerobot/xlerobot.xml and scene.xml

Source: MakerMods XLeRobot URDF + meshes (Apache-2.0, assets/makermods_xlerobot/LICENCE.txt), published with the
MakerMods XLeRobot project on tnkr.ai. What this script changes, and why:

- Placeholder `.ply` jaw meshes in the package are S3 "Access Denied" pages, not meshes: those visuals/collisions are dropped.
- Masses: the URDF puts 74.9 kg in `base_link` (cart + omniwheel base + power bank), 4.35 kg in the neck and 3.2 kg in the head;
  the neck and head get realistic values and `base_link` the rest of the documented ~12 kg total (inertia scaled with mass).
- Static camera frames are kept (fusestatic off) and real <camera> elements added: `head_camera` (optical frame of the head camera),
  `wrist_camera_right`, `wrist_camera_left`.
- Arm and head joints get the Feetech STS3215 servo model used for SO-101 in LeRobot/MuJoCo (armature, damping, friction, position
  actuators kp=998.22 kv=2.731, +-3.35 Nm). The planar base (x, y, yaw) gets velocity actuators.
- The cart is visual only (its convex hull would be a solid block); the baskets get thin collision boxes so objects can rest in them.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent.parent / "assets" / "makermods_xlerobot"
URDF = HERE / "xlerobot.urdf"
OUT = HERE / "xlerobot.xml"
SCENE = HERE / "scene.xml"

TOTAL_MASS = 12.0  # kg: XLeRobot docs (v0.3): "~12 kg". base_link (cart + omniwheel base + power bank + electronics) gets the remainder
NECK_MASS = 0.6  # kg: printed two-part neck column (top_base_link; the URDF has 4.35 kg)
HEAD_PAN_MASS = 0.15  # kg: neck-top servo bracket + tilt servo
HEAD_TILT_MASS = 0.20  # kg: head camera, case, mount
CAMERA_FOVY = {"head_camera": 100.0, "wrist_camera_right": 100.0, "wrist_camera_left": 100.0}  # deg vertical: Innomaker 1080p 130 deg diagonal, 4:3 pinhole equivalent

ARMS = {"right": "", "left": "_2"}  # URDF joint suffix per arm (the right arm carries Right_Arm_Camera)
ARM_JOINTS = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
LEROBOT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
HEAD_JOINTS = ["head_pan_joint", "head_tilt_joint"]
BASE_JOINTS = ["root_x_axis_joint", "root_y_axis_joint", "root_z_rotation_joint"]


def prepared_urdf() -> Path:
  src = URDF.read_text()
  src = re.sub(r"<(visual|collision)>(?:(?!</\1>).)*?\.ply\"(?:(?!</\1>).)*?</\1>", "", src, flags=re.S)
  compiler = ('<mujoco><compiler meshdir="meshes" strippath="true" balanceinertia="true" discardvisual="false" '
              'fusestatic="false" boundmass="0.001" boundinertia="0.000001"/></mujoco>')
  src = src.replace('<robot name="xlerobot">', '<robot name="xlerobot">' + compiler, 1)
  tmp = HERE / "_build.urdf"
  tmp.write_text(src)
  return tmp


def scale_mass(body: ET.Element, mass: float) -> None:
  inertial = body.find("inertial")
  old = float(inertial.get("mass"))
  inertial.set("mass", f"{mass:.6g}")
  for key in ("diaginertia", "fullinertia"):
    if inertial.get(key):
      vals = np.array([float(v) for v in inertial.get(key).split()]) * (mass / old)
      inertial.set(key, " ".join(f"{v:.6g}" for v in vals))


def main() -> None:
  tmp = prepared_urdf()
  model = mujoco.MjModel.from_xml_path(str(tmp))
  raw = HERE / "_raw.xml"
  mujoco.mj_saveLastXML(str(raw), model)
  tmp.unlink()
  root = ET.parse(raw).getroot()
  raw.unlink()
  root.set("model", "makermods_xlerobot")
  bodies = {b.get("name"): b for b in root.iter("body")}
  bodies["root"].set("name", "makermods_xlerobot")  # URDF root link; MuJoCo convention: root body named after the model

  comp = root.find("compiler")
  comp.set("meshdir", "meshes")
  comp.set("autolimits", "true")
  for k in ("boundmass", "boundinertia", "fusestatic", "discardvisual", "strippath", "balanceinertia"):
    comp.attrib.pop(k, None)
  opt = root.find("option") if root.find("option") is not None else ET.SubElement(root, "option")
  opt.set("timestep", "0.002")
  opt.set("integrator", "implicitfast")
  root.remove(opt)
  root.insert(1, opt)

  # servo + base defaults
  default = ET.Element("default")
  sts = ET.SubElement(default, "default", {"class": "sts3215"})
  ET.SubElement(sts, "joint", {"damping": "0.60", "frictionloss": "0.052", "armature": "0.028"})
  ET.SubElement(sts, "position", {"kp": "998.22", "kv": "2.731", "forcerange": "-3.35 3.35"})
  base = ET.SubElement(default, "default", {"class": "base"})
  ET.SubElement(base, "joint", {"damping": "1", "frictionloss": "0", "armature": "0.1"})
  ET.SubElement(base, "velocity", {"kv": "400", "forcerange": "-400 400"})
  vis = ET.SubElement(default, "default", {"class": "visual"})
  ET.SubElement(vis, "geom", {"contype": "0", "conaffinity": "0", "group": "2"})
  old_default = root.find("default")
  if old_default is not None:
    root.remove(old_default)
  root.insert(2, default)

  # masses
  scale_mass(bodies["top_base_link"], NECK_MASS)
  scale_mass(bodies["head_pan_link"], HEAD_PAN_MASS)
  scale_mass(bodies["head_tilt_link"], HEAD_TILT_MASS)
  others = sum(float(b.find("inertial").get("mass")) for n, b in bodies.items() if n != "base_link" and b.find("inertial") is not None)
  scale_mass(bodies["base_link"], TOTAL_MASS - others)

  # joints: servo class on arms/head, base class on the planar base, no URDF friction on the base
  for j in root.find("worldbody").iter("joint"):  # body joints only: the <default> joints carry the servo model
    name = j.get("name", "")
    for k in ("damping", "frictionloss", "armature", "actuatorfrcrange"):
      j.attrib.pop(k, None)
    if name in BASE_JOINTS:
      j.set("class", "base")
    elif any(name == a + s for a in ARM_JOINTS for s in ARMS.values()) or name in HEAD_JOINTS:
      j.set("class", "sts3215")

  # cart + base + head shells: visual only; baskets get thin collision trays (measured from the cart mesh: see references)
  for name in ("base_link", "head_pan_link", "head_tilt_link"):
    for g in bodies[name].findall("geom"):
      g.set("class", "visual")
      for k in ("contype", "conaffinity", "group"):
        g.attrib.pop(k, None)
  for i, (z, sx, sy) in enumerate(BASKETS):
    ET.SubElement(bodies["base_link"], "geom", {"name": f"basket_{i}_floor", "type": "box", "size": f"{sx} {sy} 0.004",
                                                "pos": f"{BASKET_X} 0 {z}", "rgba": "0 0 0 0", "group": "3"})

  # cameras (MuJoCo cameras look along -z with +y up)
  def add_camera(body_name: str, cam: str, quat: str) -> None:
    ET.SubElement(bodies[body_name], "camera", {"name": cam, "mode": "fixed", "quat": quat, "fovy": f"{CAMERA_FOVY[cam]}"})
  add_camera("head_camera_rgb_optical_frame", "head_camera", "0 1 0 0")  # optical frame: z forward, y down
  add_camera("Right_Arm_Camera", "wrist_camera_right", WRIST_CAMERA_QUAT)
  add_camera("Left_Arm_Camera", "wrist_camera_left", WRIST_CAMERA_QUAT)

  # actuators: names follow LeRobot's XLeRobot motor names
  act = root.find("actuator")
  if act is not None:
    root.remove(act)
  act = ET.SubElement(root, "actuator")
  for jn, an, ax in (("root_x_axis_joint", "base_x", "x"), ("root_y_axis_joint", "base_y", "y"), ("root_z_rotation_joint", "base_yaw", "yaw")):
    ctrl = "-0.5 0.5" if ax != "yaw" else "-1.5 1.5"
    ET.SubElement(act, "velocity", {"class": "base", "name": an, "joint": jn, "ctrlrange": ctrl})
  for arm, suf in ARMS.items():
    for jn, ln in zip(ARM_JOINTS, LEROBOT_NAMES):
      rng = next(j for j in root.iter("joint") if j.get("name") == jn + suf).get("range")
      ET.SubElement(act, "position", {"class": "sts3215", "name": f"{arm}_arm_{ln}", "joint": jn + suf, "ctrlrange": rng})
  for jn, an in zip(HEAD_JOINTS, ("head_motor_1", "head_motor_2")):
    rng = next(j for j in root.iter("joint") if j.get("name") == jn).get("range")
    ET.SubElement(act, "position", {"class": "sts3215", "name": an, "joint": jn, "ctrlrange": rng})

  ET.indent(root, "  ")
  OUT.write_text("<!-- Generated by scripts/build_xlerobot_mjcf.py from MakerMods' xlerobot.urdf (Apache-2.0). Do not edit by hand. -->\n"
                 + ET.tostring(root, encoding="unicode") + "\n")
  SCENE.write_text(SCENE_XML)
  m = mujoco.MjModel.from_xml_path(str(SCENE))
  d = mujoco.MjData(m)
  for _ in range(500):
    mujoco.mj_step(m, d)
  print(f"wrote {OUT.name} and {SCENE.name}: {m.nbody} bodies, {m.njnt} joints, {m.nu} actuators, {m.ncam} cameras, "
        f"mass {m.body_subtreemass[1]:.2f} kg; 1 s of simulation ok (max |qvel| {np.abs(d.qvel).max():.3g})")


# Basket floors: (box centre z, half-x, half-y) in base_link coordinates, and their x centre. Measured from the cart mesh
# (makermods_base_light: up-facing floors at z 0.111 / 0.391 / 0.666 m, x -0.14..+0.14, y -0.205..+0.205); box tops sit on the floors.
BASKETS = [(0.107, 0.14, 0.205), (0.387, 0.14, 0.205), (0.662, 0.14, 0.205)]
BASKET_X = 0.001
# Wrist cameras look along the gripper (wrist -> jaw tips); image "up" points away from the jaw axis (the camera sits 5 cm to its
# side), so with the gripper level, world-up appears at an image side, as on the real robot. Computed from the jaw-tip frames.
WRIST_CAMERA_QUAT = "0.7423 0.0137 -0.6697 -0.0152"

SCENE_XML = """<mujoco model="makermods_xlerobot_scene">
  <include file="xlerobot.xml"/>
  <statistic center="0 0 0.6" extent="1.4"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <global azimuth="150" elevation="-20" offwidth="1280" offheight="960"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="0 0 3.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
  </worldbody>
</mujoco>
"""

if __name__ == "__main__":
  main()
