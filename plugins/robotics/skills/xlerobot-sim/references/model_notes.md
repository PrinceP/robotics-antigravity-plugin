# MakerMods XLeRobot model notes

## 1. Sources and licences

| Item | Source | Licence |
|---|---|---|
| `xlerobot.urdf`, `meshes/*.stl` | MakerMods XLeRobot URDF package, published with the MakerMods XLeRobot project on tnkr.ai (`ryans-workspace/xlerobot`, package `makermods-xlerobot-20260930`) | Apache-2.0 (`assets/makermods_xlerobot/LICENCE.txt`) |
| SO-101 arm meshes inside that package | TheRobotStudio SO-ARM100 | Apache-2.0 |
| STS3215 servo parameters | LeRobot / MuJoCo Menagerie SO-101 model | Apache-2.0 |
| Masses, camera lens | XLeRobot documentation, v0.3 hardware (`xlerobot.readthedocs.io`): ~12 kg robot, Innomaker 1080p 130° wrist cameras, STS3215 12 V | |
| Print files (reference only, not used) | `makermods-robotics/MakerMods-XLeRobot` `.3mf` | MIT |

The package also lists `Fixed_Jaw_part*.ply` / `Moving_Jaw_part*.ply`; the hosted files are S3 "Access Denied" pages (111 bytes),
so the build drops those visuals and collisions. The jaws keep their STL meshes.

## 2. What the build script changes

1. Drops the broken `.ply` references; keeps static frames (`fusestatic` off) so camera links survive.
2. Masses: URDF `base_link` 74.9 kg, neck (`top_base_link`) 4.35 kg, head 3.2 kg. Set: neck 0.6 kg, head pan 0.15 kg, head tilt
   0.20 kg, `base_link` = 12 kg total minus everything else (9.45 kg). Inertia scaled with mass, centre of mass kept.
3. Servo model on all arm and head joints (class `sts3215`); base joints (class `base`) lose the URDF friction (10 N) that would
   fight the velocity actuators.
4. Cart, base, neck and head shells are visual only (group 2). Their convex hulls would be solid blocks. Three invisible trays
   (group 3) match the basket floors measured from `makermods_base_light.stl`: z 0.111 / 0.391 / 0.666 m, 0.28 × 0.41 m.
5. Cameras: `head_camera` in `head_camera_rgb_optical_frame` (optical z-forward/y-down converted to MuJoCo's −z/+y);
   wrist cameras in `Right_Arm_Camera` / `Left_Arm_Camera`, oriented along the gripper axis (wrist → jaw tips) with image-up
   away from the jaw axis (`WRIST_CAMERA_QUAT`, computed from the jaw-tip frames).
6. Actuators named like LeRobot's XLeRobot motors (`left_arm_shoulder_pan`, ..., `head_motor_1/2`), ctrlrange = joint range.
7. Root body renamed to the model name (audit convention); timestep 2 ms, `implicitfast`.

## 3. Checks done

- Static audit (`mujoco-audit --static_only`): passed, no errors.
- All 14 servo joints track a step target to < 0.001 rad in 3 s; no self-contact at the zero pose.
- Base: 0.2 m/s for 2 s → 0.39 m; yaw 0.5 rad/s for 2 s → 1.0 rad.
- A 5 cm cube dropped into the top basket rests at z 0.691 m and rides along when the base drives and turns.
- Head camera tilted down sees the top basket between the two arms, as the real head camera does.

## 4. Known gaps

- **Real-robot joint mapping.** LeRobot reports calibrated degrees (0 = middle pose, upper arm up, forearm forward); this URDF
  uses the older SO-101 zero (arm stretched forward) in radians. A per-joint offset and sign table is needed before replaying real
  trajectories; derive it by posing the real arm in two known poses and comparing.
- **Wheels.** MakerMods data gives wheel radius 0.0375 m and track width 0.35 m; the XLeRobot docs list 4-inch omniwheels. The base is
  a planar slide here, so neither affects the sim yet.
- **Cart collision** is only the three basket floors (no walls, legs or rims).
- **Camera intrinsics** are a pinhole approximation of a fisheye lens; calibrate against real frames for perception work.
- **Masses** are distributed from the documented total, not weighed per part.
