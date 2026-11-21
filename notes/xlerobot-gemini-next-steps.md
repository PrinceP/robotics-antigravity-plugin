# XLeRobot: next approach with Gemini (working notes)

*Status: discussion notes, nothing implemented yet. November 2026.*

Where we are: the sponge handover failed on the real robot (see the [blog post](../blog/xlerobot-sponge-handover/index.md)). The main
causes were open-loop execution of a precomputed plan, a robot model that did not match our build (arm mounts 11.6 cm higher), uncalibrated
cameras, and a 5-joint arm handing over a flat 6 cm sponge. Two references point the way forward:

- [Gemini 4 Hydrogen for robotics](https://developers.googleblog.com/gemini-hydrogen-for-robotics/) (Google, Nov 17 2026): a fast,
  cheaper Gemini 4 model used **as the agent** in Antigravity with this plugin. It plans the task, writes and runs control code that calls
  the plugin's skills, and keeps adjusting from camera images. It is not vision-only and not a VLA that outputs joint actions; actions go
  through generated code and the plugin's IK, planning and drivers. Its perception is also callable on its own (`gemini-4-hydrogen` via
  the Gemini API, as in `rgb-object-localization`). Claims are demos; no latency, pricing, success rates or limitations are given.
- [Open Gemini 4 Argon Robot Harness](https://ehehee.github.io/blog/open-gemini-harness/) (Shuangyu Xie): a zero-shot bimanual handover on
  two 6-joint YAM arms. Each turn the model sees fresh images and robot state and picks **one** move from a fixed skill menu; the harness
  solves IK, checks it (human approval or a MuJoCo collision check), executes, and observes again. One clean success in six development
  runs (metal clamp), one after a re-pick (tape roll); thin and flat objects failed.

## Three ways to bring Gemini in

| | 1. Antigravity + plugin (Hydrogen as agent) | 2. Gemini API, fixed-skill loop (Shuangyu's approach) | 3. Gemini for perception only |
|---|---|---|---|
| Who decides the next move | Hydrogen writes and runs code | the model picks one validated skill per turn | our scripted plan |
| Closed loop | yes (agent iterates) | yes, after every segment | no |
| Safety on hardware | depends on the agent following the plugin rules; arbitrary code | only known skills with numeric arguments; human or collision check per segment | human checkpoints |
| Work for us | least code | port the harness to XLeRobot | smallest change |
| Access | paid Google AI Pro / Ultra | `GEMINI_API_KEY` (pay per use) | `GEMINI_API_KEY` |
| Catch | expects the robot on the same machine; ours is on the Orin (Antigravity on Linux ARM unconfirmed; the SO-101 skills assume a local serial port) | needs camera-to-arm calibration | keeps the open-loop weakness that failed both real runs |

## Recommendation: option 2

Port the open harness to XLeRobot, using Hydrogen (faster and cheaper per turn than Argon; model calls were about half of Shuangyu's
8-minute run).

What we already have for it:

| Harness piece | What exists |
|---|---|
| Arm driver | `xlerobot_real_plan.py`: SOFollower per arm, slow interpolated moves, sag compensation, load + stall grasp, temperature stop |
| IK and collision check | `xlerobot-sim` MuJoCo model + multi-seed IK; `--arm-raise 0.116` for our arm mounts |
| Sim-to-real joint mapping | derived and verified (see `references/model_notes.md` and the blog) |
| Cameras | head + both wrist cameras (Innomaker, 640×480); head pan/tilt via `xlerobot_head.py` |
| **Camera-to-arm calibration** | **missing**: needed to turn the model's pixel picks into 3D targets |

## Order of work

1. **Head-camera calibration**: the head camera's pose relative to both arm bases (for example a printed marker the gripper touches at a
   few known points). Everything after this depends on it.
2. **Port the harness**: XLeRobot adapter for the arm driver, our MuJoCo model for IK and collision checks, the head camera as the scene
   camera, and the 5-joint limits reported back as "IK failure".
3. **Single-arm pick and place** of a rigid, narrow object (3–4 cm).
4. **Handover** with his protocol: receiver closes → observe → giver opens alone → check the grip (position + load) on both arms.
5. **The sponge** last (flat and soft; his fan and plastic clamp failed for similar reasons).

## Lessons that apply whichever model drives it

- Model the actual rig before planning (measure mount heights, base, desk, cameras on the real robot).
- Calibrate cameras to the arms; do not rely on tape-measured object positions.
- Re-observe after every move; never run a long open-loop plan.
- Check the grip on both arms before the giver lets go.
- Design grasps for 5-joint arms: the gripper points along its arm's heading, so the two arms' grasps are skewed (~45°) at a shared point.

## Open decisions

- [ ] Gemini API key (option 2) or AI Pro / Ultra (option 1)? A key lives in an environment variable on the machine, never in the repo.
- [ ] Split: the loop on the laptop (model calls, 3D preview, approving moves) with the Orin as a small robot server (moves, camera
      frames, state), close to the web bridge from the first session?
- [ ] Calibration method and target for the head camera.
- [ ] First rigid test object.
