# Handover failure cases (simulation)

Recorded while developing `scripts/xlerobot_handover_sim.py`. Each one shows a real failure mode and what it taught us. Layout of the
videos: overview | head_camera on top, wrist_camera_left | wrist_camera_right below.

| File | What happens | Cause | Fix |
|---|---|---|---|
| `01_perception_offset_jaw_pushes_sponge.mp4` | The left jaw lands on the sponge's edge and pushes it along the desk instead of gripping it. | The head-camera estimate was ~20 mm off sideways: the blob includes the sponge's side face, which biases its centre. With a 6 cm sponge, 20 mm puts a jaw on top of it. | Estimate the centre from the blob's far edge (top-face height) and near edge (desk height); move the camera estimate onto the true pose before grasping on the real robot (hover check). |
| `02_grasp_does_not_lift.mp4` | With the true sponge position, the jaws close but the sponge stays on the desk; the rest of the sequence runs empty. | Two model bugs at the time: the MakerMods package's jaw collision meshes are broken `.ply` files (jaws had no collision), and the gripper pinched at full servo force with non-parallel rigid pads, wedging the sponge out. | Jaw contact pads, parallel at the grasp opening; gripper force limited to 0.5 N m like the real load-limited squeeze. |
| `03_moving_jaw_lands_on_top.jpg` | Close-up of the grasp (top: open, bottom: closed). The fixed jaw is beside the sponge, but the moving jaw comes down on its top face. | The moving jaw swings in an arc: opened wide, its tip retracts up, so it meets the 2 cm sponge from above, presses it into the desk and slides off the top edge. | Open only as wide as needed and grasp lower, so the moving jaw meets the side face (in progress). |

Why this matters for the real robot: 11 cm into the desk the arms cannot point straight down at the sponge (the wrist would need to be
0.31 m from the shoulder; upper arm + forearm are 0.25 m), so every grasp is tilted ~60 deg, and the arc of the moving jaw decides
whether it meets the side or the top of a flat object.
