"""Read or move the XLeRobot head (pan = head_motor_1, tilt = head_motor_2; IDs 7, 8 on the left arm's bus), in calibrated degrees.

    python scripts/xlerobot_head.py                       # print pan/tilt
    python scripts/xlerobot_head.py --tilt 40 --pan 0     # move slowly there and hold (torque stays on)
    python scripts/xlerobot_head.py --tilt 40 --snap head.jpg

Uses the head entries of the XLeRobot calibration file. Stop the XLeRobot host/bridge first.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

CALIB = Path("~/.cache/huggingface/lerobot/calibration/robots/xlerobot/xlerobot_3wheel.json").expanduser()
PORT = "/dev/ttyACM0"
HEAD_CAM = "/dev/v4l/by-path/platform-3610000.usb-usb-0:2.1:1.0-video-index0"


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--pan", type=float, default=None)
  ap.add_argument("--tilt", type=float, default=None)
  ap.add_argument("--speed", type=float, default=20.0, help="deg/s")
  ap.add_argument("--snap", default=None, help="save a head camera frame here after moving")
  a = ap.parse_args()
  cal = json.loads(CALIB.read_text())
  motors = {"head_motor_1": Motor(7, "sts3215", MotorNormMode.DEGREES), "head_motor_2": Motor(8, "sts3215", MotorNormMode.DEGREES)}
  bus = FeetechMotorsBus(PORT, motors, calibration={k: MotorCalibration(**cal[k]) for k in motors})
  bus.connect()
  try:
    now = bus.sync_read("Present_Position")
    print(f"head now: pan {now['head_motor_1']:+.1f}  tilt {now['head_motor_2']:+.1f} deg")
    target = dict(now)
    if a.pan is not None:
      target["head_motor_1"] = a.pan
    if a.tilt is not None:
      target["head_motor_2"] = a.tilt
    if target != now:
      bus.sync_write("Goal_Position", now)
      bus.enable_torque(num_retry=5)
      n = max(1, int(max(abs(target[k] - now[k]) for k in motors) / a.speed * 30))
      for i in range(1, n + 1):
        bus.sync_write("Goal_Position", {k: now[k] + (target[k] - now[k]) * i / n for k in motors})
        time.sleep(1 / 30)
      time.sleep(0.6)
      now = bus.sync_read("Present_Position")
      print(f"head at:  pan {now['head_motor_1']:+.1f}  tilt {now['head_motor_2']:+.1f} deg (torque on, holding)")
  finally:
    bus.disconnect(False)
  if a.snap:
    cap = cv2.VideoCapture(HEAD_CAM, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")); cap.set(3, 640); cap.set(4, 480)
    for _ in range(15):
      ok, f = cap.read()
    cv2.imwrite(a.snap, f); cap.release()
    print(f"saved {a.snap}")


if __name__ == "__main__":
  main()
