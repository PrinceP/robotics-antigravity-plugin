# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "opencv-python-headless", "pillow"]
# ///
"""Render a demo video of an xlerobot_gemini_loop.py run: the three camera views, the model's observation, plan and action, the
result, the measured arm state, the model's sponge point on the head view, and a turn timeline. Waiting for the model is sped up.

    uv run scripts/xlerobot_gemini_demo_video.py logs/gemini_20261122_122916 --out demo.mp4 \
        --tag "Gemini API free tier · 20 requests/day"

Needs ffmpeg (H.264 output). Timing comes from the log files' modification times (copy the log folder with rsync -a / scp -p).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import textwrap
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS_OUT, FPS_SRC = 1920, 1080, 30, 15
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
BG, PANEL, FG, DIM = (16, 18, 24), (26, 29, 38), (235, 237, 242), (150, 156, 170)
AMBER, BLUE, RED, GREEN, YELLOW = (240, 170, 40), (70, 150, 240), (230, 70, 70), (80, 200, 120), (250, 220, 60)
SPEED = {"think": 12.0, "exec": 1.5, "error": 20.0}


def f(path, size):
  return ImageFont.truetype(path, size)


def parse_state(text: str) -> dict:
  out = {}
  for line in text.splitlines():
    m = re.match(r"(\w+) arm: tool point x=([-+\d.]+) y=([-+\d.]+) h=([-+\d.]+); gripper pointing ([-\d]+) deg.*?lowest jaw tip "
                 r"h=([-+\d.]+); gripper opening ([\d.]+) %, load ([\d.]+) %; holding=(\w+)", line)
    if m:
      out[m[1]] = dict(x=float(m[2]), y=float(m[3]), h=float(m[4]), tilt=int(m[5]), tip=float(m[6]), open=float(m[7]),
                       load=float(m[8]), holding=m[9] == "True")
  return out


def load_run(logdir: Path):
  n_frames = int(cv2.VideoCapture(str(logdir / "head.mp4")).get(cv2.CAP_PROP_FRAME_COUNT))
  t_end = (logdir / "head.mp4").stat().st_mtime
  t0 = t_end - n_frames / FPS_SRC
  turns = []
  for d in sorted(logdir.glob("turn_*")):
    obs = (d / "head.jpg").stat().st_mtime - t0
    tj = d / "turn.json"
    if tj.exists():
      j = json.loads(tj.read_text())
      turns.append(dict(n=int(d.name[5:]), obs=obs, think_end=obs + j["model_seconds"], end=tj.stat().st_mtime - t0, reply=j["reply"],
                        result=j["result"], state=parse_state(j["state"]), model_s=j["model_seconds"]))
    else:   # the model never answered (quota / outage)
      turns.append(dict(n=int(d.name[5:]), obs=obs, think_end=None, end=n_frames / FPS_SRC, reply=None, result=None, state=None))
  return n_frames, turns


def phase_at(t, turns):
  for tr in turns:
    if tr["obs"] <= t < tr["end"] + 1e-6:
      if tr["think_end"] is None:
        return tr, "error"
      return tr, ("think" if t < tr["think_end"] else "exec")
  return (turns[0], "think") if t < turns[0]["obs"] else (turns[-1], "exec")


def wrap(draw, xy, text, font, width_chars, fill, line_h, max_lines=99):
  x, y = xy
  lines = textwrap.wrap(text or "", width_chars)
  if len(lines) > max_lines:
    lines = lines[:max_lines]; lines[-1] += " …"
  for ln in lines:
    draw.text((x, y), ln, font=font, fill=fill); y += line_h
  return y


def card(lines, tag) -> Image.Image:
  im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
  y = 300
  for text, font, fill in lines:
    w = d.textlength(text, font=font)
    d.text(((W - w) / 2, y), text, font=font, fill=fill); y += font.size + 28
  badge(d, tag, center=True)
  return im


def badge(d, tag, center=False):
  font = f(BOLD, 26)
  w = d.textlength(tag, font=font) + 40
  x = (W - w) / 2 if center else W - w - 24
  y = 820 if center else 16
  d.rounded_rectangle([x, y, x + w, y + 44], 10, fill=RED)
  d.text((x + 20, y + 8), tag, font=font, fill=(255, 255, 255))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("logdir")
  ap.add_argument("--out", default="gemini_demo.mp4")
  ap.add_argument("--tag", default="Gemini API free tier · 20 requests/day")
  ap.add_argument("--title", default="XLeRobot × gemini-4-hydrogen · real-robot sponge pick")
  ap.add_argument("--date", default="22 Nov 2026")
  ap.add_argument("--poster", default=None, help="also save the frame at this source time (s) as a JPEG")
  a = ap.parse_args()
  logdir = Path(a.logdir)
  n_frames, turns = load_run(logdir)
  total = n_frames / FPS_SRC
  caps = {n: cv2.VideoCapture(str(logdir / f"{n}.mp4")) for n in ("head", "left", "right")}
  F = dict(t=f(BOLD, 30), h=f(BOLD, 34), b=f(FONT, 24), s=f(FONT, 20), m=f(MONO, 22), lab=f(BOLD, 22), big=f(BOLD, 50), mid=f(FONT, 34))

  ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS_OUT),
                         "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                         a.out], stdin=subprocess.PIPE)

  def emit(im: Image.Image, n=1):
    b = np.asarray(im, np.uint8).tobytes()
    for _ in range(n):
      ff.stdin.write(b)

  done = [t for t in turns if t["result"]]
  err = [t for t in turns if t["result"] is None]
  emit(card([(a.title, F["big"], FG), (f"{a.date} · closed loop on a MakerMods XLeRobot · head + both wrist cameras", F["mid"], DIM),
             ("Each turn the model sees 3 images + measured arm state and picks ONE skill;", F["mid"], FG),
             ("MuJoCo checks IK, desk clearance and collisions before any motion.", F["mid"], FG)], a.tag), FPS_OUT * 4)

  # timeline geometry
  tl_x0, tl_x1, tl_y = 24, W - 24, 1036
  seg_col = {"think": AMBER, "exec": BLUE, "error": RED}
  timeline = Image.new("RGB", (W, 44), BG); td = ImageDraw.Draw(timeline)
  X = lambda t: tl_x0 + (tl_x1 - tl_x0) * t / total  # noqa: E731
  for tr in turns:
    if tr["think_end"] is None:
      td.rectangle([X(tr["obs"]), 8, X(tr["end"]), 28], fill=RED)
    else:
      td.rectangle([X(tr["obs"]), 8, X(tr["think_end"]), 28], fill=AMBER)
      td.rectangle([X(tr["think_end"]), 8, X(tr["end"]), 28], fill=BLUE)
    td.line([X(tr["obs"]), 4, X(tr["obs"]), 32], fill=BG, width=3)
    td.text((X(tr["obs"]) + 4, 8), str(tr["n"]), font=F["s"], fill=BG)

  acc, poster_done = 0.0, False
  for i in range(n_frames):
    frames = {}
    for n, c in caps.items():
      ok, fr = c.read()
      frames[n] = fr if ok else None
    t = i / FPS_SRC
    tr, ph = phase_at(t, turns)
    acc += FPS_OUT / (FPS_SRC * SPEED[ph])
    k = int(acc); acc -= k
    want_poster = a.poster and not poster_done and t >= float(a.poster)
    if k == 0 and not want_poster:
      continue
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    d.text((24, 20), f"{a.title} · {a.date}", font=F["t"], fill=FG)
    badge(d, a.tag)
    # camera row
    for col, (n, label) in enumerate((("head", "HEAD camera"), ("left", "LEFT wrist"), ("right", "RIGHT wrist"))):
      fr = frames[n]
      if fr is None:
        continue
      fr = cv2.resize(fr, (640, 480 if n != "right" else 360))
      if n == "head" and tr["reply"] and tr["reply"].get("sponge_head_point") and ph != "think":
        py, px = tr["reply"]["sponge_head_point"]
        u, v = int(px * 640 / 1000), int(py * 480 / 1000)
        cv2.circle(fr, (u, v), 18, (60, 220, 250), 3); cv2.line(fr, (u - 28, v), (u + 28, v), (60, 220, 250), 2)
        cv2.line(fr, (u, v - 28), (u, v + 28), (60, 220, 250), 2)
        cv2.putText(fr, "model: sponge", (u + 22, v - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 220, 250), 2)
      y0 = 76 + (0 if n != "right" else 60)
      im.paste(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)), (col * 640, y0))
      d.rectangle([col * 640 + 8, 84, col * 640 + 8 + d.textlength(label, font=F["lab"]) + 16, 114], fill=(0, 0, 0))
      d.text((col * 640 + 16, 86), label, font=F["lab"], fill=YELLOW)
    # bottom panel
    d.rectangle([0, 564, W, 1022], fill=PANEL)
    head = {"think": f"TURN {tr['n']} · gemini-4-hydrogen thinking… {max(0.0, t - tr['obs']):.0f} s",
            "exec": f"TURN {tr['n']} · executing (model took {tr.get('model_s') or 0:.0f} s)",
            "error": f"TURN {tr['n']} · HTTP 429 RESOURCE_EXHAUSTED · free-tier daily quota used up"}[ph]
    d.text((24, 580), head, font=F["h"], fill=seg_col[ph])
    d.text((W - 260, 584), f"playback {SPEED[ph]:g}×", font=F["b"], fill=DIM)
    if ph == "error":
      y = 640
      for ln in ("Every retry returned 429: generate_content_free_tier_requests, limit 20 per day per model (gemini-4-hydrogen).",
                 "The loop stopped safely; the left arm held its pose with torque on.",
                 f"Previous result: {turns[-2]['result'] if len(turns) > 1 else ''}"):
        y = wrap(d, (24, y), ln, F["b"], 100, FG, 34)
    elif tr["reply"]:
      r = tr["reply"]
      y = 632
      if ph == "think":
        prev = [p for p in turns if p["n"] == tr["n"] - 1]
        d.text((24, y), "Sent:", font=F["lab"], fill=DIM)
        y = wrap(d, (110, y), "head + left wrist + right wrist images, the measured arm state (right) and the history of earlier turns",
                 F["b"], 82, FG, 31) + 10
        if prev:
          d.text((24, y), "Last:", font=F["lab"], fill=DIM)
          y = wrap(d, (110, y + 2), json.dumps(prev[0]["reply"].get("action", {})), F["m"], 80, YELLOW, 29, 2) + 4
          wrap(d, (110, y), "→ " + prev[0]["result"], F["b"], 82, RED if "empty" in prev[0]["result"] else GREEN, 31, 2)
        else:
          wrap(d, (24, y), "Task: pick up the sponge with the LEFT arm, hand it to the RIGHT arm, place it on the right.", F["b"], 90, FG, 31)
      else:
        d.text((24, y), "Sees:", font=F["lab"], fill=DIM); y = wrap(d, (110, y), r.get("observation", ""), F["b"], 82, FG, 31, 4) + 6
        d.text((24, y), "Plan:", font=F["lab"], fill=DIM); y = wrap(d, (110, y), r.get("plan", ""), F["b"], 82, FG, 31, 2) + 6
      if ph == "exec":
        d.text((24, y), "Action:", font=F["lab"], fill=DIM)
        y = wrap(d, (130, y + 2), json.dumps(r.get("action", {})), F["m"], 78, YELLOW, 29, 2) + 6
        res = tr["result"]
        d.text((24, y), "Result:", font=F["lab"], fill=DIM)
        wrap(d, (130, y), res, F["b"], 78, RED if ("empty" in res or "refused" in res) else GREEN, 31, 3)
    # state table (measured at the start of this turn)
    st = tr["state"] or (turns[-2]["state"] if len(turns) > 1 else None)
    if st:
      x0 = 1240
      d.text((x0, 632), "Arm state at turn start (measured joints → FK)", font=F["lab"], fill=DIM)
      cols = ["", "x cm", "y cm", "h cm", "tilt°", "grip %", "load %", "holding"]
      xs = [x0, x0 + 86, x0 + 166, x0 + 246, x0 + 326, x0 + 396, x0 + 476, x0 + 560]
      for c, xx in zip(cols, xs):
        d.text((xx, 672), c, font=F["s"], fill=DIM)
      for row, arm in enumerate(("left", "right")):
        s = st[arm]; yy = 706 + row * 40
        vals = [arm, f"{s['x'] * 100:.1f}", f"{s['y'] * 100:+.1f}", f"{s['h'] * 100:.1f}", f"{s['tilt']}", f"{s['open']:.0f}",
                f"{s['load']:.0f}", "yes" if s["holding"] else "no"]
        for v, xx in zip(vals, xs):
          col = RED if (v == vals[6] and s["load"] > 80) else FG
          d.text((xx, yy), v, font=F["m"], fill=col)
      d.text((x0, 800), "x forward, y to the robot's left, h above the desk top", font=F["s"], fill=DIM)
      d.text((x0, 828), "(tool point = where a grasped object's centre would be)", font=F["s"], fill=DIM)
      legend = [("model thinking", AMBER), ("robot moving", BLUE), ("quota error", RED)]
      lx = x0
      for name, c in legend:
        d.rectangle([lx, 884, lx + 22, 906], fill=c); d.text((lx + 30, 882), name, font=F["s"], fill=FG); lx += 220
      d.text((x0, 930), f"turns executed: {len(done)}   ·   run time {total / 60:.1f} min", font=F["b"], fill=FG)
    im.paste(timeline, (0, tl_y - 8))
    px = X(t); d.polygon([(px - 9, tl_y - 14), (px + 9, tl_y - 14), (px, tl_y)], fill=FG)
    d.line([px, tl_y, px, tl_y + 26], fill=FG, width=2)
    if want_poster:
      im.save(a.poster_out if hasattr(a, "poster_out") else Path(a.out).with_suffix(".jpg"), quality=88); poster_done = True
    emit(im, k)

  times = [t["model_s"] for t in done]
  emit(card([("Result: no grasp yet", F["big"], FG),
             (f"{len(done)} turns executed; every move reached its target within 3 mm (joint readings).", F["mid"], FG),
             ("Grasp missed: the jaws closed a few cm beyond the sponge; the wrist camera cannot see under its own jaws.", F["mid"], FG),
             (f"Model latency {min(times):.0f}–{max(times):.0f} s per turn (high demand, HTTP 503 retries).", F["mid"], FG),
             (f"Stopped at turn {err[0]['n'] if err else '-'}: HTTP 429, free-tier limit of 20 gemini-4-hydrogen requests per day.",
              F["mid"], RED)], a.tag), FPS_OUT * 7)
  ff.stdin.close(); ff.wait()
  print(f"wrote {a.out}")


if __name__ == "__main__":
  main()
