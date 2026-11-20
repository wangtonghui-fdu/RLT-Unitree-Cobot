"""sheet_wide.py DEMO_DIR OUT.jpg [N]: contact sheet of the first N finished episodes: head-camera frame 0 (initial scene), the handover and
the insertion (head + left wrist + right wrist from preview.mp4, 960x240) -> one row per episode, to review the randomised layouts."""
import sys, json, glob, subprocess, io, os
from PIL import Image, ImageDraw
FF = "/path/to/home/miniconda3/envs/fastwam_rt/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
d, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 16
eps = [os.path.dirname(f) for f in sorted(glob.glob(d + "/s*/meta.json")) if json.load(open(f)).get("ok")][:n]
def frame(mp4, k):
    raw = subprocess.run([FF, "-loglevel", "error", "-i", mp4, "-vf", f"select=eq(n\\,{k})", "-vframes", "1", "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True).stdout
    return Image.open(io.BytesIO(raw)).convert("RGB")
rows = []
for e in eps:
    m = json.load(open(e + "/meta.json")); lg = json.load(open(e + "/log.json")); ph = [r["phase"] for r in lg]
    k1 = ph.index("R_close"); k2 = len(ph) - 1 - ph[::-1].index("R_insert")
    a = frame(e + "/preview.mp4", 0).crop((0, 0, 320, 240)); b = frame(e + "/preview.mp4", k1); c = frame(e + "/preview.mp4", k2)
    row = Image.new("RGB", (320 + 960 + 960, 240)); row.paste(a, (0, 0)); row.paste(b, (320, 0)); row.paste(c, (1280, 0))
    r = m["rand"]; ImageDraw.Draw(row).text((4, 4), f'{os.path.basename(e)}  strip ({r["strip_xy"][0]:.3f},{r["strip_xy"][1]:.3f}) {r["strip_yaw_deg"]:+.0f} deg   charger ({r["charger_xy"][0]:.3f},{r["charger_xy"][1]:.3f}) {r["charger_yaw_deg"]:+.0f} deg   {m["duration_s"]} s', fill=(255, 255, 0))
    rows.append(row)
sh = Image.new("RGB", (2240, 240 * len(rows)))
for i, r in enumerate(rows): sh.paste(r, (0, 240 * i))
sh.save(out, quality=85); print("sheet", len(rows), "episodes ->", out)
