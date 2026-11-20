"""px_extract.py (11-18): dense near-field frames for the pixel-ceiling control.
Every valid frame (strip static, frame <= t*) whose right-TCP-to-target distance is < --near mm, all three cameras at the
native 480x640, decoded with lerobot 0.4.2 (no resize).  Labels as in probe.py (hindsight target = right TCP at t*).
Writes OUT/px_meta.npz (episode, frame, gindex, lab [mm, target TCP frame], dist, rot) and OUT/<cam>.npy uint8 [N,480,640,3].
usage: px_extract.py --ds DS --phases phases.json --out DIR [--near 30] [--workers 32]
"""
import argparse, glob, json, os, sys, time
import numpy as np, pandas as pd, torch

ap = argparse.ArgumentParser()
ap.add_argument("--ds", required=True)
ap.add_argument("--phases", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--near", type=float, default=30)
ap.add_argument("--workers", type=int, default=32)
ap.add_argument("--fk", default="/path/to/storage/lingbot-cobot/tools")
ap.add_argument("--tip", type=float, default=0.135)
args = ap.parse_args()
sys.path.insert(0, args.fk)
from piper_fk import fk_sdk
from lerobot.datasets.lerobot_dataset import LeRobotDataset

t0 = time.time()
d = pd.read_parquet(glob.glob(os.path.join(args.ds, "data/*/*.parquet"))[0],
                    columns=["episode_index", "frame_index", "index", "observation.state"])
S = np.stack(d["observation.state"].values).astype(np.float64)
E = d["episode_index"].values; FI = d["frame_index"].values; GI = d["index"].values
ph = {int(k): v for k, v in json.load(open(args.phases)).items() if v}
rows = []
for e, v in ph.items():
    m = np.where(E == e)[0]
    s = S[m]
    TR = fk_sdk(s[:, 7:13]); TL = fk_sdk(s[:, 0:6])
    tip = TR[:, :3, 3] + args.tip * TR[:, :3, 2]
    ts = v["t_star"]; Rs = TR[ts, :3, :3]
    lab = ((tip - tip[ts]) @ Rs) * 1000.0
    dist = np.linalg.norm(lab, axis=1)
    Rrel = np.einsum("ji,njk->nik", Rs, TR[:, :3, :3])
    rot = np.degrees(np.arccos(np.clip((np.trace(Rrel, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    holding = s[ts, 6] > 0.03
    lmove = np.linalg.norm(TL[:, :3, 3] - TL[ts, :3, 3], axis=1) * 1000.0
    ok = (np.arange(len(m)) <= ts) & (dist < args.near) & ~(holding & (lmove > 3.0))
    for i in np.where(ok)[0]:
        rows.append((e, FI[m[i]], GI[m[i]], *lab[i], dist[i], rot[i]))
rows = np.array(rows)
print(f"[px] {len(rows)} near-field frames (< {args.near:g} mm) over {len(np.unique(rows[:, 0]))} episodes", flush=True)
os.makedirs(args.out, exist_ok=True)
np.savez(os.path.join(args.out, "px_meta.npz"), episode=rows[:, 0].astype(int), frame=rows[:, 1].astype(int),
         gindex=rows[:, 2].astype(int), lab=rows[:, 3:6], dist=rows[:, 6], rot=rows[:, 7], tip=args.tip, near=args.near)

name = args.ds.rstrip("/").split("/")[-1]
ds = LeRobotDataset(name, root=args.ds)
CAMS = {"top": "observation.images.camera_top", "wr": "observation.images.camera_wrist_right",
        "wl": "observation.images.camera_wrist_left"}
gidx = rows[:, 2].astype(int)


class Sel(torch.utils.data.Dataset):
    def __len__(self):
        return len(gidx)

    def __getitem__(self, i):
        it = ds[int(gidx[i])]
        assert int(it["index"]) == int(gidx[i])
        return {k: (it[v] * 255).round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0) for k, v in CAMS.items()}


mm = {k: np.lib.format.open_memmap(os.path.join(args.out, f"{k}.npy.tmp"), mode="w+", dtype=np.uint8,
                                   shape=(len(gidx), 480, 640, 3)) for k in CAMS}
loader = torch.utils.data.DataLoader(Sel(), batch_size=32, num_workers=args.workers)
n = 0
for bi, b in enumerate(loader):
    for k in CAMS:
        mm[k][n:n + len(b[k])] = b[k].numpy()
    n += len(b["wr"])
    if bi % 20 == 0:
        print(f"[px] decoded {n}/{len(gidx)} ({time.time() - t0:.0f}s)", flush=True)
for k in CAMS:
    mm[k].flush(); del mm[k]
    os.replace(os.path.join(args.out, f"{k}.npy.tmp"), os.path.join(args.out, f"{k}.npy"))
print(f"[px] done: {n} frames x 3 cams in {time.time() - t0:.0f}s", flush=True)
