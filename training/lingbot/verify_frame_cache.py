"""Bit-exactness check of a frame cache against LingBot's own frame path (decode_video_frames on the v3 file at
from_timestamp + timestamp, then torchvision v2 Resize on the stacked [current, current + 49 frames] tensor, exactly as
LeRobotDataset._query_videos + image_transforms do). Marks cache_meta.json verified=True only if every sample matches.
usage: verify_frame_cache.py DATASET_ROOT CACHE_DIR N_PER_CAMERA"""
import glob, json, os, sys
import numpy as np, pandas as pd, torch
from torchvision.transforms.v2 import Resize
from lingbotvla.data.vla_data.video_utils import decode_video_frames

ROOT, CACHE, NS = sys.argv[1], sys.argv[2], int(sys.argv[3])
meta = json.load(open(f"{CACHE}/cache_meta.json"))
info = json.load(open(f"{ROOT}/meta/info.json"))
fps = info["fps"]
eps = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"{ROOT}/meta/episodes/chunk-*/file-*.parquet"))]).sort_values("episode_index").reset_index(drop=True)
data = pd.concat([pd.read_parquet(p, columns=["index", "episode_index", "frame_index", "timestamp"]) for p in sorted(glob.glob(f"{ROOT}/data/chunk-*/*.parquet"))]).set_index("index")
assert meta["total_frames"] == len(data) and meta["episode_from"] == eps["dataset_from_index"].astype(int).tolist()
S = tuple(meta["size"])
resize = Resize(S)
rng = np.random.default_rng(1111)
all_ok = True
for cam, fn in meta["cameras"].items():
    arr = np.load(f"{CACHE}/{fn}", mmap_mode="r")
    n_ok = n = 0
    max_diff = 0
    for g in rng.integers(0, meta["total_frames"], NS):
        row = data.loc[int(g)]
        e = eps.iloc[int(row.episode_index)]
        assert int(e.episode_index) == int(row.episode_index)
        k_fut = min(int(row.frame_index) + 49, int(e.length) - 1)       # future frame offset (49 frames), clamped
        ts = [float(row.timestamp), float(data.loc[int(e.dataset_from_index) + k_fut].timestamp)]
        q = [float(e[f"videos/{cam}/from_timestamp"]) + t for t in ts]
        path = os.path.join(ROOT, info["video_path"].format(video_key=cam, chunk_index=int(e[f"videos/{cam}/chunk_index"]), file_index=int(e[f"videos/{cam}/file_index"])))
        ref = resize(decode_video_frames(path, q, 1e-4, "torchcodec")).numpy()
        got = arr[[int(g), int(e.dataset_from_index) + k_fut]]
        d = int(np.abs(ref.astype(np.int16) - got.astype(np.int16)).max())
        max_diff = max(max_diff, d)
        n += 1; n_ok += d == 0
    print(f"[verify] {cam}: {n_ok}/{n} samples (current + future frame) bit-identical, max abs diff {max_diff}", flush=True)
    all_ok &= n_ok == n
if all_ok:
    meta["verified"] = True
    meta["verify_samples_per_camera"] = NS
    json.dump(meta, open(f"{CACHE}/cache_meta.json", "w"), indent=1)
print("[verify] ALL OK" if all_ok else "[verify] MISMATCH", flush=True)
