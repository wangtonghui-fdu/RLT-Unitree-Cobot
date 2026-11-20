"""Decoded + resized frame cache for a LeRobot v3.0 dataset (lingbot-cobot-ppu, 11-11).

LingBot's LeRobotDataset decodes the queried frames of every camera and immediately applies Resize((S, S))
(img_size, default 256); everything downstream uses the resized frames. This builds, per camera, one concatenated
uint8 array [total_frames, 3, S, S] indexed by the dataset-wide frame index (episode dataset_from_index + frame),
so training can read frames instead of decoding video. Each (camera, video file) is decoded sequentially once with
the same PyAV rgb24 conversion as video_utils.decode_video_frames_pyav_direct, resized with the same
torchvision.transforms.v2.Resize. Built in TMP (tmpfs, safe for concurrent memmap writes), checked complete, then
copied to OUT with cache_meta.json. Exactness is checked separately (verify_frame_cache.py).
usage: build_frame_cache.py DATASET_ROOT TMP_DIR OUT_DIR SIZE NPROC"""
import glob
import json
import os
import shutil
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
import av  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torchvision.transforms.v2 import Resize  # noqa: E402

ROOT, TMP, OUT, SIZE, NPROC = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
info = json.load(open(f"{ROOT}/meta/info.json"))
FPS = info["fps"]
VKEYS = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
EPS = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"{ROOT}/meta/episodes/chunk-*/file-*.parquet"))])
EPS = EPS.sort_values("episode_index").reset_index(drop=True)
N = int(EPS["dataset_to_index"].max())
assert int(EPS["dataset_from_index"].min()) == 0 and int((EPS["dataset_to_index"] - EPS["dataset_from_index"] - EPS["length"]).abs().max()) == 0


def video_path(cam, ch, fi):
    return os.path.join(ROOT, info["video_path"].format(video_key=cam, chunk_index=int(ch), file_index=int(fi)))


def run(job):
    cam, ch, fi = job
    torch.set_num_threads(1)
    sub = EPS[(EPS[f"videos/{cam}/chunk_index"] == ch) & (EPS[f"videos/{cam}/file_index"] == fi)]
    sub = sub.sort_values(f"videos/{cam}/from_timestamp")
    starts = sub[f"videos/{cam}/from_timestamp"].to_numpy(np.float64)
    froms = sub["dataset_from_index"].to_numpy(np.int64)
    lens = sub["length"].to_numpy(np.int64)
    arr = np.load(f"{TMP}/{cam}.npy", mmap_mode="r+")
    filled = np.load(f"{TMP}/{cam}.filled.npy", mmap_mode="r+")
    resize = Resize((SIZE, SIZE))
    t0, n_used, n_skip = time.time(), 0, 0
    buf, idx = [], []

    def flush():
        if buf:
            out = resize(torch.stack(buf)).numpy()
            arr[np.array(idx)] = out
            filled[np.array(idx)] += 1
            buf.clear(); idx.clear()

    with av.open(video_path(cam, ch, fi)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        s.codec_context.thread_count = 2
        tb = float(s.time_base)
        for fr in c.decode(s):
            t = float(fr.pts * tb) if fr.pts is not None else float(fr.time)
            e = int(np.searchsorted(starts, t + 0.5 / FPS, side="right")) - 1
            if e < 0:
                n_skip += 1; continue
            k = int(round((t - starts[e]) * FPS))
            if not (0 <= k < lens[e]) or abs(t - starts[e] - k / FPS) > 0.25 / FPS:
                n_skip += 1; continue
            buf.append(torch.from_numpy(fr.to_ndarray(format="rgb24")).permute(2, 0, 1))
            idx.append(froms[e] + k)
            n_used += 1
            if len(buf) == 64:
                flush()
    flush()
    arr.flush(); filled.flush()
    return cam, int(ch), int(fi), len(sub), n_used, n_skip, round(time.time() - t0, 1)


if __name__ == "__main__":
    from multiprocessing import get_context
    os.makedirs(TMP, exist_ok=True)
    jobs = []
    for cam in VKEYS:
        np.lib.format.open_memmap(f"{TMP}/{cam}.npy", mode="w+", dtype=np.uint8, shape=(N, 3, SIZE, SIZE))
        np.lib.format.open_memmap(f"{TMP}/{cam}.filled.npy", mode="w+", dtype=np.uint8, shape=(N,))
        for ch, fi in sorted(set(zip(EPS[f"videos/{cam}/chunk_index"], EPS[f"videos/{cam}/file_index"]))):
            jobs.append((cam, ch, fi))
    print(f"[cache] {len(VKEYS)} cameras, {N} frames, {len(EPS)} episodes, {len(jobs)} video files, size {SIZE}", flush=True)
    t0 = time.time()
    with get_context("fork").Pool(NPROC) as pool:
        for r in pool.imap_unordered(run, jobs):
            print("[cache] done", r, flush=True)
    ok = True
    for cam in VKEYS:
        f = np.load(f"{TMP}/{cam}.filled.npy", mmap_mode="r")
        bad = int((f != 1).sum())
        print(f"[cache] {cam}: frames filled exactly once: {N - bad}/{N}", flush=True)
        ok &= bad == 0
    if not ok:
        sys.exit("[cache] INCOMPLETE, not copied")
    os.makedirs(OUT, exist_ok=True)
    for cam in VKEYS:
        shutil.copyfile(f"{TMP}/{cam}.npy", f"{OUT}/{cam}.npy.tmp")
        assert os.path.getsize(f"{OUT}/{cam}.npy.tmp") == os.path.getsize(f"{TMP}/{cam}.npy")
        os.replace(f"{OUT}/{cam}.npy.tmp", f"{OUT}/{cam}.npy")
    meta = dict(dataset_root=os.path.realpath(ROOT), fps=FPS, size=[SIZE, SIZE], total_frames=N, n_episodes=len(EPS),
                episode_index=EPS["episode_index"].astype(int).tolist(),
                episode_from=EPS["dataset_from_index"].astype(int).tolist(), episode_length=EPS["length"].astype(int).tolist(),
                cameras={cam: f"{cam}.npy" for cam in VKEYS}, layout="[frame, 3, H, W] uint8, frame = episode_from + frame_index",
                resize="torchvision.transforms.v2.Resize((S, S)) on rgb24 frames (PyAV), same as LingBot LeRobotDataset",
                verified=False, built=time.strftime("%F %T"), build_seconds=round(time.time() - t0))
    json.dump(meta, open(f"{OUT}/cache_meta.json", "w"), indent=1)
    print(f"[cache] copied to {OUT} in {time.time() - t0:.0f}s total", flush=True)
