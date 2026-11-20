"""to_lerobot_dual.py (11-20): gen_dual_v6.py episodes -> LeRobot v3.0 (for LingBot-VLA and openpi pi0.5), dual-arm R1-7a column robot.
Keys (same names as the user's Cobot insert-charge set): observation.images.{camera_top, camera_wrist_left, camera_wrist_right} (video),
observation.state / action (16: left J1..J7 rad, left gripper opening 0..1, right J1..J7, right gripper), and the LingBot split keys
observation.state.{arm,effector}.position / action.{arm,effector}.position (arm = 14 joints left+right, effector = 2 grippers).
state[t] = measured, action[t] = commanded target of the next frame cmd[min(t+1, L-1)] (same convention as the SmolVLA run).
usage: to_lerobot_dual.py DEMO_DIR START COUNT OUT_ROOT REPO_ID   (episodes START .. START+COUNT-1 of the ok episodes, by seed)
       to_lerobot_dual.py --aggregate OUT_ROOT REPO_ID PART1 PART2 ...
"""
import argparse, io, json, shutil, sys, time
from pathlib import Path
import numpy as np

if sys.argv[1] == "--aggregate":
    from lerobot.datasets.aggregate import aggregate_datasets
    root, rid, parts = Path(sys.argv[2]), sys.argv[3], sys.argv[4:]
    if (root / rid).exists(): shutil.rmtree(root / rid)
    aggregate_datasets(repo_ids=parts, aggr_repo_id=rid, roots=[root / p for p in parts], aggr_root=root / rid)
    src = [json.load(open(root / p / "meta" / "r1_source.json")) for p in parts]
    json.dump({"source": src[0]["source"], "episodes": sum((s["episodes"] for s in src), []), "vcodec": src[0]["vcodec"], "parts": parts},
              open(root / rid / "meta" / "r1_source.json", "w"), indent=1)
    info = json.load(open(root / rid / "meta" / "info.json")); print("AGGREGATED", rid, info["total_episodes"], "episodes", info["total_frames"], "frames", flush=True); sys.exit()

import h5py
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ap = argparse.ArgumentParser()
ap.add_argument("demo_dir"); ap.add_argument("start", type=int); ap.add_argument("n", type=int); ap.add_argument("out_root"); ap.add_argument("repo_id")
ap.add_argument("--vcodec", default="h264"); ap.add_argument("--threads", type=int, default=8)
a = ap.parse_args()
eps = sorted(p.parent for p in Path(a.demo_dir).glob("s*/meta.json") if json.loads(p.read_text()).get("ok") and (p.parent / "episode.hdf5").exists())[a.start:a.start + a.n]
assert len(eps) == a.n, (len(eps), a.n)
JL, JR = [f"left_J{i}" for i in range(1, 8)], [f"right_J{i}" for i in range(1, 8)]
N16 = JL + ["left_gripper_open"] + JR + ["right_gripper_open"]; ARM = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14]; EFF = [7, 15]
img = {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]}
CAMS = {"observation.images.camera_top": "head_camera", "observation.images.camera_wrist_left": "left_wrist_camera", "observation.images.camera_wrist_right": "right_wrist_camera"}
features = {**{k: dict(img) for k in CAMS},
            "observation.state": {"dtype": "float32", "shape": (16,), "names": N16}, "action": {"dtype": "float32", "shape": (16,), "names": N16},
            "observation.state.arm.position": {"dtype": "float32", "shape": (14,), "names": JL + JR},
            "observation.state.effector.position": {"dtype": "float32", "shape": (2,), "names": ["left_gripper_open", "right_gripper_open"]},
            "action.arm.position": {"dtype": "float32", "shape": (14,), "names": JL + JR},
            "action.effector.position": {"dtype": "float32", "shape": (2,), "names": ["left_gripper_open", "right_gripper_open"]}}
root = Path(a.out_root) / a.repo_id
if root.exists(): shutil.rmtree(root)
ds = LeRobotDataset.create(repo_id=a.repo_id, fps=30, features=features, root=root, robot_type="unitree_r1_7a_dual_column_dex1",
                           use_videos=True, image_writer_threads=a.threads, vcodec=a.vcodec)
t0 = time.time()
for i, d in enumerate(eps):
    with h5py.File(d / "episode.hdf5", "r") as h:
        v = np.asarray(h["joint_action/vector"], np.float32); q = np.asarray(h["observation/joint_state"], np.float32)
        assert v.shape[1] == 16 and q.shape == v.shape, (d, v.shape, q.shape)
        act = v[np.minimum(np.arange(len(v)) + 1, len(v) - 1)]; task = h.attrs["instruction"]
        cams = {k: h[f"observation/{c}/rgb"] for k, c in CAMS.items()}
        for t in range(len(v)):
            fr = {k: np.asarray(Image.open(io.BytesIO(c[t].tobytes())).convert("RGB")) for k, c in cams.items()}
            fr.update({"observation.state": q[t], "action": act[t], "observation.state.arm.position": q[t, ARM], "observation.state.effector.position": q[t, EFF],
                       "action.arm.position": act[t, ARM], "action.effector.position": act[t, EFF], "task": task})
            ds.add_frame(fr)
    ds.save_episode()
    print(f"episode {a.start + i} {d.name} frames {len(v)} ({time.time() - t0:.0f} s)", flush=True)
ds.finalize()
json.dump({"source": a.demo_dir, "episodes": [d.name for d in eps], "vcodec": a.vcodec}, open(root / "meta" / "r1_source.json", "w"), indent=1)
print("DONE", root, flush=True)
