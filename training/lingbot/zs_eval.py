"""Open-loop eval (lingbot-cobot-ppu copy of lingbot-cobot/tools/ab/ab_eval.py, 11-11: LeRobot v3.0 datasets via
lerobot 0.4.2 as well as v2.1; summary adds the hold-still baseline).
A/B open-loop eval: one model on one GPU over every dataset in an eval list.
Same observation/GT handling as the official scripts/open_loop_eval.py (apply+unapply for GT, policy server for preds),
but: chunk starts every --stride frames over the whole episode (only starts whose 50-step GT is fully inside the
episode), batched inference, per-batch seeds that depend only on (dataset, batch) so every model sees the same noise,
per-robot robot configs from our robot_config_root (each has its own norm_stats). Writes OUT/<dataset>.npz
(pred/gt absolute actions [N, chunk, D], state, episode, frame) and skips datasets already written.
usage: ab_eval.py --model HF_CKPT_DIR --list ab_eval.txt --robot-cfg DIR --out DIR [--stride 60] [--batch 8] [--dry]
HF_CKPT_DIR/../../../lingbotvla_cli.yaml must exist (official deploy layout)."""
import argparse, os, sys, time, types, zlib
import numpy as np, torch, yaml

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--list", required=True)
ap.add_argument("--robot-cfg", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--stride", type=int, default=60)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--max-batches", type=int, default=0, help="debug: stop each dataset after this many batches")
ap.add_argument("--dry", action="store_true", help="CPU check of the data path; no weights, no GPU")
args = ap.parse_args()

CODE = os.environ["LINGBOT_CODE"]
sys.path.insert(0, CODE)
if args.dry:  # import-time device queries (group_gemm config lookup) without touching a GPU
    torch.cuda.get_device_capability = lambda *a, **k: (8, 0)
    torch.cuda.get_device_name = lambda *a, **k: "dry"
import lingbotvla  # noqa: F401  (torch 2.4 compat + MetaX speedups)
import deploy.lingbot_vla_v2_policy as pol
from lingbotvla.data.vla_data.base_dataset import LeRobotDataset
from lingbotvla.data.vla_data.utils import FeatureTransform
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # lerobot 0.4.x (v3.0)
    V3 = True
except ImportError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    V3 = False

if args.dry:
    class _Stub(torch.nn.Module):
        def __init__(self, *a, **k):
            super().__init__()
            self.model = types.SimpleNamespace(float=lambda: None, sample_actions=None)
        def cuda(self, *a, **k): return self
        def eval(self): return self
    pol.LingBotVlaV2InferencePolicy = _Stub
    pol.LingbotVLAv2Server.load_model_weights = lambda self, *a, **k: None
else:
    torch.set_float32_matmul_precision("high")  # training ran fp32 params with TF32 matmuls
    if os.environ.get("LINGBOT_MEM_FRACTION"):
        torch.cuda.set_per_process_memory_fraction(float(os.environ["LINGBOT_MEM_FRACTION"]), 0)

t0 = time.time()
server = pol.LingbotVLAv2Server(path_to_pi_model=args.model, robot_norm_path="unused", use_length=-1, chunk_ret=True,
                                use_bf16=False, use_fp32=True, use_compile=False)
CHUNK = server.config.chunk_size
print(f"[ab_eval] model loaded in {time.time() - t0:.0f}s chunk {CHUNK} model {args.model}", flush=True)


def set_robot(robot_cfg_path):
    """server.reset() without the cwd-relative configs/robot_configs path; norm stats come from the robot config."""
    server.global_step = 0
    server.last_action_chunk = None
    server.last_normalized_action_chunk = None
    ft = FeatureTransform(robot_cfg_path, server.data_config, server.config, server.processor,
                          chunk_size=CHUNK, norm_stats_path=None)
    server.vla.feature_transform = ft
    server.action_key = ft.org_features["actions"]
    return ft


def prepare(ft, item):
    """= official prepare_eval_observation, taking the feature transform directly (runs in loader workers)."""
    traj = dict(item)
    for k in ft.org_features["images"]:
        traj[k] = traj[k].to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    for k in ft.org_features["actions"]:
        if len(traj[k].shape) == 1:
            traj[k] = traj[k].unsqueeze(-1)
    for k in ft.org_features["states"]:
        if len(traj[k].shape) == 0:
            traj[k] = traj[k].unsqueeze(-1)
    org = {k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v) for k, v in traj.items()}
    old = ft.disabled_image_features
    ft.disabled_image_features = True
    try:
        org = ft.unapply(ft.apply(org))
    finally:
        ft.disabled_image_features = old
    gt = np.concatenate([np.asarray(org[k])[:CHUNK] for k in ft.org_features["actions"]], -1).astype(np.float32)
    st = np.concatenate([np.asarray(org[k]).reshape(-1) for k in ft.org_features["states"]], -1).astype(np.float32)
    return traj, gt, st


class Starts(torch.utils.data.Dataset):
    def __init__(self, ds, ft, idx):
        self.ds, self.ft, self.idx = ds, ft, idx
    def __len__(self):
        return len(self.idx)
    def __getitem__(self, i):
        return prepare(self.ft, self.ds[self.idx[i]])


os.makedirs(args.out, exist_ok=True)
lines = [l.split() for l in open(args.list) if l.strip()]
for robot, path in lines:
    name = path.rstrip("/").split("/")[-1]
    dst = os.path.join(args.out, name + ".npz")
    if os.path.exists(dst) and not args.dry:
        print(f"[ab_eval] skip {name} (done)", flush=True)
        continue
    ft = set_robot(os.path.join(args.robot_cfg, robot + ".yaml"))
    meta = LeRobotDatasetMetadata(name, root=path)
    dts = {k: [t / meta.fps for t in range(CHUNK)] for k in ft.org_features["actions"]}
    ds = LeRobotDataset(name, root=path, delta_timestamps=dts)
    if V3:
        eps_meta = meta.episodes
        ranges = [(int(eps_meta[i]["dataset_from_index"]), int(eps_meta[i]["dataset_to_index"])) for i in range(meta.total_episodes)]
    else:
        edi = ds.episode_data_index
        ranges = [(int(edi["from"][ep]), int(edi["to"][ep])) for ep in range(meta.total_episodes)]
    starts = []
    for ep in range(meta.total_episodes):
        a, b = ranges[ep]
        starts += [(ep, f - a, f) for f in range(a, b - CHUNK + 1, args.stride)]
    loader = torch.utils.data.DataLoader(Starts(ds, ft, [s[2] for s in starts]), batch_size=args.batch,
                                         num_workers=0 if args.dry else args.workers, collate_fn=list,
                                         drop_last=True)  # the deploy server keeps the first batch's image grid: a short last batch crashes
    preds, gts, sts = [], [], []
    t1 = time.time()
    for bi, items in enumerate(loader):
        trajs = [x[0] for x in items]
        gts += [x[1] for x in items]
        sts += [x[2] for x in items]
        if args.dry:
            obs = server._prepare_model_input(trajs[0])
            print("[dry]", name, "item keys", sorted(k for k in items[0][0].keys())[:12])
            for k in ft.org_features["images"]:
                print("[dry] image", k, items[0][0][k].shape, items[0][0][k].dtype, items[0][0][k].min(), items[0][0][k].max())
            print("[dry] gt", items[0][1].shape, "state", items[0][2].shape, "gt0-state max",
                  float(np.abs(items[0][1][0, :items[0][2].shape[0]] - items[0][2]).max()))
            print("[dry] model input", {k: (tuple(v.shape), str(v.dtype)) if torch.is_tensor(v) else type(v).__name__ for k, v in obs.items()})
            break
        seed = zlib.crc32(f"{name}/{bi}".encode())
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)
        with torch.no_grad():
            out = server.infer({"batch": trajs})
        preds.append(np.concatenate([np.asarray(out[k], dtype=np.float32) for k in server.action_key], -1))
        if bi % 20 == 0:
            print(f"[ab_eval] {name} batch {bi + 1}/{len(loader)} {(time.time() - t1) / (bi + 1):.2f}s/batch", flush=True)
        if args.max_batches and bi + 1 >= args.max_batches:
            break
    if args.dry:
        print(f"[dry] {name}: {len(starts)} starts over {meta.total_episodes} eps", flush=True)
        continue
    pred = np.concatenate(preds, 0)
    n = len(pred)
    gt = np.stack(gts[:n]); st = np.stack(sts[:n])
    assert pred.shape == gt.shape, (pred.shape, gt.shape)
    ep = np.array([s[0] for s in starts[:n]]); fr = np.array([s[1] for s in starts[:n]])
    np.savez(dst + ".tmp.npz", pred=pred, gt=gt, state=st, episode=ep, frame=fr, robot=robot)
    os.replace(dst + ".tmp.npz", dst)
    mae = np.abs(pred - gt).mean(axis=(0, 1))
    hold = np.abs(st[:, None, :] - gt).mean(axis=(0, 1))   # hold-still baseline: action = current state
    print(f"[ab_eval] {name}: hold-still baseline arm MAE {np.degrees(hold[:12].mean()):.2f} deg, gripper {hold[12:].mean():.4f}", flush=True)
    print(f"[ab_eval] {name}: {n} chunks in {time.time() - t1:.0f}s; arm MAE {np.degrees(mae[:12].mean()):.2f} deg, "
          f"gripper MAE {mae[12:].mean():.4f}", flush=True)
print(f"[ab_eval] done in {time.time() - t0:.0f}s", flush=True)
