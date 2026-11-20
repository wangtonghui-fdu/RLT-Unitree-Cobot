"""build_teacher_cache.py CLI_YAML FRAME_CACHE_DIR OUT_DIR SHARD NSHARDS [--bs 16] [--cam KEY] [--limit N]   (one process per GPU)
Teacher targets for every frame of the first camera, through the SAME functions and settings the trainer uses
(module_utils.get_depth_target / get_video_target, bf16 autocast, compiled depth models when train.use_compile):
  depth target of frame g, and the future-video target of the pair (g, g_f) with g_f = min(g + chunk_size - 1, last frame of g's episode)
keyed by pixel fingerprints (lingbotvla/utils/teacher_cache.py).  Shard files only; finish with --finalize OUT_DIR NSHARDS."""
import sys, os, json, time, argparse
import numpy as np, torch, yaml

if sys.argv[1] == "--finalize":
    out, K = sys.argv[2], int(sys.argv[3]); S = [json.load(open(f"{out}/shard_{k}.json")) for k in range(K)]
    assert all(s["depth_shape"] == S[0]["depth_shape"] and s["video_shape"] == S[0]["video_shape"] and s["vcur_shape"] == S[0]["vcur_shape"] for s in S)
    tot = sum(s["rows"] for s in S); assert tot == S[0]["total_frames"], (tot, S[0]["total_frames"])
    meta = dict(complete=True, shard_rows=[s["rows"] for s in S], depth_shape=S[0]["depth_shape"], video_shape=S[0]["video_shape"], vcur_shape=S[0]["vcur_shape"], video_flags=S[0]["video_flags"], total_frames=tot,
                built=time.strftime("%F %T"), seconds=[s["seconds"] for s in S], **{k: S[0][k] for k in ("camera", "offset", "fps", "effective_fps", "fps_mode", "frame_cache", "cli_yaml", "bs", "compiled")})
    json.dump(meta, open(f"{out}/meta.json", "w"), indent=1); print("FINALIZED", tot, "frames", meta["depth_shape"], meta["video_shape"]); sys.exit()

ap = argparse.ArgumentParser(); [ap.add_argument(k) for k in ("cli_yaml", "frame_cache", "out", "shard", "nshards")]
ap.add_argument("--bs", type=int, default=16); ap.add_argument("--cam", default=""); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--fps_mode", default="none")   # none = like the trainer (the batch carries no fps -> teacher default); dataset = fps / (chunk - 1)
a = ap.parse_args(); shard, K = int(a.shard), int(a.nshards); os.makedirs(a.out, exist_ok=True)
cfg = yaml.safe_load(open(a.cli_yaml)); align = cfg["train"]["align_params"]; chunk = int(cfg["train"].get("chunk_size", 50)); use_compile = bool(cfg["train"].get("use_compile", False))
fm = json.load(open(os.path.join(a.frame_cache, "cache_meta.json"))); cam = a.cam or list(fm["cameras"].keys())[0]
F = np.load(os.path.join(a.frame_cache, fm["cameras"][cam]), mmap_mode="r"); N = len(F); fps = float(fm["fps"])
ep_from = np.sort(np.array(fm["episode_from"], dtype=np.int64)); ep_end = np.r_[ep_from[1:], N]
last_of = np.empty(N, np.int64)
for s_, e_ in zip(ep_from, ep_end): last_of[s_:e_] = e_ - 1
fut_of = np.minimum(np.arange(N) + (chunk - 1), last_of)
lo, hi = N * shard // K, N * (shard + 1) // K
if a.limit: hi = min(hi, lo + a.limit)
print(f"shard {shard}/{K}: frames {lo}..{hi - 1} of {N}, camera {cam}, frame {F.shape[1:]}, chunk {chunk}, fps {fps}, compile {use_compile}", flush=True)

from lingbotvla.models.vla.vision_models.module_utils import build_depth_model, build_video_model, get_depth_target, get_video_target
from lingbotvla.utils.teacher_cache import Fingerprint
torch.backends.cuda.matmul.allow_tf32 = True; torch.set_float32_matmul_precision("high")
moge, morgbd = build_depth_model(align)
if use_compile: moge, morgbd = torch.compile(moge), torch.compile(morgbd)
video_teacher = build_video_model(align["video"]); fp = Fingerprint(); dev = torch.device("cuda")
eff = fps / float(max(1, chunk - 1)); n = hi - lo; dmm = vmm = cmm = None; dkeys = np.zeros((n, 2), np.int64); vkeys = np.zeros((n, 4), np.int64); t0 = time.time()
for b0 in range(lo, hi, a.bs):
    idx = np.arange(b0, min(b0 + a.bs, hi)); m = len(idx)
    if m < a.bs: idx = np.r_[idx, np.full(a.bs - m, idx[-1])]                  # keep the batch shape (compiled graphs, no squeeze surprises)
    cur = torch.from_numpy(np.ascontiguousarray(F[idx])).to(dev).unsqueeze(1)          # [B, 1, 3, H, W] uint8
    fut = torch.from_numpy(np.ascontiguousarray(F[fut_of[idx]])).to(dev).unsqueeze(1)
    efps = torch.full((len(idx),), eff, dtype=torch.float32, device=dev)
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            d, _ = get_depth_target("MoRGBD", (moge, morgbd), cur)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = get_video_target(video_teacher, cur, fut, align["video"], effective_fps=(efps if a.fps_mode == "dataset" else None))
    assert isinstance(v, dict) and v.get("cls") is None and torch.is_tensor(v.get("patch")) and torch.is_tensor(v.get("current_patch")), "this cache holds: patch + current_patch, no cls"
    vc = v["current_patch"]; v = v["patch"]
    assert d.dtype == torch.bfloat16 and v.dtype == torch.bfloat16 and vc.dtype == torch.bfloat16, (d.dtype, v.dtype, vc.dtype)
    if dmm is None:
        dshape, vshape, cshape = list(d.shape[1:]), list(v.shape[1:]), list(vc.shape[1:])
        dmm = np.memmap(f"{a.out}/depth_{shard}.bin", dtype=np.int16, mode="w+", shape=(n, *dshape)); vmm = np.memmap(f"{a.out}/video_{shard}.bin", dtype=np.int16, mode="w+", shape=(n, *vshape))
        cmm = np.memmap(f"{a.out}/vcur_{shard}.bin", dtype=np.int16, mode="w+", shape=(n, *cshape))
        print("target shapes: depth", dshape, "video", vshape, "video current", cshape, flush=True)
    r0 = b0 - lo; kc, kf = fp(cur[:, 0]), fp(fut[:, 0])
    dmm[r0:r0 + m] = d[:m].contiguous().view(torch.int16).cpu().numpy(); vmm[r0:r0 + m] = v[:m].contiguous().view(torch.int16).cpu().numpy(); cmm[r0:r0 + m] = vc[:m].contiguous().view(torch.int16).cpu().numpy()
    dkeys[r0:r0 + m] = kc[:m].cpu().numpy(); vkeys[r0:r0 + m] = torch.cat([kc, kf], 1)[:m].cpu().numpy()
    if (r0 // a.bs) % 100 == 0: print(f"  {r0 + m}/{n} rows, {time.time() - t0:.0f} s", flush=True)
dmm.flush(); vmm.flush(); cmm.flush(); np.save(f"{a.out}/depth_keys_{shard}.npy", dkeys); np.save(f"{a.out}/video_keys_{shard}.npy", vkeys)
vcfg = align["video"]
json.dump(dict(rows=n, depth_shape=dshape, video_shape=vshape, vcur_shape=cshape, video_flags=[bool(vcfg.get("use_patch_loss", True)), bool(vcfg.get("use_cls_loss", False)), bool(vcfg.get("use_current_patch_loss", False)), bool(vcfg.get("use_warmup_frame", False)), int(vcfg.get("num_future_frames", 1)), int(vcfg.get("input_size", 256))], total_frames=(N if not a.limit else None), camera=cam, offset=chunk - 1, fps=fps, effective_fps=(eff if a.fps_mode == "dataset" else None), fps_mode=a.fps_mode, frame_cache=os.path.realpath(a.frame_cache),
               cli_yaml=a.cli_yaml, bs=a.bs, compiled=use_compile, seconds=round(time.time() - t0, 1), unique_depth_keys=len(set(map(tuple, dkeys.tolist())))), open(f"{a.out}/shard_{shard}.json", "w"))
print(f"SHARD_DONE {shard}: {n} rows in {time.time() - t0:.0f} s", flush=True)
