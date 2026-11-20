"""probe.py (11-18): how precisely can each representation read the plug-to-socket offset?

Label (hindsight, no annotation): the right TCP (flange + 0.135 m along flange +z = gripper axis, the insertion axis)
at frame t, expressed in the TCP frame at t* (insertion complete, just before the right gripper releases):
  p_rel(t) = R*^T (p(t) - p*)  [mm].  The power strip is static during the approach (held still by the left gripper or
lying on the table); frames where the left gripper holds the strip and the left TCP moved > 3 mm from its t* pose are
dropped.
Features (fe_rlt.py shards): z (RL token), final-layer prefix tokens per camera (pre_*), vision-tower tokens per camera
(vis_*), language+query tokens (rest), proprio (14-d state) and a zero baseline ("assume already at the target").
Probes: ridge (pooled features; lambda by inner validation), MLP (pooled), attention pooling over the 8x8 patch tokens.
5-fold cross-validation by episode; inner 20 % of the training episodes for model selection / early stopping.
usage: probe.py --feat DIR --ds DS --phases phases.json --out DIR --sets z,pre_wr,... --probes ridge,mlp,attn [--gpu 0]
"""
import argparse, glob, json, os, sys, time
import numpy as np, pandas as pd, torch, torch.nn as nn

ap = argparse.ArgumentParser()
ap.add_argument("--feat", required=True)
ap.add_argument("--ds", required=True)
ap.add_argument("--phases", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--sets", default="zero,proprio,z,pre_top,pre_wr,pre_wl,pre_all,vis_top,vis_wr,vis_wl,rest")
ap.add_argument("--tok-sets", default="pre_top,pre_wr,pre_wl,vis_wr")
ap.add_argument("--probes", default="ridge,mlp,attn")
ap.add_argument("--fk", default="/path/to/storage/lingbot-cobot/tools")
ap.add_argument("--tip", type=float, default=0.135)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tag", default="")
ap.add_argument("--near", type=float, default=0, help="if > 0: train and evaluate only on frames within this many mm of the target")
args = ap.parse_args()
sys.path.insert(0, args.fk)
from piper_fk import fk_sdk

dev = "cuda"
torch.manual_seed(args.seed); np.random.seed(args.seed)
t0 = time.time()
files = sorted(glob.glob(os.path.join(args.feat, "feat_*.npz")))
F = [dict(np.load(f)) for f in files]   # eager: every pooled/token set reads from RAM, not the npz again
layout = json.loads(str(F[0]["layout"]))
cams = [c.split(".")[-1] for c in layout["cams"]]
cam_key = {"camera_top": "top", "camera_wrist_right": "wr", "camera_wrist_left": "wl"}
ci = {cam_key[c]: i for i, c in enumerate(cams)}
ep = np.concatenate([f["episode"] for f in F]); fr = np.concatenate([f["frame"] for f in F])
print(f"[probe] {len(ep)} frames from {len(files)} shards, cams {cams}, loaded index in {time.time() - t0:.0f}s", flush=True)

# ---------------- labels ----------------
d = pd.read_parquet(glob.glob(os.path.join(args.ds, "data/*/*.parquet"))[0], columns=["episode_index", "frame_index", "observation.state"])
S_all = np.stack(d["observation.state"].values).astype(np.float64)
ep_all = d["episode_index"].values; fi_all = d["frame_index"].values
ph = {int(k): v for k, v in json.load(open(args.phases)).items() if v}
lab = np.zeros((len(ep), 3)); rot = np.zeros(len(ep)); valid = np.zeros(len(ep), bool); state = np.zeros((len(ep), 14))
for e in np.unique(ep):
    m = ep_all == e
    s = S_all[m]; fidx = fi_all[m]
    TR = fk_sdk(s[:, 7:13]); TL = fk_sdk(s[:, 0:6])
    tip = TR[:, :3, 3] + args.tip * TR[:, :3, 2]
    ts = ph[int(e)]["t_star"]
    Rs = TR[ts, :3, :3]
    sel = np.where(ep == e)[0]
    rows = np.searchsorted(fidx, fr[sel])
    lab[sel] = ((tip[rows] - tip[ts]) @ Rs) * 1000.0
    Rrel = np.einsum("ji,njk->nik", Rs, TR[rows, :3, :3])
    rot[sel] = np.degrees(np.arccos(np.clip((np.trace(Rrel, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    holding = s[ts, 6] > 0.03
    lmove = np.linalg.norm(TL[rows, :3, 3] - TL[ts, :3, 3], axis=1) * 1000.0
    valid[sel] = ~(holding & (lmove > 3.0))
    state[sel] = s[rows]
dist = np.linalg.norm(lab, axis=1)
print(f"[probe] labels: valid {valid.mean():.3f}; distance median {np.median(dist):.0f} mm, <100 mm {np.mean(dist < 100):.2f}, "
      f"<20 mm {np.mean(dist < 20):.2f}; rotation to target median {np.median(rot):.1f} deg", flush=True)

keep = valid & (dist < args.near) if args.near > 0 else valid
print(f"[probe] frames used: {keep.sum()} (near {args.near:g} mm)" if args.near > 0 else f"[probe] frames used: {keep.sum()}", flush=True)
eps_u = np.unique(ep[keep])
rng = np.random.RandomState(args.seed)
perm = rng.permutation(eps_u)
fold_of = {e: i % 5 for i, e in enumerate(perm)}
for e in np.unique(ep):            # episodes with no valid frame (strip moved by the left hand) still need a fold id
    fold_of.setdefault(e, -1)
fold = np.array([fold_of[e] for e in ep])


def pooled(name):
    if name == "zero":
        return None
    if name == "proprio":
        return state.astype(np.float32)
    if name == "z":
        return np.concatenate([f["z"] for f in F]).astype(np.float32)
    if name == "rest":
        return np.concatenate([f["rest"].astype(np.float32).mean(1) for f in F])
    kind, cam = name.split("_")
    if cam == "all":
        return np.concatenate([np.concatenate([f[kind][:, i, 1:-1].astype(np.float32).mean(1) if kind == "pre" else
                                               f[kind][:, i].astype(np.float32).mean(1) for f in F]) for i in range(3)], 1)
    i = ci[cam]
    return np.concatenate([(f[kind][:, i, 1:-1] if kind == "pre" else f[kind][:, i]).astype(np.float32).mean(1) for f in F])


def tokens(name):
    kind, cam = name.split("_")
    i = ci[cam]
    return np.concatenate([(f[kind][:, i, 1:-1] if kind == "pre" else f[kind][:, i]) for f in F])   # fp16 [N, 64, D]


Y = torch.tensor(lab, dtype=torch.float32, device=dev)


def split(k):
    te = (fold == k) & keep
    tr_eps = [e for e in eps_u if fold_of[e] != k]
    inner = set(np.random.RandomState(100 + k).permutation(tr_eps)[: max(1, len(tr_eps) // 5)])
    va = np.array([e in inner for e in ep]) & (fold != k) & keep
    tr = (fold != k) & keep & ~va
    return tr, va, te


def ridge(X):
    pred = np.zeros_like(lab)
    Xt = torch.tensor(X, device=dev, dtype=torch.float64)
    lams = [1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e5]
    for k in range(5):
        tr, va, te = split(k)
        mu, sd = Xt[tr].mean(0), Xt[tr].std(0) + 1e-6
        Xn = (Xt - mu) / sd
        ym = Y[tr].double().mean(0)

        def fit(idx, lam):
            A = Xn[idx]; n, p = A.shape
            if p <= n:
                W = torch.linalg.solve(A.T @ A + lam * n * torch.eye(p, device=dev, dtype=A.dtype), A.T @ (Y[idx].double() - ym))
            else:
                W = A.T @ torch.linalg.solve(A @ A.T + lam * n * torch.eye(n, device=dev, dtype=A.dtype), Y[idx].double() - ym)
            return W
        best = None
        for lam in lams:
            W = fit(tr, lam)
            err = torch.linalg.norm(Xn[va] @ W + ym - Y[va].double(), dim=1).median().item()
            if best is None or err < best[0]:
                best = (err, lam)
        W = fit(tr | va, best[1])
        pred[te] = (Xn[te] @ W + ym).float().cpu().numpy()
    return pred


class MLP(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(p, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 3))

    def forward(self, x):
        return self.net(x)


class AttnProbe(nn.Module):
    def __init__(self, D, n_tok=64, w=256):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, w))
        self.pos = nn.Parameter(torch.randn(1, n_tok, w) * 0.02)
        self.enc = nn.TransformerEncoderLayer(w, 4, 4 * w, dropout=0.1, batch_first=True, norm_first=True)
        self.q = nn.Parameter(torch.randn(1, 1, w) * 0.02)
        self.pool = nn.MultiheadAttention(w, 4, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, w), nn.GELU(), nn.Linear(w, 3))

    def forward(self, x):
        h = self.enc(self.inp(x) + self.pos)
        o, _ = self.pool(self.q.expand(len(x), -1, -1), h, h)
        return self.head(o[:, 0])


def train_nn(make, Xall, is_tok, epochs=60, lr=1e-3, bs=256):
    pred = np.zeros_like(lab)
    Xg = torch.tensor(Xall, device=dev) if not is_tok else torch.from_numpy(Xall).to(dev)   # tokens stay fp16 on GPU
    for k in range(5):
        tr, va, te = split(k)
        if is_tok:
            mu = Xg[tr].float().mean((0, 1)); sd = Xg[tr].float().std((0, 1)) + 1e-4
        else:
            mu = Xg[tr].float().mean(0); sd = Xg[tr].float().std(0) + 1e-6
        ym = Y[tr].mean(0); ys = Y[tr].std(0)
        net = make().to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        tri = np.where(tr)[0]; vai = np.where(va)[0]
        best = (1e18, None, 0)
        for epch in range(epochs):
            net.train()
            for b in np.array_split(np.random.permutation(tri), max(1, len(tri) // bs)):
                x = (Xg[b].float() - mu) / sd
                loss = nn.functional.smooth_l1_loss(net(x), (Y[b] - ym) / ys)
                opt.zero_grad(); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                pv = torch.cat([net((Xg[b].float() - mu) / sd) for b in np.array_split(vai, max(1, len(vai) // 1024))]) * ys + ym
                err = torch.linalg.norm(pv - Y[vai], dim=1).median().item()
            if err < best[0]:
                best = (err, {kk: v.detach().clone() for kk, v in net.state_dict().items()}, epch)
            elif epch - best[2] > 12:
                break
        net.load_state_dict(best[1]); net.eval()
        tei = np.where(te)[0]
        with torch.no_grad():
            pred[te] = (torch.cat([net((Xg[b].float() - mu) / sd) for b in np.array_split(tei, max(1, len(tei) // 1024))]) * ys + ym).cpu().numpy()
    return pred


BINS = [0, 2, 5, 10, 20, 50, 100, 1e9]


def report(name, pred):
    e = np.linalg.norm(pred - lab, axis=1)
    out = {"n": int(keep.sum()), "median_all": float(np.median(e[keep])),
           "median_lt100": float(np.median(e[keep & (dist < 100)]))}
    for a, b in zip(BINS[:-1], BINS[1:]):
        m = keep & (dist >= a) & (dist < b)
        out[f"bin_{a:g}_{b:g}"] = dict(n=int(m.sum()), median=float(np.median(e[m])) if m.any() else None,
                                      p90=float(np.percentile(e[m], 90)) if m.any() else None)
    line = " ".join(f"{a:g}-{b:g}:{out[f'bin_{a:g}_{b:g}']['median']:.1f}" for a, b in zip(BINS[:-1], BINS[1:]) if out[f'bin_{a:g}_{b:g}']['median'] is not None)
    print(f"[probe] {name:22s} median err mm  all {out['median_all']:6.1f} | <100mm {out['median_lt100']:6.1f} | by true distance {line}", flush=True)
    return out


os.makedirs(args.out, exist_ok=True)
results, preds = {}, {}
probes = args.probes.split(",")
for s in args.sets.split(","):
    if s == "zero":
        preds["zero"] = np.zeros_like(lab); results["zero"] = report("zero (at target)", preds["zero"]); continue
    X = pooled(s)
    for pr in probes:
        if pr == "attn":
            continue
        t1 = time.time()
        p = ridge(X) if pr == "ridge" else train_nn(lambda: MLP(X.shape[1]), X, False)
        key = f"{s}/{pr}"; preds[key] = p; results[key] = report(key, p); results[key]["sec"] = time.time() - t1
if "attn" in probes:
    for s in args.tok_sets.split(","):
        if not s:
            continue
        t1 = time.time()
        X = tokens(s)
        p = train_nn(lambda: AttnProbe(X.shape[-1], X.shape[1]), X, True, epochs=40, lr=3e-4)
        key = f"{s}/attn"; preds[key] = p; results[key] = report(key, p); results[key]["sec"] = time.time() - t1
        del X; torch.cuda.empty_cache()
tag = args.tag or "default"
json.dump(dict(results=results, layout=layout, tip=args.tip, n_frames=int(len(ep)), n_valid=int(keep.sum()),
               n_episodes=int(len(eps_u))), open(os.path.join(args.out, f"probe_{tag}.json"), "w"), indent=1)
np.savez(os.path.join(args.out, f"probe_{tag}_preds.npz"), lab=lab, dist=dist, rot=rot, valid=valid, episode=ep, frame=fr,
         fold=fold, **{k.replace("/", "__"): v for k, v in preds.items()})
print(f"[probe] done in {time.time() - t0:.0f}s -> {args.out}/probe_{tag}.json", flush=True)
