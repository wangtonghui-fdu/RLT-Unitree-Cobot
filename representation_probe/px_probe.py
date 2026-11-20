"""px_probe.py (11-18): pixel-ceiling control for the representation diagnosis (near-field frames from px_extract.py).
  --mode vt : LingBot-VLA v2's own (fine-tuned) Qwen3-VL vision tower on one camera at several input resolutions
              (256x256 = training setting -> 64 tokens; 448x448 -> 196; 480x640 native -> 300), attention probe on the
              output tokens.  Same preprocessing as training: Resize((h,w)) on the float image, then the Qwen image processor.
  --mode cnn: ResNet-18 trained from scratch end-to-end on the raw camera image (downscaled by --cnn-scale), the
              task-trained pixel ceiling.
Target: plug-to-socket offset p_rel [mm] (px_meta.npz).  5-fold cross-validation by episode, inner 20 % of training
episodes for early stopping.  Writes OUT/px_<tag>.json (+ per-frame predictions npz).
usage: px_probe.py --px DIR --out DIR --mode vt --cams wr,top --res 256x256,448x448,480x640 [--model HF --rlt X]
       px_probe.py --px DIR --out DIR --mode cnn --cams wr [--cnn-scale 0.5]
"""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn

ap = argparse.ArgumentParser()
ap.add_argument("--px", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--mode", choices=["vt", "cnn"], required=True)
ap.add_argument("--cams", default="wr")
ap.add_argument("--res", default="256x256,448x448,480x640")
ap.add_argument("--model", default="")
ap.add_argument("--cnn-scale", type=float, default=0.5)
ap.add_argument("--pretrained", default="", help="torchvision resnet18 ImageNet state dict (.pth); empty = from scratch")
ap.add_argument("--lr", type=float, default=0)
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--tag", default="")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
dev = "cuda"
torch.manual_seed(args.seed); np.random.seed(args.seed)
t0 = time.time()
M = np.load(os.path.join(args.px, "px_meta.npz"))
ep, lab, dist = M["episode"], M["lab"].astype(np.float32), M["dist"]
N = len(ep)
eps_u = np.unique(ep)
perm = np.random.RandomState(args.seed).permutation(eps_u)
fold_of = {e: i % 5 for i, e in enumerate(perm)}
fold = np.array([fold_of[e] for e in ep])
Y = torch.tensor(lab, device=dev)
BINS = [0, 2, 5, 10, 20, 30]
print(f"[px] {N} frames, {len(eps_u)} episodes, mode {args.mode}", flush=True)


def split(k):
    te = fold == k
    tr_eps = [e for e in eps_u if fold_of[e] != k]
    inner = set(np.random.RandomState(100 + k).permutation(tr_eps)[: max(1, len(tr_eps) // 5)])
    va = np.array([e in inner for e in ep]) & ~te
    return np.where(~te & ~va)[0], np.where(va)[0], np.where(te)[0]


def report(name, pred):
    e = np.linalg.norm(pred - lab, axis=1)
    out = {"n": int(N), "median_all": float(np.median(e)), "mean_all": float(e.mean())}
    for a, b in zip(BINS[:-1], BINS[1:]):
        m = (dist >= a) & (dist < b)
        out[f"bin_{a}_{b}"] = dict(n=int(m.sum()), median=float(np.median(e[m])), p90=float(np.percentile(e[m], 90)))
    print(f"[px] {name:24s} median err mm all {out['median_all']:5.1f} | " +
          " ".join(f"{a}-{b}:{out[f'bin_{a}_{b}']['median']:.1f}" for a, b in zip(BINS[:-1], BINS[1:])), flush=True)
    return out


def fit(make, get_x, epochs, lr, bs, aug=None, wd=1e-4):
    """Generic 5-fold trainer. get_x(idx) -> float tensor batch on GPU."""
    pred = np.zeros_like(lab)
    for k in range(5):
        tr, va, te = split(k)
        ym, ys = Y[tr].mean(0), Y[tr].std(0)
        net = make().to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        best = (1e18, None, 0)
        for epch in range(epochs):
            net.train()
            for b in np.array_split(np.random.permutation(tr), max(1, len(tr) // bs)):
                x = get_x(b)
                if aug is not None:
                    x = aug(x)
                loss = Fn.smooth_l1_loss(net(x), (Y[b] - ym) / ys)
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
            net.eval()
            with torch.no_grad():
                pv = torch.cat([net(get_x(b)) for b in np.array_split(va, max(1, len(va) // 256))]) * ys + ym
                err = torch.linalg.norm(pv - Y[va], dim=1).median().item()
            if err < best[0]:
                best = (err, {kk: v.detach().clone() for kk, v in net.state_dict().items()}, epch)
            elif epch - best[2] > 12:
                break
        net.load_state_dict(best[1]); net.eval()
        with torch.no_grad():
            pred[te] = (torch.cat([net(get_x(b)) for b in np.array_split(te, max(1, len(te) // 256))]) * ys + ym).cpu().numpy()
        print(f"[px]   fold {k}: best inner-val median {best[0]:.2f} mm at epoch {best[2]}", flush=True)
    return pred


class AttnProbe(nn.Module):
    def __init__(self, D, n_tok, w=256):
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


results, preds = {}, {}
os.makedirs(args.out, exist_ok=True)
if args.mode == "vt":
    sys.path.insert(0, os.environ["LINGBOT_CODE"])
    import lingbotvla  # noqa: F401
    import deploy.lingbot_vla_v2_policy as pol
    from torchvision.transforms import Resize
    torch.set_float32_matmul_precision("high")
    server = pol.LingbotVLAv2Server(path_to_pi_model=args.model, robot_norm_path="unused", use_length=-1, chunk_ret=True,
                                    use_bf16=False, use_fp32=True, use_compile=False)
    model = server.vla.model
    proc = server.processor.image_processor
    print(f"[px] vision tower loaded in {time.time() - t0:.0f}s", flush=True)
    for cam in args.cams.split(","):
        imgs = np.load(os.path.join(args.px, f"{cam}.npy"), mmap_mode="r")
        for res in args.res.split(","):
            h, w = (int(v) for v in res.split("x"))
            rs = Resize((h, w))
            toks = None
            t1 = time.time()
            with torch.no_grad():
                for s in range(0, N, 32):
                    xb = torch.from_numpy(np.ascontiguousarray(imgs[s:s + 32])).permute(0, 3, 1, 2).float()
                    pix, grid = [], []
                    for im in xb:
                        im = rs(im) if (h, w) != tuple(im.shape[1:]) else im
                        p = proc(im)
                        pix.append(p["pixel_values"]); grid.append(p["image_grid_thw"].reshape(-1, 3))
                    pix = torch.stack(pix).to(dev, torch.float32); grid = torch.cat(grid).to(dev)
                    # precompute_grid_thw caches pos-embeds / split sizes from the first call: recompute for this shape
                    model.qwenvl_with_expert.position_embeddings = None
                    emb = model.qwenvl_with_expert.embed_image(pix, grid)[0]
                    if toks is None:
                        toks = torch.empty((N,) + tuple(emb.shape[1:]), dtype=torch.float16, device=dev)
                    toks[s:s + len(emb)] = emb.half()
            print(f"[px] {cam} {res}: tokens {tuple(toks.shape)} in {time.time() - t1:.0f}s", flush=True)
            mu = toks.float().mean((0, 1)); sd = toks.float().std((0, 1)) + 1e-4
            p = fit(lambda: AttnProbe(toks.shape[-1], toks.shape[1]), lambda b: (toks[b].float() - mu) / sd,
                    epochs=40, lr=3e-4, bs=128)
            key = f"vt_{cam}_{res}"; preds[key] = p; results[key] = report(key, p)
            del toks; torch.cuda.empty_cache()
else:
    import torchvision
    for cam in args.cams.split(","):
        imgs = np.load(os.path.join(args.px, f"{cam}.npy"), mmap_mode="r")
        H, W = int(480 * args.cnn_scale), int(640 * args.cnn_scale)
        X = torch.empty((N, 3, H, W), dtype=torch.uint8, device=dev)
        for s in range(0, N, 256):
            xb = torch.from_numpy(np.ascontiguousarray(imgs[s:s + 256])).to(dev).permute(0, 3, 1, 2).float()
            X[s:s + len(xb)] = Fn.interpolate(xb, size=(H, W), mode="bilinear", antialias=True, align_corners=False).round().clamp(0, 255).to(torch.uint8)
        if args.pretrained:   # ImageNet statistics for an ImageNet backbone
            mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1) * 255
            std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1) * 255
        else:
            mean = X[:512].float().mean((0, 2, 3), keepdim=True); std = X[:512].float().std((0, 2, 3), keepdim=True)

        def get_x(b):
            return (X[b].float() - mean) / std

        def aug(x):   # photometric only: geometry (where things are in the wrist image) is the signal
            g = 1 + 0.2 * (torch.rand(len(x), 1, 1, 1, device=dev) - 0.5) * 2
            o = 0.2 * (torch.rand(len(x), 1, 1, 1, device=dev) - 0.5) * 2
            return x * g + o + 0.05 * torch.randn_like(x)

        def make():
            if args.pretrained:
                m = torchvision.models.resnet18(weights=None)
                m.load_state_dict(torch.load(args.pretrained, map_location="cpu"))
                m.fc = nn.Linear(512, 3)
                return m
            return torchvision.models.resnet18(weights=None, num_classes=3)
        t1 = time.time()
        lr = args.lr or (3e-4 if args.pretrained else 1e-3)
        p = fit(make, get_x, epochs=args.epochs, lr=lr, bs=32 if H >= 480 else 64, aug=aug)
        key = f"cnn{'_in1k' if args.pretrained else ''}_{cam}_{H}x{W}"
        preds[key] = p; results[key] = report(key, p); results[key]["sec"] = time.time() - t1
        del X; torch.cuda.empty_cache()
zero = np.zeros_like(lab); results["zero"] = report("zero (at target)", zero)
tag = args.tag or f"{args.mode}_{args.cams}"
json.dump(dict(results=results, n=int(N), episodes=int(len(eps_u))), open(os.path.join(args.out, f"px_{tag}.json"), "w"), indent=1)
np.savez(os.path.join(args.out, f"px_{tag}_preds.npz"), lab=lab, dist=dist, episode=ep, fold=fold, **preds)
print(f"[px] done in {time.time() - t0:.0f}s", flush=True)
