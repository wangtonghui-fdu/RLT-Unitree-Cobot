"""RL token (RLT, Xu et al. 2026, Physical Intelligence) stage 1 for LingBot-VLA v2, trained side by side with the main
VLA fine-tune without touching it (lingbot-cobot-ppu, 2026-11-10).

What it learns: a small encoder reads the VLA's final-layer prefix hidden states (3 camera images + language + the 8+8
learned depth/video query tokens, i.e. exactly the tokens the deployed policy also computes; LingBot's prefix never
attends to the action suffix) and squeezes them into one vector z_rl (the RL token) through a learned query token.
A small causal decoder must reconstruct the whole prefix from z_rl (teacher forcing on the previous prefix tokens,
most of which are masked out = context dropout, so the decoder cannot get around the bottleneck).

Design (merged from the paper, RLinf rlt_token_transformer.py and the atharva-pantheon port, plus our own changes):
- encoder 2 layers / decoder 2 layers, width 2048, 8 heads, SwiGLU 4x, RMSNorm pre-norm, learned positions (the
  LingBot prefix layout is fixed), z_rl = final-norm output at the appended query token, dim 2048 (paper: 1x2048).
- targets = stop-gradient VLA features, standardised per dimension with statistics frozen at the first micro-batch
  (all-reduced over ranks): the loss is then a fraction of variance unexplained, comparable across runs/checkpoints.
- loss = per-element MSE over valid prefix tokens, the constant first token (vision_start, sees only itself) excluded.
- context dropout 0.9 on the teacher-forced decoder inputs (replaced by a learned mask token): atharva reports z_rl is
  ignored without it. Diagnostics measure that directly: reconstruction from z_rl alone (all context masked), the same
  with z_rl shuffled across the batch, and the gap between the two (how much sample-specific information z_rl carries).

Zero interference with the main run (the point of "does not disturb the main line"):
- inputs are detached outputs of the compiled VLA forward: no gradient reaches the VLA;
- own AdamW, own grad clipping (the VLA grad norm / clipping is computed over model.parameters() only), own lr;
- own CUDA generator for every random draw, parameter init and every RLT step under fork_rng (SDPA kernels advance
  the global Philox offset even without dropout): the global RNG stream that drives the VLA's flow-matching
  noise/time sampling and the data order is untouched (unit-tested);
- parameters replicated on every rank, gradients averaged by one flat all-reduce per optimizer step.
Checkpoints: <global_step_N>/rlt/rlt_state.pt (module + optimizer + generator, for resume) and rlt_module.pt
(weights + config, for inference), written next to every DCP checkpoint so each VLA checkpoint has a matched encoder.

Env: LINGBOT_RLT=1 enables (read by modeling_lingbot_vla_v2 too). LINGBOT_RLT_CFG='{"lr":1e-4,...}' overrides
defaults below. LINGBOT_RLT_INIT=<rlt_module.pt|rlt_state.pt> initial weights when the run starts without a DCP that
carries RLT state.
"""
import json
import math
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

DEFAULTS = dict(
    in_dim=2560, dim=2048, enc_layers=2, dec_layers=2, heads=8, mlp_ratio=4.0, max_len=1024,
    ctx_dropout=0.9, lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01, warmup=200, clip=1.0,
    diag_every=10, seed=20261110, std_floor_frac=0.05,
)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.to(x.dtype) * self.weight.to(x.dtype)


class Block(nn.Module):
    def __init__(self, d, heads, mlp_ratio):
        super().__init__()
        self.heads = heads
        hidden = int(d * mlp_ratio)
        self.norm1 = RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.norm2 = RMSNorm(d)
        self.w12 = nn.Linear(d, 2 * hidden, bias=False)
        self.w3 = nn.Linear(hidden, d, bias=False)

    def forward(self, x, attn_mask):
        B, T, D = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(B, T, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        u, g = self.w12(self.norm2(x)).chunk(2, dim=-1)
        return x + self.w3(F.silu(g) * u)


class RLTokenModule(nn.Module):
    def __init__(self, in_dim=2560, dim=2048, enc_layers=2, dec_layers=2, heads=8, mlp_ratio=4.0, max_len=1024, **_):
        super().__init__()
        self.config = dict(in_dim=in_dim, dim=dim, enc_layers=enc_layers, dec_layers=dec_layers, heads=heads,
                           mlp_ratio=mlp_ratio, max_len=max_len)
        self.register_buffer("feat_mean", torch.zeros(in_dim))
        self.register_buffer("feat_std", torch.ones(in_dim))
        self.register_buffer("stats_ready", torch.zeros((), dtype=torch.bool))
        self.enc_in = nn.Linear(in_dim, dim)
        self.enc_pos = nn.Parameter(torch.randn(1, max_len + 1, dim) * 0.02)
        self.rl_query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.enc = nn.ModuleList([Block(dim, heads, mlp_ratio) for _ in range(enc_layers)])
        self.enc_norm = RMSNorm(dim)
        self.z_in = nn.Linear(dim, dim)
        self.dec_in = nn.Linear(in_dim, dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dec_pos = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)
        self.dec = nn.ModuleList([Block(dim, heads, mlp_ratio) for _ in range(dec_layers)])
        self.dec_norm = RMSNorm(dim)
        self.out = nn.Linear(dim, in_dim)

    def normalize(self, h):
        return (h.float() - self.feat_mean) / self.feat_std

    def encode(self, hn, valid):
        """hn: normalised prefix [B, P, in_dim]; valid: bool [B, P]. Returns z_rl [B, dim]."""
        B, P, _ = hn.shape
        x = self.enc_in(hn) + self.enc_pos[:, :P]
        x = torch.cat([x, self.rl_query.expand(B, -1, -1) + self.enc_pos[:, P:P + 1]], dim=1)
        keys = torch.cat([valid, valid.new_ones(B, 1)], dim=1)
        mask = keys[:, None, None, :]
        for blk in self.enc:
            x = blk(x, mask)
        return self.enc_norm(x[:, -1])

    def decode(self, z, hn, valid, ctx_keep):
        """Position t predicts prefix token t; its input is z_rl at t=0, else prefix token t-1 (or the mask token
        where ctx_keep is False)."""
        B, P, _ = hn.shape
        ctx = self.dec_in(hn[:, :-1])
        ctx = torch.where(ctx_keep[..., None], ctx, self.mask_token.to(ctx.dtype))
        x = torch.cat([self.z_in(z)[:, None], ctx], dim=1) + self.dec_pos[:, :P]
        causal = torch.ones(P, P, dtype=torch.bool, device=hn.device).tril()
        key_valid = torch.cat([valid.new_ones(B, 1), valid[:, :-1]], dim=1)
        mask = causal[None, None] & key_valid[:, None, None, :]
        for blk in self.dec:
            x = blk(x, mask)
        return self.out(self.dec_norm(x))

    @staticmethod
    def recon_loss(pred, hn, loss_mask):
        err = (pred.float() - hn.float()).pow(2).mean(-1)
        return (err * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)

    @torch.no_grad()
    def rl_token(self, prefix_hidden, valid):
        """Inference: raw VLA prefix hidden states (final norm output) -> z_rl."""
        return self.encode(self.normalize(prefix_hidden), valid.bool())


class RLTTrainer:
    def __init__(self, device, rank, world, **overrides):
        self.cfg = dict(DEFAULTS)
        env_cfg = os.environ.get("LINGBOT_RLT_CFG", "")
        if env_cfg:
            self.cfg.update(json.loads(env_cfg))
        self.cfg.update(overrides)
        c = self.cfg
        self.device, self.rank, self.world = device, rank, world
        # torch.manual_seed also reseeds every CUDA device: fork CPU + all devices so the init leaves no trace
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(c["seed"])
            module = RLTokenModule(**c)
        self.module = module.to(device)
        self.params = [p for p in self.module.parameters() if p.requires_grad]
        decay = [p for p in self.params if p.ndim >= 2]
        no_decay = [p for p in self.params if p.ndim < 2]
        self.opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": c["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}],
            lr=c["lr"], betas=tuple(c["betas"]), eps=1e-8)
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(c["seed"] + 1000 + rank)
        self.n_updates = 0
        self._reset_acc()
        self._diag_now = False
        n = sum(p.numel() for p in self.params)
        if rank == 0:
            print(f"[RLT] enabled: {n / 1e6:.1f} M params, cfg {json.dumps(c)}", flush=True)

    def _reset_acc(self):
        self.acc = {"loss": torch.zeros((), device=self.device)}
        self.diag = {}
        self.t_rlt = 0.0
        self.n_tokens = 0
        self.prefix_len = 0

    @torch.no_grad()
    def _init_stats(self, h, valid):
        m = valid.reshape(-1).float()
        x = h.reshape(-1, h.shape[-1]).float()
        s = torch.stack([(x * m[:, None]).sum(0), (x.pow(2) * m[:, None]).sum(0)])
        cnt = m.sum().reshape(1)
        dist.all_reduce(s)
        dist.all_reduce(cnt)
        mean = s[0] / cnt
        std = (s[1] / cnt - mean.pow(2)).clamp(min=0).sqrt()
        std = std.clamp(min=1e-4 + self.cfg["std_floor_frac"] * std.median())
        self.module.feat_mean.copy_(mean)
        self.module.feat_std.copy_(std)
        self.module.stats_ready.fill_(True)
        if self.rank == 0:
            print(f"[RLT] feature stats from {int(cnt.item())} tokens: |mean| {mean.abs().mean():.4f}, "
                  f"std median {std.median():.4f} min {std.min():.4f} max {std.max():.4f}", flush=True)

    def micro_step(self, loss_log, n_micro):
        # every RLT kernel runs inside fork_rng: SDPA kernels advance the global CUDA Philox offset even without
        # dropout (unit test 11-10), which would shift the VLA's flow-matching noise/time draws.
        with torch.random.fork_rng(devices=[self.device]):
            self._micro_step(loss_log, n_micro)

    def _micro_step(self, loss_log, n_micro):
        h = loss_log.pop("_rlt_prefix", None)
        valid = loss_log.pop("_rlt_mask", None)
        if h is None:
            raise RuntimeError("LINGBOT_RLT=1 but the model did not export _rlt_prefix (modeling patch missing?)")
        t0 = time.time()
        h = h.detach()
        valid = valid.detach().bool()
        B, P, _ = h.shape
        assert P <= self.cfg["max_len"], (P, self.cfg["max_len"])
        if not bool(self.module.stats_ready):
            self._init_stats(h, valid)
        loss_mask = valid.clone()
        loss_mask[:, 0] = False
        loss_mask = loss_mask.float()
        keep = torch.rand(B, P - 1, device=h.device, generator=self.gen) >= self.cfg["ctx_dropout"]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hn = self.module.normalize(h)
            z = self.module.encode(hn, valid)
            pred = self.module.decode(z, hn, valid, keep)
            loss = RLTokenModule.recon_loss(pred, hn, loss_mask)
        (loss / n_micro).backward()
        self.acc["loss"] += loss.detach() / n_micro
        self.n_tokens += int(loss_mask.sum().item())
        self.prefix_len = P
        if self._diag_now and not self.diag and B >= 2:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                none = torch.zeros_like(keep)
                full = torch.ones_like(keep)
                perm = torch.roll(torch.arange(B, device=h.device), 1)
                self.diag = {
                    "z_only": RLTokenModule.recon_loss(self.module.decode(z, hn, valid, none), hn, loss_mask),
                    "z_only_shuf": RLTokenModule.recon_loss(self.module.decode(z[perm], hn, valid, none), hn, loss_mask),
                    "full_ctx": RLTokenModule.recon_loss(self.module.decode(z, hn, valid, full), hn, loss_mask),
                    "full_ctx_shuf": RLTokenModule.recon_loss(self.module.decode(z[perm], hn, valid, full), hn, loss_mask),
                    "z_norm": z.float().norm(dim=-1).mean(),
                    "z_batch_std": z.float().std(dim=0).mean(),
                }
        torch.cuda.synchronize()
        self.t_rlt += time.time() - t0

    def begin_step(self, global_step):
        self._diag_now = (global_step % self.cfg["diag_every"] == 0) or global_step <= 3

    def opt_step(self, global_step):
        with torch.random.fork_rng(devices=[self.device]):
            return self._opt_step(global_step)

    def _opt_step(self, global_step):
        t0 = time.time()
        grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in self.params]
        flat = torch._utils._flatten_dense_tensors(grads)
        dist.all_reduce(flat)
        flat.div_(self.world)
        for g, s in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
            g.copy_(s)
        for p, g in zip(self.params, grads):
            p.grad = g
        gn = torch.nn.utils.clip_grad_norm_(self.params, self.cfg["clip"])
        self.n_updates += 1
        lr = self.cfg["lr"] * min(1.0, self.n_updates / max(1, self.cfg["warmup"]))
        for g in self.opt.param_groups:
            g["lr"] = lr
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        keys = ["loss"] + sorted(self.diag)
        vals = torch.stack([self.acc["loss"]] + [self.diag[k].float() for k in sorted(self.diag)])
        dist.all_reduce(vals)
        vals = (vals / self.world).tolist()
        out = dict(zip(keys, vals))
        if "z_only" in out:
            out["z_gap"] = out["z_only_shuf"] - out["z_only"]
            out["ctx_gap"] = out["full_ctx_shuf"] - out["full_ctx"]
        out.update(grad_norm=float(gn), lr=lr, time=self.t_rlt + time.time() - t0, prefix_len=self.prefix_len,
                   tokens=self.n_tokens)
        self._reset_acc()
        return out

    def log_line(self, step, m):
        s = f"[RLT] step {step} loss {m['loss']:.4f} gn {m['grad_norm']:.3f} lr {m['lr']:.2e} t {m['time']:.2f}s P {m['prefix_len']}"
        if "z_gap" in m:
            s += (f" | z_only {m['z_only']:.4f} shuf {m['z_only_shuf']:.4f} gap {m['z_gap']:.4f}"
                  f" | full_ctx {m['full_ctx']:.4f} shuf {m['full_ctx_shuf']:.4f} gap {m['ctx_gap']:.4f}"
                  f" | |z| {m['z_norm']:.2f} z_std {m['z_batch_std']:.3f}")
        return s

    def save(self, ckpt_dir, global_step):
        out = os.path.join(ckpt_dir, "rlt")
        if self.rank == 0:
            os.makedirs(out, exist_ok=True)
            torch.save({"module": self.module.state_dict(), "opt": self.opt.state_dict(), "gen": self.gen.get_state(),
                        "n_updates": self.n_updates, "global_step": global_step, "cfg": self.cfg},
                       os.path.join(out, "rlt_state.pt.tmp"))
            os.replace(os.path.join(out, "rlt_state.pt.tmp"), os.path.join(out, "rlt_state.pt"))
            torch.save({"module": self.module.state_dict(), "config": self.module.config, "global_step": global_step,
                        "note": "z_rl = RLTokenModule(**config).rl_token(vla_prefix_final_norm_hidden, prefix_pad_mask)"},
                       os.path.join(out, "rlt_module.pt"))
            print(f"[RLT] saved {out} (update {self.n_updates})", flush=True)
        dist.barrier()

    def maybe_load(self, loaded_ckpt):
        src = None
        if loaded_ckpt and os.path.isfile(os.path.join(loaded_ckpt, "rlt", "rlt_state.pt")):
            src = os.path.join(loaded_ckpt, "rlt", "rlt_state.pt")
        elif os.environ.get("LINGBOT_RLT_INIT"):
            src = os.environ["LINGBOT_RLT_INIT"]
        if src is None:
            if self.rank == 0:
                print(f"[RLT] fresh init (no RLT state in {loaded_ckpt})", flush=True)
            return
        sd = torch.load(src, map_location=self.device, weights_only=False)
        self.module.load_state_dict(sd["module"])
        if "opt" in sd:
            self.opt.load_state_dict(sd["opt"])
            self.n_updates = sd["n_updates"]
            # per-rank dropout stream continues from a seed derived from the update count (rank 0's saved
            # generator state would not fit the other ranks anyway)
            self.gen.manual_seed(self.cfg["seed"] + 1000 + self.rank + 7919 * self.n_updates)
        if self.rank == 0:
            print(f"[RLT] loaded {src} (updates {self.n_updates})", flush=True)
