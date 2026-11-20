"""teacher_cache.py (11-20): content-addressed cache of the two frozen teachers' outputs (depth tokens, future-video tokens).

The depth teacher's target is a pure function of ONE uint8 image (first camera) and the video teacher's target a pure function
of the (current, future) image pair (plus the constant effective fps), and the training pipeline feeds them the frames
unchanged (image_augment: false, frame cache).  So instead of tracking dataset indices through the data pipeline the cache is
keyed by a 128-bit fingerprint of the input pixels: a wrong assumption in the builder (camera order, future offset) can only
cause a MISS (-> the trainer computes that micro-batch online, as before), never a wrong target.

Layout of a cache dir (written by build_teacher_cache.py, K shards = one per build process):
  meta.json                     shapes, dtypes, shard row counts, builder info
  depth_keys_k.npy  [n_k, 2] int64     fingerprint of the image            depth_k.bin  [n_k, T_d, C_d] bf16 stored as int16
  video_keys_k.npy  [n_k, 4] int64     fingerprints of (current, future)   video_k.bin  [n_k, T_v, C_v] bf16 stored as int16
                                                                          vcur_k.bin   [n_k, T_c, C_c] (current-frame patch target of the same clip)
Use: LINGBOT_TEACHER_CACHE=<dir>; LINGBOT_TEACHER_CACHE_VERIFY=N also computes the first N looked-up micro-batches online and
prints the difference (rank 0)."""
import json, os
import numpy as np
import torch

_SEED = 20261120


def _weights(n, device):
    g = torch.Generator().manual_seed(_SEED)
    w = torch.randint(1, 2 ** 31 - 1, (2, n), generator=g, dtype=torch.int64)
    return w.to(device)


class Fingerprint:
    """128-bit fingerprint of uint8 images [B, C, H, W]: two wrap-around int64 dot products with fixed random weights."""

    def __init__(self):
        self._w = {}

    def __call__(self, img):
        if img.dtype != torch.uint8:
            raise TypeError(f"teacher cache expects uint8 teacher inputs, got {img.dtype}")
        x = img.reshape(img.shape[0], -1).to(torch.int64)
        key = (x.shape[1], str(x.device))
        if key not in self._w:
            self._w[key] = _weights(x.shape[1], x.device)
        w = self._w[key]
        return torch.stack([(x * w[0]).sum(1), (x * w[1]).sum(1)], 1)        # [B, 2] int64


class TeacherCache:
    def __init__(self, cache_dir, rank=0, video_cfg=None):
        self.dir = cache_dir; self.rank = rank
        self.meta = json.load(open(os.path.join(cache_dir, "meta.json")))
        if not self.meta.get("complete"):
            raise RuntimeError(f"teacher cache {cache_dir} is not complete")
        self.fp = Fingerprint(); self.depth_idx, self.video_idx = {}, {}; self.depth_arr, self.video_arr, self.vcur_arr = [], [], []
        td, cd = self.meta["depth_shape"]; tv, cv = self.meta["video_shape"]; tc_, cc_ = self.meta["vcur_shape"]
        for k, n in enumerate(self.meta["shard_rows"]):
            dk = np.load(os.path.join(cache_dir, f"depth_keys_{k}.npy")); vk = np.load(os.path.join(cache_dir, f"video_keys_{k}.npy"))
            assert len(dk) == n and len(vk) == n
            self.depth_arr.append(np.memmap(os.path.join(cache_dir, f"depth_{k}.bin"), dtype=np.int16, mode="r", shape=(n, td, cd)))
            self.video_arr.append(np.memmap(os.path.join(cache_dir, f"video_{k}.bin"), dtype=np.int16, mode="r", shape=(n, tv, cv)))
            self.vcur_arr.append(np.memmap(os.path.join(cache_dir, f"vcur_{k}.bin"), dtype=np.int16, mode="r", shape=(n, tc_, cc_)))
            for r, key in enumerate(map(tuple, dk.tolist())): self.depth_idx[key] = (k, r)
            for r, key in enumerate(map(tuple, vk.tolist())): self.video_idx[key] = (k, r)
        self.n_hit = self.n_miss = 0; self.broken = False
        if video_cfg is not None:                                          # the cache holds exactly: patch + current patch, no cls target
            want = (bool(video_cfg.get("use_patch_loss", True)), bool(video_cfg.get("use_cls_loss", False)), bool(video_cfg.get("use_current_patch_loss", False)),
                    bool(video_cfg.get("use_warmup_frame", False)), int(video_cfg.get("num_future_frames", 1)), int(video_cfg.get("input_size", 256)))
            have = tuple(self.meta["video_flags"])
            if want != have:
                self.broken = True
                if rank == 0: print(f"[TCACHE] DISABLED: built for video flags {have}, this run uses {want}", flush=True)
        self.verify_left = int(os.environ.get("LINGBOT_TEACHER_CACHE_VERIFY", "0") or 0)
        if rank == 0:
            print(f"[TCACHE] {cache_dir}: {len(self.depth_idx)} depth keys, {len(self.video_idx)} video keys, depth {td}x{cd}, video {tv}x{cv}", flush=True)

    @staticmethod
    def _rows(arrs, locs, device):
        out = np.stack([arrs[k][r] for k, r in locs], 0)                       # [B, T, C] int16 (bf16 bits)
        return torch.from_numpy(out).to(device, non_blocking=True).view(torch.bfloat16)

    def lookup(self, pil_images, future_pil_images, effective_fps=None):
        """-> (depth_targets, future_depth_targets, future_video_targets, future_video_current_patch) or None on any miss."""
        if self.broken:
            return None
        if (effective_fps is None) != (self.meta.get("fps_mode", "none") == "none"):      # built without an fps argument <-> the batch must carry none
            self.broken = True
            if self.rank == 0:
                print(f"[TCACHE] DISABLED: cache fps_mode {self.meta.get('fps_mode')} but this run passes effective_fps={'None' if effective_fps is None else 'a tensor'}", flush=True)
            return None
        try:
            cur, fut = pil_images[:, 0], future_pil_images[:, 0]
            kc, kf = self.fp(cur).tolist(), self.fp(fut).tolist()
        except Exception as e:                                              # never take the training down: switch the cache off
            self.broken = True
            if self.rank == 0:
                print(f"[TCACHE] DISABLED (teachers run online from now on): {type(e).__name__}: {e}", flush=True)
            return None
        try:
            dc = [self.depth_idx[tuple(a)] for a in kc]; df = [self.depth_idx[tuple(a)] for a in kf]
            vv = [self.video_idx[tuple(a) + tuple(b)] for a, b in zip(kc, kf)]
        except KeyError:
            self.n_miss += 1
            if self.rank == 0 and self.n_miss in (1, 10, 100, 1000):
                print(f"[TCACHE] miss #{self.n_miss} (hits {self.n_hit}) -> teachers run online for this micro-batch", flush=True)
            return None
        self.n_hit += 1
        dev = pil_images.device
        return self._rows(self.depth_arr, dc, dev), self._rows(self.depth_arr, df, dev), self._rows(self.video_arr, vv, dev), self._rows(self.vcur_arr, vv, dev)

    def report(self, cached, online):
        """LINGBOT_TEACHER_CACHE_VERIFY: cached vs online targets of one micro-batch."""
        self.verify_left -= 1
        if self.rank != 0:
            return
        parts = []
        for name, a, b in zip(("depth", "future_depth", "future_video", "video_current"), cached, online):
            a32, b32 = a.float(), b.float(); d = (a32 - b32).abs()
            cos = torch.nn.functional.cosine_similarity(a32.flatten(1), b32.flatten(1), dim=1).min().item()
            parts.append(f"{name}: shape {tuple(a.shape)} vs {tuple(b.shape)} max|d| {d.max().item():.4g} mean|d| {d.mean().item():.3g} "
                         f"mean|x| {b32.abs().mean().item():.3g} identical {bool((a32 == b32).all())} min cos {cos:.6f}")
        print("[TCACHE-VERIFY] " + " || ".join(parts), flush=True)
