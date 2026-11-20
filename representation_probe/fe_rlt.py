"""fe_rlt.py (11-18): feature extraction for the representation diagnosis (what does the RL token know about the
plug-to-socket offset in the precision phase?).

For selected frames of the insert-charge demos it runs the deployed LingBot-VLA v2 policy's PREFIX ONLY (no denoising)
and stores, per frame:
  z      [2048]           RL token = rlt_module.rl_token(final-layer prefix hidden, prefix mask)
  pre    [3, T, 2560]     final-layer prefix hidden states of each camera's image chunk (vision_start, patches, vision_end)
  vis    [3, n, 2560]     vision-tower output tokens per camera (input to the LLM)
  rest   [R, 2560]        final-layer hidden states of the language + query tokens
Cameras in robot-config order: camera_top, camera_wrist_left, camera_wrist_right.
Frames: per episode, the precision window [t_approach - 2 s, t*] with stride 3, plus context frames every 15 from
t_approach - 10 s (phases.json from phase scan).  Shard over GPUs by episode: --shard i --nshard k.
Writes OUT/feat_<shard>.npz (fp16 features + episode/frame/global index).
usage: fe_rlt.py --model HF_CKPT --rlt RLT_MODULE --ds DS --robot-cfg YAML --phases phases.json --out DIR --shard 0 --nshard 4
"""
import argparse, json, os, sys, time
import numpy as np, torch

ap = argparse.ArgumentParser()
for k in ("model", "rlt", "ds", "robot_cfg", "phases", "out"):
    ap.add_argument("--" + k.replace("_", "-"), dest=k, required=True)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshard", type=int, default=1)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--max-batches", type=int, default=0)
args = ap.parse_args()

sys.path.insert(0, os.environ["LINGBOT_CODE"])
import lingbotvla  # noqa: F401
import deploy.lingbot_vla_v2_policy as pol
import lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 as MV2
from lingbotvla.models.vla.lingbot_vla.rl_token import RLTokenModule
from lingbotvla.data.vla_data.base_dataset import LeRobotDataset
from lingbotvla.data.vla_data.utils import FeatureTransform
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

torch.set_float32_matmul_precision("high")
t0 = time.time()
server = pol.LingbotVLAv2Server(path_to_pi_model=args.model, robot_norm_path="unused", use_length=-1, chunk_ret=True,
                                use_bf16=False, use_fp32=True, use_compile=False)
CHUNK = server.config.chunk_size
model = server.vla.model
sd = torch.load(args.rlt, map_location="cpu", weights_only=False)
rlt = RLTokenModule(**sd["config"])
rlt.load_state_dict(sd["module"])
rlt = rlt.cuda().eval()
print(f"[fe] model + RLT (step {sd.get('global_step')}) loaded in {time.time() - t0:.0f}s, chunk {CHUNK}", flush=True)

CAP = {}
_orig_embed_image = model.qwenvl_with_expert.embed_image


def _embed_image_hook(*a, **k):
    out = _orig_embed_image(*a, **k)
    CAP["vis"] = out[0].detach()          # (b*n, num_patch, D)
    return out


model.qwenvl_with_expert.embed_image = _embed_image_hook


@torch.no_grad()
def prefix_only(images, img_masks, lang_tokens, lang_masks, state, noise=None, image_grid_thw=None, **_):
    (prefix_embs, prefix_pad_masks, prefix_att_masks, prefix_position_ids, visual_pos_masks,
     deepstack_visual_embeds) = model.embed_prefix(images, img_masks, lang_tokens, lang_masks, image_grid_thw=image_grid_thw)
    att = MV2.make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    outs, _, _ = model.qwenvl_with_expert.forward(
        attention_mask=att, position_ids=prefix_position_ids, vlm_position_ids=prefix_position_ids,
        past_key_values=None, inputs_embeds=[prefix_embs, None], use_cache=model.config.use_cache,
        fill_kv_cache=True, visual_pos_masks=visual_pos_masks, deepstack_visual_embeds=deepstack_visual_embeds)
    h = outs[0] if isinstance(outs, (list, tuple)) else outs
    CAP["prefix"] = h.detach()
    CAP["mask"] = prefix_pad_masks.detach()
    CAP["z"] = rlt.rl_token(h.float(), prefix_pad_masks)
    b = state.shape[0]
    return torch.zeros(b, model.config.n_action_steps, model.config.max_action_dim, device=state.device, dtype=state.dtype)


server.sample_actions_fn = prefix_only

ft = FeatureTransform(args.robot_cfg, server.data_config, server.config, server.processor, chunk_size=CHUNK,
                      norm_stats_path=None)
server.vla.feature_transform = ft
server.action_key = ft.org_features["actions"]
server.global_step = 0
server.last_action_chunk = None
server.last_normalized_action_chunk = None
CAMS = list(ft.org_features["images"])
print("[fe] cameras", CAMS, flush=True)


def prepare(item):
    traj = dict(item)
    for k in ft.org_features["images"]:
        traj[k] = traj[k].to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    for k in ft.org_features["actions"]:
        if len(traj[k].shape) == 1:
            traj[k] = traj[k].unsqueeze(-1)
    for k in ft.org_features["states"]:
        if len(traj[k].shape) == 0:
            traj[k] = traj[k].unsqueeze(-1)
    return traj


name = args.ds.rstrip("/").split("/")[-1]
meta = LeRobotDatasetMetadata(name, root=args.ds)
dts = {k: [t / meta.fps for t in range(CHUNK)] for k in ft.org_features["actions"]}
ds = LeRobotDataset(name, root=args.ds, delta_timestamps=dts)
eps_meta = meta.episodes
ph = {int(k): v for k, v in json.load(open(args.phases)).items() if v}
sel = []
for ep in sorted(ph):
    if ep % args.nshard != args.shard:
        continue
    a = int(eps_meta[ep]["dataset_from_index"])
    v = ph[ep]
    ts, ta = v["t_star"], v["t_approach"]
    fr = set(range(max(0, ta - 60), ts + 1, 3)) | set(range(max(0, ta - 300), max(0, ta - 60), 15))
    sel += [(ep, f, a + f) for f in sorted(fr)]
print(f"[fe] shard {args.shard}/{args.nshard}: {len(sel)} frames over {len({s[0] for s in sel})} episodes", flush=True)


class Sel(torch.utils.data.Dataset):
    def __len__(self):
        return len(sel)

    def __getitem__(self, i):
        return prepare(ds[sel[i][2]])


loader = torch.utils.data.DataLoader(Sel(), batch_size=args.batch, num_workers=args.workers, collate_fn=list)
Z, PRE, VIS, REST = [], [], [], []
layout = None
t1 = time.time()
n_done = 0
for bi, trajs in enumerate(loader):
    n_real = len(trajs)
    if n_real < args.batch:   # the deploy server keeps the first batch's image grid: pad the last batch
        trajs = trajs + [trajs[-1]] * (args.batch - n_real)
    server.infer({"batch": trajs})
    h, m, z, vis = CAP["prefix"], CAP["mask"], CAP["z"], CAP["vis"]
    nimg = len(CAMS)
    npatch = vis.shape[1]
    tl = npatch + 2
    if layout is None:
        layout = dict(prefix_len=int(h.shape[1]), n_images=nimg, num_patch=int(npatch), image_token_len=tl,
                      valid_tokens=int(m[0].sum()), cams=CAMS, dim=int(h.shape[-1]))
        print("[fe] layout", layout, flush=True)
    B = h.shape[0]
    pre = h[:, :nimg * tl].reshape(B, nimg, tl, -1)
    rest = h[:, nimg * tl:]
    vis = vis.reshape(B, nimg, npatch, -1)
    Z.append(z[:n_real].half().cpu().numpy())
    PRE.append(pre[:n_real].half().cpu().numpy())
    VIS.append(vis[:n_real].half().cpu().numpy())
    REST.append(rest[:n_real].half().cpu().numpy())
    n_done += n_real
    if bi % 20 == 0:
        print(f"[fe] batch {bi + 1}/{len(loader)} {(time.time() - t1) / (bi + 1):.2f}s/batch", flush=True)
    if args.max_batches and bi + 1 >= args.max_batches:
        break
os.makedirs(args.out, exist_ok=True)
s = np.array(sel[:n_done])
dst = os.path.join(args.out, f"feat_{args.shard}.npz")
np.savez(dst + ".tmp.npz", z=np.concatenate(Z), pre=np.concatenate(PRE), vis=np.concatenate(VIS),
         rest=np.concatenate(REST), episode=s[:, 0], frame=s[:, 1], gindex=s[:, 2], layout=json.dumps(layout),
         rlt_step=sd.get("global_step", -1), model=args.model)
os.replace(dst + ".tmp.npz", dst)
print(f"[fe] wrote {dst}: {n_done} frames in {time.time() - t1:.0f}s", flush=True)
