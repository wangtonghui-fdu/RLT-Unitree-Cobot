"""lingbot-cobot-ppu: patch official lingbot-vla-v2 be969b8 for the PPU run + RL token side training. Idempotent,
every anchor asserted. usage: patch_ppu_rlt.py <code_root> <cobot_code_root> <rl_token.py>
1. video_utils.py / multi_vla_dataset.py from the MetaX copy (PyAV direct decode when torchcodec is missing, which is
   the case in proj/.venv; first 3 sample-load tracebacks logged). Only those two hunks differ from official.
2. lingbotvla/models/vla/lingbot_vla/rl_token.py (new).
3. modeling_lingbot_vla_v2.py: with LINGBOT_RLT=1 the training forward also returns the detached final-norm prefix
   hidden states and the prefix pad mask in its metrics dict (keys _rlt_prefix/_rlt_mask). Off: unchanged graph.
4. tasks/vla/train_lingbotvla.py: RLTTrainer hooks (build, resume, per-micro-batch step after the VLA backward, own
   optimizer step after the VLA step, log line + tensorboard rlt/*, save next to every DCP checkpoint).
5. torch 2.6 / compile fixes carried over from the October PPU copy (lingbot-clean2500/proj/lingbot-vla-v2, which trained
   compiled for weeks): flex kernel options without num_warps/num_stages (inductor 2.6 passes them itself -> "multiple
   values for keyword argument 'num_warps'"); Qwen3-VL deepstack add as a fixed-size gather + where (same values, no
   boolean-index output under compile); dataset non-video keys read via select_columns (same values, reads one column).
"""
import os
import re
import shutil
import sys

root, cobot_root, rlt_src = sys.argv[1:4]


def rd(p):
    return open(p).read()


def wr(p, s):
    open(p, "w").write(s)


def sub_once(s, old, new, tag):
    if tag in s:
        return s, False
    assert s.count(old) == 1, (tag, s.count(old))
    return s.replace(old, new), True


# 1
for rel in ["lingbotvla/data/vla_data/video_utils.py", "lingbotvla/data/vla_data/multi_vla_dataset.py"]:
    src, dst = os.path.join(cobot_root, rel), os.path.join(root, rel)
    if rd(src) != rd(dst):
        shutil.copy2(src, dst)
        print("copied", rel)

# 2
shutil.copy2(rlt_src, os.path.join(root, "lingbotvla/models/vla/lingbot_vla/rl_token.py"))

# 3
p = os.path.join(root, "lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py")
s = rd(p)
if "RLT_EXPORT" not in s:
    m = re.search(r"^import [^\n]*\n", s, re.M)
    s = s[:m.start()] + 'import os as _rlt_os\n_RLT_EXPORT = _rlt_os.environ.get("LINGBOT_RLT", "0") == "1"  # RLT_EXPORT\n' + s[m.start():]
    old = ("        seq_wise_loss, router_z_loss, moe_metrics = self._moe_losses_and_metrics(\n"
           "            router_logits_list, losses\n        )\n")
    assert s.count(old) == 1
    s = s.replace(old, old + ("        if _RLT_EXPORT:  # RLT_EXPORT: detached prefix for the RL token side module\n"
                              "            moe_metrics = dict(moe_metrics or {})\n"
                              "            moe_metrics[\"_rlt_prefix\"] = outputs_embeds.detach()\n"
                              "            moe_metrics[\"_rlt_mask\"] = prefix_pad_masks\n"))
    wr(p, s)
    print("patched modeling")

# 4
p = os.path.join(root, "tasks/vla/train_lingbotvla.py")
s = rd(p)
ch = False
s, c = sub_once(s,
                "    if args.train.global_rank == 0:\n        log_dir=f\"{args.train.output_dir}/runs/\"\n",
                "    rlt_trainer = None  # RLT_PATCH\n"
                "    if os.environ.get(\"LINGBOT_RLT\", \"0\") == \"1\":\n"
                "        from lingbotvla.models.vla.lingbot_vla.rl_token import RLTTrainer\n"
                "        rlt_trainer = RLTTrainer(torch.device(f\"cuda:{args.train.local_rank}\"), args.train.global_rank, args.train.world_size)\n"
                "    if args.train.global_rank == 0:\n        log_dir=f\"{args.train.output_dir}/runs/\"\n",
                "rlt_trainer = None  # RLT_PATCH"); ch |= c
s, c = sub_once(s, "    if candidates:\n        last_err = None\n",
                "    _rlt_loaded_cp = None  # RLT_LOADCP\n    if candidates:\n        last_err = None\n",
                "_rlt_loaded_cp = None  # RLT_LOADCP"); ch |= c
s, c = sub_once(s, "                logger.info_rank0(f\"Load distributed checkpoint from {cp} successfully!\")\n",
                "                logger.info_rank0(f\"Load distributed checkpoint from {cp} successfully!\")\n"
                "                _rlt_loaded_cp = cp  # RLT_SETCP\n",
                "# RLT_SETCP"); ch |= c
s, c = sub_once(s, "    helper.empty_cache()\n    model_fwd_context, model_bwd_context = build_activation_offloading_context(\n",
                "    if rlt_trainer is not None:  # RLT_RESUME\n        rlt_trainer.maybe_load(_rlt_loaded_cp)\n"
                "    helper.empty_cache()\n    model_fwd_context, model_bwd_context = build_activation_offloading_context(\n",
                "# RLT_RESUME"); ch |= c
s, c = sub_once(s, "            start_time = time.time()\n            for micro_batch in micro_batches:\n",
                "            start_time = time.time()\n"
                "            if rlt_trainer is not None:  # RLT_BEGIN\n                rlt_trainer.begin_step(global_step)\n"
                "            for micro_batch in micro_batches:\n",
                "# RLT_BEGIN"); ch |= c
s, c = sub_once(s, "                with model_bwd_context:\n                    loss.backward()\n",
                "                with model_bwd_context:\n                    loss.backward()\n"
                "                if rlt_trainer is not None:  # RLT_MICRO\n"
                "                    rlt_trainer.micro_step(loss_log, len(micro_batches))\n",
                "# RLT_MICRO"); ch |= c
s, c = sub_once(s, "            optimizer.step()\n            lr_scheduler.step()\n            optimizer.zero_grad()\n",
                "            optimizer.step()\n            lr_scheduler.step()\n            optimizer.zero_grad()\n"
                "            rlt_metrics = rlt_trainer.opt_step(global_step) if rlt_trainer is not None else None  # RLT_OPT\n",
                "# RLT_OPT"); ch |= c
s, c = sub_once(s, "                f\"Ignore_Batch_Num {ignore_batch_num}\"\n            )\n",
                "                f\"Ignore_Batch_Num {ignore_batch_num}\"\n            )\n"
                "            if rlt_metrics is not None:  # RLT_LOG\n"
                "                logger.info_rank0(rlt_trainer.log_line(global_step, rlt_metrics))\n"
                "                if args.train.global_rank == 0:\n"
                "                    for _k, _v in rlt_metrics.items():\n"
                "                        writer.add_scalar(f\"rlt/{_k}\", _v, global_step)\n",
                "# RLT_LOG"); ch |= c
if "# RLT_SAVE" not in s:
    pat = re.compile(r"^( *)logger\.info_rank0\(f\"Distributed checkpoint saved at \{save_checkpoint_path\} successfully!\"\)\n", re.M)
    n = len(pat.findall(s))
    assert n == 3, n
    s = pat.sub(lambda m: m.group(0) + f"{m.group(1)}if rlt_trainer is not None:  # RLT_SAVE\n"
                f"{m.group(1)}    rlt_trainer.save(save_checkpoint_path, global_step)\n", s)
    ch = True
if ch:
    wr(p, s)
    print("patched train loop")
# 5
def patch_file(rel, old, new, tag):
    p = os.path.join(root, rel)
    t = rd(p)
    t2, c = sub_once(t, old, new, tag)
    if c:
        wr(p, t2)
        print("patched", rel, tag)


patch_file("lingbotvla/models/vla/lingbot_vla/flex_attention.py",
           'FLEX_KERNEL_OPTIONS = {"BLOCK_M": 32, "BLOCK_N": 64, "num_warps": 4, "num_stages": 2}\n',
           '# PyTorch 2.6 supplies num_warps/num_stages internally.  # PPU_FIX\nFLEX_KERNEL_OPTIONS = {"BLOCK_M": 32, "BLOCK_N": 64}\n',
           "# PPU_FIX")
patch_file("lingbotvla/models/vla/lingbot_vla/qwen3vl_in_vla.py",
           "class Qwen3VLTextModel(_Qwen3VLTextModel):\n",
           "class Qwen3VLTextModel(_Qwen3VLTextModel):\n"
           "    def _deepstack_process(self, hidden_states, visual_pos_masks, visual_embeds):  # PPU_FIX\n"
           "        # Fixed-size gather avoids boolean-index outputs and in-place CopySlices.\n"
           "        mask = visual_pos_masks.to(hidden_states.device)\n"
           "        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)\n"
           "        if visual_embeds.shape[0] == 0:\n"
           "            return hidden_states\n"
           "        indices = (mask.reshape(-1).long().cumsum(0) - 1).clamp_min(0)\n"
           "        additions = visual_embeds.index_select(0, indices).reshape_as(hidden_states)\n"
           "        return torch.where(mask.unsqueeze(-1), hidden_states + additions, hidden_states)\n\n",
           "# PPU_FIX")
patch_file("lingbotvla/data/vla_data/base_dataset.py",
           "            result[key] = torch.stack(self.hf_dataset[relative_indices][key])\n",
           "            result[key] = torch.stack(self.hf_dataset.select_columns([key])[relative_indices][key])  # PPU_FIX\n",
           "# PPU_FIX")
# 6 benchmark runs: LINGBOT_NO_FINAL_SAVE=1 skips the automatic DCP + hf_ckpt save when max_steps is reached
p6 = os.path.join(root, "tasks/vla/train_lingbotvla.py")
t = rd(p6)
t2, c = sub_once(t, "            already_saved = args.train.save_steps and global_step % args.train.save_steps == 0\n",
                 "            already_saved = args.train.save_steps and global_step % args.train.save_steps == 0\n"
                 "            already_saved = already_saved or os.environ.get(\"LINGBOT_NO_FINAL_SAVE\") == \"1\"  # NO_FINAL_SAVE\n",
                 "# NO_FINAL_SAVE")
if c:
    wr(p6, t2)
    print("patched NO_FINAL_SAVE")
# 7 bf16 mixed precision: LINGBOT_HF_SAVE_FP32=1 keeps hf_ckpt in fp32 (master weights are fp32; same switch as the MetaX copy)
p7 = os.path.join(root, "lingbotvla/utils/async_hf_checkpoint.py")
t = rd(p7)
t2, c = sub_once(t, "            if enable_fp32:\n                save_kwargs[\"save_dtype\"] = torch.float32\n",
                 "            if enable_fp32 or os.environ.get(\"LINGBOT_HF_SAVE_FP32\") == \"1\":  # HF_SAVE_FP32\n"
                 "                save_kwargs[\"save_dtype\"] = torch.float32\n",
                 "# HF_SAVE_FP32")
if c:
    assert "\nimport os" in t2 or t2.startswith("import os"), "async_hf_checkpoint.py lacks import os"
    wr(p7, t2)
    print("patched HF_SAVE_FP32")
# 8 frame cache (LINGBOT_FRAME_CACHE=<dir with cache_meta.json>, default off): LeRobotDataset._query_videos reads the
#   decoded + resized frames from the concatenated per-camera arrays of tools/build_frame_cache.py instead of decoding
#   video, and the per-camera Resize is skipped (the cache already holds Resize((S, S)) output). Only used for the dataset
#   whose realpath equals cache_meta.dataset_root, only if verified, and only if the dataset's Resize size equals the
#   cache size. Bit-identical frames (verify_frame_cache.py).
p8 = os.path.join(root, "lingbotvla/data/vla_data/base_dataset.py")
t = rd(p8)
if "# FRAME_CACHE" not in t:
    helper = (
        "\n_FRAME_CACHES = {}  # FRAME_CACHE\n\n\n"
        "def _frame_cache_for(dataset_root):\n"
        "    cache_dir = os.environ.get(\"LINGBOT_FRAME_CACHE\")\n"
        "    if not cache_dir:\n"
        "        return None\n"
        "    key = os.path.realpath(str(dataset_root))\n"
        "    if key not in _FRAME_CACHES:\n"
        "        import json as _json\n"
        "        import numpy as _np\n"
        "        meta = _json.load(open(os.path.join(cache_dir, \"cache_meta.json\")))\n"
        "        if meta.get(\"dataset_root\") != key:\n"
        "            _FRAME_CACHES[key] = None\n"
        "        else:\n"
        "            if not meta.get(\"verified\"):\n"
        "                raise RuntimeError(f\"frame cache {cache_dir} is not verified\")\n"
        "            _FRAME_CACHES[key] = dict(\n"
        "                meta=meta, fps=meta[\"fps\"], size=list(meta[\"size\"]),\n"
        "                ep_from={int(e): int(f) for e, f in zip(meta[\"episode_index\"], meta[\"episode_from\"])},\n"
        "                arrays={cam: _np.load(os.path.join(cache_dir, fn), mmap_mode=\"r\") for cam, fn in meta[\"cameras\"].items()})\n"
        "    return _FRAME_CACHES[key]\n"
    )
    anchor = "\n\ndef _to_relative_indices(dataset, query_indices):\n"
    assert t.count(anchor) == 1
    t = t.replace(anchor, "\n" + helper + anchor, 1)
    old_q = ("        item = {}\n"
             "        for vid_key, query_ts in query_timestamps.items():\n"
             "            if LEROBOT_DATASET_API == \"v3\":\n")
    new_q = ("        _fc = _frame_cache_for(self.root)  # FRAME_CACHE: read cached frames instead of decoding\n"
             "        if _fc is not None:\n"
             "            import numpy as _np\n"
             "            if list(getattr(self.image_transforms, \"size\", _fc[\"size\"])) != _fc[\"size\"]:\n"
             "                raise RuntimeError(f\"frame cache size {_fc['size']} != dataset Resize {self.image_transforms.size}\")\n"
             "            item = {}\n"
             "            ep_from = _fc[\"ep_from\"][int(ep_idx)]\n"
             "            for vid_key, query_ts in query_timestamps.items():\n"
             "                idx = [ep_from + int(round(float(ts) * _fc[\"fps\"])) for ts in query_ts]\n"
             "                item[vid_key] = torch.from_numpy(_np.ascontiguousarray(_fc[\"arrays\"][vid_key][idx])).squeeze(0)\n"
             "            return item\n"
             "        item = {}\n"
             "        for vid_key, query_ts in query_timestamps.items():\n"
             "            if LEROBOT_DATASET_API == \"v3\":\n")
    assert t.count(old_q) == 1
    t = t.replace(old_q, new_q, 1)
    old_t = "        if self.image_transforms is not None and self.load_image:\n"
    new_t = "        if self.image_transforms is not None and self.load_image and _frame_cache_for(self.root) is None:  # FRAME_CACHE: cached frames are already resized\n"
    assert t.count(old_t) == 1
    t = t.replace(old_t, new_t, 1)
    wr(p8, t)
    print("patched FRAME_CACHE")
# 9 persistent dataloader workers across epochs (LINGBOT_PERSISTENT_WORKERS=1, default off): avoids re-forking all
#   workers and re-initialising the datasets at every epoch boundary (observed ~6 min per epoch on 4 x 16 workers).
p9 = os.path.join(root, "lingbotvla/data/data_loader.py")
t = rd(p9)
if "# PERSISTENT_WORKERS" not in t:
    old = "        prefetch_factor=prefetch_factor,\n    )\n    if use_rmpad:\n"
    new = ("        prefetch_factor=prefetch_factor,\n"
           "        persistent_workers=(num_workers > 0 and __import__(\"os\").environ.get(\"LINGBOT_PERSISTENT_WORKERS\") == \"1\"),  # PERSISTENT_WORKERS\n"
           "    )\n    if use_rmpad:\n")
    assert t.count(old) == 1
    t = t.replace(old, new, 1)
    wr(p9, t)
    print("patched PERSISTENT_WORKERS")
print("ok")
