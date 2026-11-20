"""patch_opt.py CODE_DIR (11-20): speed experiments for LingBot-VLA v2 full fine-tuning on 4 x PPU, ported by description from the
8 x RTX PRO 5000 notes (no diff available).  Everything is OFF by default and switched by environment variables, so the patched
tree behaves exactly like code_v2 unless a switch is set:
  LINGBOT_STEP_TIMING=1    per-step breakdown on rank 0 ([TIMING] lines; adds cuda synchronize calls)
  OPT_NO_RESHARD_BWD=1     gradient accumulation: keep the FSDP2 parameters unsharded between the micro-batches of one step
                           (FSDPModule.set_reshard_after_backward(False) for all but the last micro-batch) -> 1 parameter
                           all-gather per optimizer step instead of 1 per micro-batch; same arithmetic
  MUON_OWNER_COMPUTE=1     DistributedMuon mega-batch path: every matrix is orthogonalised on ONE rank (owner = index % world)
                           instead of on every rank; local shards go to the owner and the result slices come back with two
                           bf16 all_to_all exchanges (Newton-Schulz already runs in bf16, so the inputs are the same numbers)
Idempotent; writes <file>.bak_opt next to each patched file the first time."""
import sys, os, shutil

root = sys.argv[1]


def patch(rel, edits, marker):
    p = os.path.join(root, rel); s = open(p).read()
    if marker in s: print("already patched:", rel); return
    for old, new in edits:
        assert s.count(old) == 1, (rel, s.count(old), old[:70])
        s = s.replace(old, new)
    if not os.path.exists(p + ".bak_opt"): shutil.copy(p, p + ".bak_opt")
    open(p, "w").write(s); print("patched:", rel)


# ---------------------------------------------------------------- train loop: timing + no reshard between micro-batches
T = "tasks/vla/train_lingbotvla.py"
patch(T, [
    ("""            try:
                micro_batches: List[Dict[str, Any]] = next(data_iterator)
""", """            _tm_on = os.environ.get("LINGBOT_STEP_TIMING") == "1"  # OPT_PATCH
            _tm_d0 = time.time()
            try:
                micro_batches: List[Dict[str, Any]] = next(data_iterator)
"""),
    ("""            torch.cuda.synchronize()
            start_time = time.time()
            if rlt_trainer is not None:  # RLT_BEGIN
                rlt_trainer.begin_step(global_step)
            for micro_batch in micro_batches:
""", """            _tm_data = time.time() - _tm_d0
            torch.cuda.synchronize()
            start_time = time.time()
            _tm = dict(teacher=0.0, fwd=0.0, bwd=0.0)
            if rlt_trainer is not None:  # RLT_BEGIN
                rlt_trainer.begin_step(global_step)
            _nrb = os.environ.get("OPT_NO_RESHARD_BWD") == "1" and len(micro_batches) > 1
            for _mi, micro_batch in enumerate(micro_batches):
                if _nrb and (_mi == 0 or _mi == len(micro_batches) - 1):
                    _opt_set_reshard_after_backward(model, _mi == len(micro_batches) - 1, args.train.local_rank)
"""),
    ("""                depth_forward_time = 0
                if use_depth_align:
""", """                depth_forward_time = 0
                if _tm_on:
                    torch.cuda.synchronize(); _tm_t0 = time.time()
                if use_depth_align:
"""),
    ("""                with model_fwd_context:
                    model_outputs = model(
""", """                if _tm_on:
                    torch.cuda.synchronize(); _tm_t1 = time.time(); _tm["teacher"] += _tm_t1 - _tm_t0
                with model_fwd_context:
                    model_outputs = model(
"""),
    ("""                with model_bwd_context:
                    loss.backward()
""", """                if _tm_on:
                    torch.cuda.synchronize(); _tm_t2 = time.time(); _tm["fwd"] += _tm_t2 - _tm_t1
                with model_bwd_context:
                    loss.backward()
                if _tm_on:
                    torch.cuda.synchronize(); _tm["bwd"] += time.time() - _tm_t2
"""),
    ("""            if global_step > args.train.stable_train_steps:
                max_grad_norm = args.train.decayed_max_grad_norm
""", """            if _tm_on:
                torch.cuda.synchronize(); _tm_t3 = time.time()
            if global_step > args.train.stable_train_steps:
                max_grad_norm = args.train.decayed_max_grad_norm
"""),
    ("""            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
""", """            if _tm_on:
                torch.cuda.synchronize(); _tm_t4 = time.time()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if _tm_on:
                torch.cuda.synchronize(); _tm_t5 = time.time()
"""),
    ("""            torch.cuda.synchronize()
            delta_time = time.time() - start_time
""", """            torch.cuda.synchronize()
            delta_time = time.time() - start_time
            if _tm_on and args.train.local_rank == 0:
                _tm_loop = _tm_t3 - start_time
                print(f"[TIMING] step {global_step} total {delta_time:.2f} data_wait {_tm_data:.2f} | teacher {_tm['teacher']:.2f} fwd {_tm['fwd']:.2f} bwd {_tm['bwd']:.2f} "
                      f"loop_other {_tm_loop - _tm['teacher'] - _tm['fwd'] - _tm['bwd']:.2f} | clip {_tm_t4 - _tm_t3:.2f} opt {_tm_t5 - _tm_t4:.2f} tail {start_time + delta_time - _tm_t5:.2f} "
                      f"| micro {len(micro_batches)} mem_peak_GiB {torch.cuda.max_memory_allocated() / 2**30:.1f}", flush=True)
"""),
    ("""from lingbotvla.utils.dist_utils import all_reduce
""", """from lingbotvla.utils.dist_utils import all_reduce

_OPT_FSDP_MODULES = None  # OPT_PATCH


def _opt_set_reshard_after_backward(model, flag, local_rank=0):
    \"\"\"OPT_NO_RESHARD_BWD: FSDP2 units keep their unsharded parameters after backward unless ``flag``.\"\"\"
    global _OPT_FSDP_MODULES
    if _OPT_FSDP_MODULES is None:
        from torch.distributed.fsdp import FSDPModule
        _OPT_FSDP_MODULES = [m for m in model.modules() if isinstance(m, FSDPModule)]
        if local_rank == 0:
            print(f"[OPT] OPT_NO_RESHARD_BWD: {len(_OPT_FSDP_MODULES)} FSDP2 units keep their parameters between micro-batches", flush=True)
    for m in _OPT_FSDP_MODULES:
        m.set_reshard_after_backward(flag, recurse=False)
"""),
], "OPT_PATCH")

# ---------------------------------------------------------------- Muon: owner compute
M = "lingbotvla/optim/muon.py"
patch(M, [
    ("""        # Phase 2: Stack and batched all-gather
        stacked_local = torch.stack(local_updates, dim=0)  # [N, local_M, K]
""", """        if _OWNER_COMPUTE and world_size > 1 and len(params) >= world_size and params[0].ndim == 2:  # OPT_PATCH
            self._owner_compute_chunk(params, local_updates, has_grad, ns_coefficients, ns_steps, eps, lr, weight_decay,
                                      adjusted_lr, pg, world_size, rank, shard_dim)
            return

        # Phase 2: Stack and batched all-gather
        stacked_local = torch.stack(local_updates, dim=0)  # [N, local_M, K]
"""),
    ("""    @staticmethod
    def _compute_ortho(
""", """    def _owner_compute_chunk(self, params, local_updates, has_grad, ns_coefficients, ns_steps, eps, lr, weight_decay,
                             adjusted_lr, pg, world_size, rank, shard_dim):
        \"\"\"MUON_OWNER_COMPUTE: matrix i of the chunk is orthogonalised only on rank i % world_size.
        hop 1: every rank sends its local shard of the matrices rank d owns to d (bf16, the dtype Newton-Schulz uses);
        the owner concatenates the shards in rank order (= FSDP2's contiguous chunks) and runs the same batched NS;
        hop 2: the owner sends every rank that rank's slice of the result; the update is applied locally as before.\"\"\"
        ws = world_size; N = len(params); cdt = torch.bfloat16; gd = shard_dim + 1
        gsize = params[0].shape[shard_dim]; floor_, rem = gsize // ws, gsize % ws
        sizes = [floor_ + (1 if r < rem else 0) for r in range(ws)]
        starts = [r * floor_ + min(r, rem) for r in range(ws)]
        owned = [[i for i in range(N) if i % ws == d] for d in range(ws)]
        mine = owned[rank]; dev = local_updates[0].device
        oshape = list(local_updates[0].shape)                       # this rank's local shard shape
        send = [torch.stack([local_updates[i] for i in owned[d]], 0).to(cdt).contiguous() for d in range(ws)]
        del local_updates
        recv = []
        for s_ in range(ws):
            shp = list(oshape); shp[shard_dim] = sizes[s_]
            recv.append(torch.empty([len(mine)] + shp, dtype=cdt, device=dev))
        dist.all_to_all(recv, send, group=pg)
        del send
        ortho = batched_newton_schulz(torch.cat(recv, dim=gd), ns_coefficients, ns_steps, eps)   # [len(mine), M, K] bf16
        del recv
        send2 = [ortho.narrow(gd, starts[d], sizes[d]).contiguous() for d in range(ws)]
        recv2 = [torch.empty([len(owned[s_])] + oshape, dtype=cdt, device=dev) for s_ in range(ws)]
        dist.all_to_all(recv2, send2, group=pg)
        del send2, ortho
        for s_ in range(ws):
            for j, i in enumerate(owned[s_]):
                if not has_grad[i]:
                    continue
                p = params[i]; p_local = p.to_local() if isinstance(p, DTensor) else p
                if weight_decay != 0.0:
                    p_local.mul_(1 - lr * weight_decay)
                p_local.add_(recv2[s_][j].to(dtype=p_local.dtype), alpha=-adjusted_lr)

    @staticmethod
    def _compute_ortho(
"""),
    ("""_MEGABATCH_MAX_GROUP_SIZE = 32
""", """_MEGABATCH_MAX_GROUP_SIZE = 32
_OWNER_COMPUTE = __import__("os").environ.get("MUON_OWNER_COMPUTE") == "1"  # OPT_PATCH
"""),
], "OPT_PATCH")
print("done")
