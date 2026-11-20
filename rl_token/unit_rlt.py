import os, torch, torch.distributed as dist, time, tempfile
os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "62599")
dist.init_process_group("nccl", rank=0, world_size=1)
from lingbotvla.models.vla.lingbot_vla.rl_token import RLTTrainer
dev = torch.device("cuda:0")
cpu_state = torch.get_rng_state().clone(); cuda_state = torch.cuda.get_rng_state().clone()
tr = RLTTrainer(dev, 0, 1, warmup=5, diag_every=1)
assert torch.equal(cpu_state, torch.get_rng_state()), "global CPU RNG touched by RLT init"
B, P, D = 16, 247, 2560
g = torch.Generator(device=dev).manual_seed(0)
base = torch.randn(4, P, D, device=dev, generator=g) * 2 + 0.5
valid = torch.ones(B, P, dtype=torch.bool, device=dev); valid[:, 200:230] = False
for step in range(1, 41):
    tr.begin_step(step)
    for _ in range(4):
        idx = torch.randint(0, 4, (B,), device=dev, generator=g)
        h = base[idx] + 0.3 * torch.randn(B, P, D, device=dev, generator=g)
        tr.micro_step({"_rlt_prefix": h, "_rlt_mask": valid}, 4)
    m = tr.opt_step(step)
    if step in (1, 2, 10, 20, 40): print(tr.log_line(step, m), flush=True)
assert torch.equal(cuda_state, torch.cuda.get_rng_state()), "global CUDA RNG touched by RLT"
d = tempfile.mkdtemp(); tr.save(d, 40)
tr2 = RLTTrainer(dev, 0, 1); tr2.maybe_load(d)
z1 = tr.module.rl_token(h, valid); z2 = tr2.module.rl_token(h, valid)
print("reload z maxdiff", (z1 - z2).abs().max().item(), "z shape", tuple(z1.shape), "mem GiB", torch.cuda.max_memory_allocated() / 2**30)
dist.destroy_process_group()
