import os, torch, torch.distributed as dist
os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "62598")
dist.init_process_group("nccl", rank=0, world_size=1)
from lingbotvla.models.vla.lingbot_vla.rl_token import RLTTrainer
dev = torch.device("cuda:0")
def st(): return torch.cuda.get_rng_state().clone()
s0 = st()
g = torch.Generator(device=dev).manual_seed(0)
x = torch.randn(4, 4, device=dev, generator=g); print("after randn(gen)", torch.equal(s0, st()))
i = torch.randint(0, 4, (16,), device=dev, generator=g); print("after randint(gen)", torch.equal(s0, st()))
tr = RLTTrainer(dev, 0, 1, warmup=5, diag_every=1); print("after init", torch.equal(s0, st()))
B, P, D = 16, 247, 2560
h = torch.randn(B, P, D, device=dev, generator=g); valid = torch.ones(B, P, dtype=torch.bool, device=dev)
tr.begin_step(1); tr.micro_step({"_rlt_prefix": h, "_rlt_mask": valid}, 1); print("after micro", torch.equal(s0, st()))
tr.opt_step(1); print("after opt", torch.equal(s0, st()))
print(s0[-16:].tolist(), st()[-16:].tolist())
