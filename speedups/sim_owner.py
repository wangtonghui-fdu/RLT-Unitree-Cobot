"""sim_owner.py CODE_DIR: single-process simulation of MUON_OWNER_COMPUTE's index logic: ws fake ranks (threads), all_to_all
emulated through a shared mailbox; compares the applied update with the original 'full stack on every rank' result (CPU, tiny)."""
import sys, os, types, random, threading, importlib.util
import torch
os.environ["MUON_OWNER_COMPUTE"] = "1"
spec = importlib.util.spec_from_file_location("muon_mod", os.path.join(sys.argv[1], "lingbotvla/optim/muon.py")); mm = importlib.util.module_from_spec(spec); spec.loader.exec_module(mm)
assert mm._OWNER_COMPUTE


class PT:                                  # stands in for a sharded DTensor parameter: global .shape, local tensor via to_local()
    ndim = 2
    def __init__(self, t, shape): self.t, self.shape = t, shape
    def to_local(self): return self.t


mm.DTensor = PT
torch.manual_seed(0); random.seed(0); worst = 0.0; nbad = 0; T = 40
for trial in range(T):
    ws = random.choice([2, 4, 8]); N = random.randint(ws, 32); sd = random.choice([0, 1]); M_, K_ = random.choice([(13, 7), (16, 16), (9, 21), (30, 10)])
    g = M_ if sd == 0 else K_; fl, rem = g // ws, g % ws; sizes = [fl + (1 if r < rem else 0) for r in range(ws)]; starts = [r * fl + min(r, rem) for r in range(ws)]
    full = [torch.randn(M_, K_) for _ in range(N)]; has_grad = [random.random() > 0.1 for _ in range(N)]
    for i in range(N):
        if not has_grad[i]: full[i].zero_()
    local = [[f.narrow(sd, starts[r], sizes[r]).clone() for f in full] for r in range(ws)]
    plocal = [[torch.zeros_like(x) for x in local[r]] for r in range(ws)]
    ref = mm.batched_newton_schulz(torch.stack(full, 0), mm.DEFAULT_NS_COEFFICIENTS, 5, 1e-7)      # original path: [N, M, K] on every rank
    box, lock, barrier, ncall = {}, threading.Lock(), threading.Barrier(ws), [0] * ws
    tl = threading.local()

    def fake_all_to_all(out, inp, group=None):
        me = tl.rank; k = ncall[me]; ncall[me] += 1
        with lock:
            for d in range(ws): box[(k, me, d)] = inp[d]
        barrier.wait()
        for s in range(ws):
            src = box[(k, s, me)]; assert out[s].shape == src.shape and out[s].dtype == src.dtype, (out[s].shape, src.shape); out[s].copy_(src)
        barrier.wait()

    mm.dist = types.SimpleNamespace(all_to_all=fake_all_to_all); errs = []

    def run(r):
        tl.rank = r
        try:
            mm.DistributedMuon._owner_compute_chunk(None, [PT(plocal[r][i], (M_, K_)) for i in range(N)], [x.clone() for x in local[r]], has_grad,
                                                     mm.DEFAULT_NS_COEFFICIENTS, 5, 1e-7, 0.0, 0.0, 1.0, None, ws, r, sd)
        except Exception as e:
            errs.append(repr(e)); barrier.abort()

    th = [threading.Thread(target=run, args=(r,)) for r in range(ws)]; [t.start() for t in th]; [t.join() for t in th]
    assert not errs, errs
    got = [torch.cat([plocal[r][i] for r in range(ws)], dim=sd) for i in range(N)]                   # p = 0 - 1.0 * ortho
    err = max(float((got[i] + (ref[i].to(torch.float32) if has_grad[i] else 0)).abs().max()) for i in range(N)); worst = max(worst, err); nbad += err > 0
print(f"trials {T}, max |owner - reference| = {worst}, trials with any difference: {nbad}")
