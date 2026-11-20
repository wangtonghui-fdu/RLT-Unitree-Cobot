"""gpu_consistency.py [SECONDS]: every GPU runs the same seeded workload (fp32 + bf16 matmul / softmax chains, ~10 GB) in its own
process; identical cards must give bit-identical checksums.  Reports per GPU: iterations, mismatches against the majority, NaN/Inf."""
import sys, time, hashlib, torch, torch.multiprocessing as mp
SEC = float(sys.argv[1]) if len(sys.argv) > 1 else 60


def work(i, q):
    dev = torch.device(f"cuda:{i}"); g = torch.Generator().manual_seed(1234)
    A = torch.randn(6000, 6000, generator=g).to(dev); B = torch.randn(6000, 6000, generator=g).to(dev)
    big = torch.randn(8, 4096, 4096, generator=g).to(dev)            # extra resident memory, also read every round
    out = []; t0 = time.time(); k = 0
    while time.time() - t0 < SEC:
        x = A
        for _ in range(6):
            x = torch.tanh(x @ B / 77.0) + 0.001 * A
        y = (A.to(torch.bfloat16) @ B.to(torch.bfloat16)).float().softmax(-1)
        z = torch.bmm(big, big.transpose(1, 2) / 4096.0).sum(0)
        bad = int((~torch.isfinite(x)).sum() + (~torch.isfinite(y)).sum() + (~torch.isfinite(z)).sum())
        h = hashlib.md5(x.cpu().numpy().tobytes() + y.cpu().numpy().tobytes() + z.cpu().numpy().tobytes()).hexdigest()
        out.append((h, bad)); k += 1
    q.put((i, out))


if __name__ == "__main__":
    mp.set_start_method("spawn"); n = torch.cuda.device_count(); q = mp.Queue()
    ps = [mp.Process(target=work, args=(i, q)) for i in range(n)]; [p.start() for p in ps]
    res = dict(q.get() for _ in range(n)); [p.join() for p in ps]
    from collections import Counter
    ref = Counter(h for o in res.values() for h, _ in o).most_common(1)[0][0]
    for i in range(n):
        o = res[i]; print(f"GPU {i}: {len(o)} rounds, mismatching rounds {sum(h != ref for h, _ in o)}, rounds with NaN/Inf {sum(b > 0 for _, b in o)}")
    print("distinct checksums overall:", len(set(h for o in res.values() for h, _ in o)))
