"""Side-by-side progress of the two insert-charge lines (VLA + RL token), step blocks aligned (same seed -> same batches).
usage: compare_progress.py LOG_A NAME_A LOG_B NAME_B"""
import re, sys, statistics as st
def parse(p):
    t = open(p, errors="ignore").read()
    vla = {int(m[0]): (float(m[1]), float(m[2]), float(m[3]), float(m[4])) for m in re.findall(
        r"Step (\d+)/\d+, Epoch \d+, Loss [0-9.]+, VLA_Loss ([0-9.]+), Depth_Loss ([0-9.]+),.*?GradNorm ([0-9.]+).*?StepTime ([0-9.]+)s", t)}
    rlt = {int(m[0]): float(m[1]) for m in re.findall(r"\[RLT\] step (\d+) loss ([0-9.]+)", t)}
    diag = {int(m[0]): (float(m[1]), float(m[2]), float(m[3])) for m in re.findall(r"\[RLT\] step (\d+) loss [0-9.]+ .*?z_only ([0-9.]+) shuf ([0-9.]+) gap ([-0-9.]+)", t)}
    return vla, rlt, diag, t.count("Traceback")
A = parse(sys.argv[1]); B = parse(sys.argv[3]); na, nb = sys.argv[2], sys.argv[4]
print(f"{na}: {max(A[0]) if A[0] else 0} steps, tracebacks {A[3]} | {nb}: {max(B[0]) if B[0] else 0} steps, tracebacks {B[3]}")
edges = [0, 10, 25, 50, 100, 200, 300, 400, 600, 800, 1000, 1250, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 8000, 10000, 15000, 20000]
hdr = f"{'steps':>11} | {'VLA '+na:>13} {'VLA '+nb:>13} | {'RLT '+na:>12} {'RLT '+nb:>12} | {'z_gap '+na:>13} {'z_gap '+nb:>13} | {'s/step':>11}"
print(hdr); print("-" * len(hdr))
def blk(d, a, b, idx=None):
    v = [d[k] if idx is None else d[k][idx] for k in d if a < k <= b]
    return st.mean(v) if v else None
fmt = lambda x, w, p: f"{x:{w}.{p}f}" if x is not None else " " * (w - 1) + "-"
for a, b in zip(edges, edges[1:]):
    if not any(a < k <= b for k in list(A[0]) + list(B[0])): break
    va, vb = blk(A[0], a, b, 0), blk(B[0], a, b, 0)
    ra, rb = blk(A[1], a, b), blk(B[1], a, b)
    ga, gb = blk(A[2], a, b, 2), blk(B[2], a, b, 2)
    ta = st.median([A[0][k][3] for k in A[0] if a < k <= b]) if any(a < k <= b for k in A[0]) else None
    tb = st.median([B[0][k][3] for k in B[0] if a < k <= b]) if any(a < k <= b for k in B[0]) else None
    print(f"{a+1:>5}-{b:<5} | {fmt(va,13,4)} {fmt(vb,13,4)} | {fmt(ra,12,3)} {fmt(rb,12,3)} | {fmt(ga,13,4)} {fmt(gb,13,4)} | {fmt(ta,5,2)}/{fmt(tb,5,2)}")
