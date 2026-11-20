"""an_opt.py LOGDIR [FORMAL_LOG]: summary of the PPU speed tests: per-test median step time and [TIMING] breakdown (steps >= 6),
and per-step losses against T0_base (and against the formal run's first steps if its log is given)."""
import sys, re, glob, os, statistics as st
d = sys.argv[1]
def parse(f):
    t = open(f, errors="ignore").read().replace("\r", "\n")
    tim = {}
    for m in re.finditer(r"\[TIMING\] step (\d+) total ([\d.]+) data_wait ([\d.]+) \| teacher ([\d.]+) fwd ([\d.]+) bwd ([\d.]+) loop_other ([\d.]+) \| clip ([\d.]+) opt ([\d.]+) tail ([\d.]+) \| micro (\d+) mem_peak_GiB ([\d.]+)", t):
        tim[int(m.group(1))] = [float(x) for x in m.groups()[1:]]
    loss = {}
    for m in re.finditer(r"Step (\d+)/\d+, Epoch \d+, Loss ([\d.]+), VLA_Loss ([\d.]+), Depth_Loss ([\d.]+), Future_Depth_Loss ([\d.]+), FutureVideo_Loss ([\d.]+).*?GradNorm ([\d.]+)", t):
        loss.setdefault(int(m.group(1)), [float(x) for x in m.groups()[1:]])
    its = [float(x) for x in re.findall(r"Step: +\d+/\d+ \[[^\]]*?([\d.]+)s/it\]", t)]
    oom = len(re.findall(r"out of memory|OutOfMemory", t)); nccl = sorted(set(re.findall(r"NCCL INFO [^\n]{0,140}(?:P2P|SHM|NET|via)[^\n]{0,80}", t)))[:6]
    return tim, loss, its, oom, nccl
names = ["total", "data_wait", "teacher", "fwd", "bwd", "loop_other", "clip", "opt", "tail", "micro", "mem"]
res = {}
for f in sorted(glob.glob(d + "/T*.log")):
    tag = os.path.basename(f)[:-4]; tim, loss, its, oom, nccl = parse(f); res[tag] = (tim, loss)
    print(f"== {tag}: steps logged {len(loss)}, oom lines {oom}")
    ks = [k for k in tim if k >= 6]
    if ks:
        med = [st.median(tim[k][i] for k in ks) for i in range(len(names))]
        print("   median over steps >= 6:  " + "  ".join(f"{n} {v:.2f}" for n, v in zip(names, med)))
    if nccl: print("   NCCL:", " || ".join(x[:150] for x in nccl[:4]))
base = res.get("T0_base", ({}, {}))[1]
def cmp(a, b, label):
    ks = sorted(set(a) & set(b))
    if not ks: print(f"   {label}: no common steps"); return
    same = sum(a[k] == b[k] for k in ks); rel = max(abs(a[k][j] - b[k][j]) / max(abs(b[k][j]), 1e-9) for k in ks for j in range(5))
    print(f"   {label}: {same}/{len(ks)} steps with identical logged losses (4 decimals); max relative difference {rel:.4f}; last common step {ks[-1]}: {a[ks[-1]][:2]} vs {b[ks[-1]][:2]}")
for tag, (tim, loss) in res.items():
    if tag != "T0_base": cmp(loss, base, f"{tag} vs T0_base")
if len(sys.argv) > 2:
    _, floss, _, _, _ = parse(sys.argv[2]); cmp(base, floss, "T0_base vs formal run")
