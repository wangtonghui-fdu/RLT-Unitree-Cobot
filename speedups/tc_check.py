"""tc_check.py CACHE_LOG REF_LOG: PASS/FAIL for the teacher-cache validation run (two switches + cache, 24 steps from init) against
the reference run with the same switches and online teachers.  PASS = no cache miss, cached targets ~ online targets in the
verified micro-batches (min cosine >= 0.999), per-step losses within 2 % of the reference, and a faster median step."""
import sys, re, statistics as st
def parse(f):
    t = open(f, errors="ignore").read().replace("\r", "\n"); loss = {}; tot = {}; hm = None
    for m in re.finditer(r"Step (\d+)/\d+, Epoch \d+, Loss ([\d.]+), VLA_Loss ([\d.]+), Depth_Loss ([\d.]+), Future_Depth_Loss ([\d.]+), FutureVideo_Loss ([\d.]+)", t):
        loss.setdefault(int(m.group(1)), [float(x) for x in m.groups()[1:]])
    for m in re.finditer(r"\[TIMING\] step (\d+) total ([\d.]+) .*?teacher ([\d.]+)", t): tot[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    for m in re.finditer(r"tcache hit (\d+) miss (\d+)", t): hm = (int(m.group(1)), int(m.group(2)))
    cos = [float(x) for x in re.findall(r"min cos ([\d.]+)", t)]; ident = re.findall(r"identical (True|False)", t)
    return loss, tot, hm, cos, ident, ("DISABLED" in t)
L, T, hm, cos, ident, dis = parse(sys.argv[1]); RL, RT, _, _, _, _ = parse(sys.argv[2])
ks = sorted(set(L) & set(RL)); rel = max((abs(L[k][j] - RL[k][j]) / max(abs(RL[k][j]), 1e-9) for k in ks for j in range(5)), default=9)
med = lambda D, i: st.median(D[k][i] for k in D if k >= 6) if any(k >= 6 for k in D) else float("nan")
print(f"steps {len(L)} (common {len(ks)}), hits/misses on rank 0 {hm}, verified micro-batches {len(cos) // 4}, min cos {min(cos) if cos else None}, bit-identical targets {ident.count('True')}/{len(ident)}, disabled {dis}")
print(f"max relative loss difference vs reference {rel:.4f}; median step {med(T, 0):.2f} s (teacher {med(T, 1):.2f}) vs reference {med(RT, 0):.2f} s (teacher {med(RT, 1):.2f})")
ok = len(ks) >= 24 and hm is not None and hm[1] == 0 and hm[0] > 0 and cos and min(cos) >= 0.999 and rel <= 0.02 and not dis and med(T, 0) < med(RT, 0) - 0.3
print("TC_CHECK", "PASS" if ok else "FAIL")
