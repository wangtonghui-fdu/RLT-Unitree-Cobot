"""an_wide.py DIR: feasibility map of the wide-randomisation physics-only study (dry6f/DIR): per seed RAND + result / failure stage."""
import json, glob, re, sys, numpy as np, collections
d = sys.argv[1]; rows = []
for f in sorted(glob.glob(d + "/s*.log")):
    t = open(f, errors="ignore").read(); m = re.search(r"RAND (\{.*\})", t)
    if not m: continue
    r = json.loads(m.group(1)); res = re.search(r"RESULT (\{.*\})", t)
    if res: R = json.loads(res.group(1)); st = "ok" if R["ok"] else ("not_seated" if not R["seated"] else "contact")
    else:
        last = [l for l in t.strip().splitlines() if l.strip()][-1]
        st = "fail:" + last.strip()[:46]
    rows.append(dict(seed=int(re.search(r"s(\d+)\.log", f).group(1)), st=st, sx=r["strip_xy"][0], sy=r["strip_xy"][1], sa=r["strip_yaw_deg"], cx=r["charger_xy"][0], cy=r["charger_xy"][1], ca=r["charger_yaw_deg"],
                     dur=(R["duration_s"] if res else None), fb=t.count("-> joint space")))
print(len(rows), "runs;", collections.Counter(r["st"].split(" (")[0][:60] for r in rows).most_common(14))
def stage(s):
    if s == "ok": return "ok"
    if s in ("not_seated", "contact"): return s
    return "charger" if re.search(r"left (approach|vertical|to the middle)|left: no|left .*lift|left approach", s) else "handover" if re.search(r"right (to the tail|fingers)|left rise|cannot rise", s) else "strip"
for r in rows: r["stage"] = stage(r["st"])
print("by stage:", collections.Counter(r["stage"] for r in rows))
def table(key, edges, cond, label):
    print(label)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r in rows if lo <= r[key] < hi]; good = [r for r in sel if cond(r)]
        print(f"   {key} {lo:7.3f}..{hi:7.3f}: {len(good):3d}/{len(sel):3d}")
chg_ok = lambda r: r["stage"] != "charger"                     # the charger part worked (pick + bring to the middle)
print("--- charger (pick + to the middle) ---")
table("cx", np.linspace(0.20, 0.38, 10), chg_ok, "x"); table("cy", np.linspace(-0.425, -0.265, 9), chg_ok, "y"); table("ca", np.linspace(-40, 40, 9), chg_ok, "yaw")
S = [r for r in rows if r["stage"] in ("ok", "strip", "not_seated", "contact")]; rows_all = rows; rows = S
st_ok = lambda r: r["stage"] == "ok"
print("--- strip (carry, hold, insert), among runs whose charger part worked:", len(S), "---")
table("sx", np.linspace(-0.215, -0.055, 9), st_ok, "x"); table("sy", np.linspace(-0.415, -0.275, 8), st_ok, "y"); table("sa", np.linspace(-30, 30, 7), st_ok, "yaw")
ok = [r for r in rows_all if r["st"] == "ok"]
print("ok", len(ok), "durations", np.percentile([r["dur"] for r in ok], [0, 50, 90, 100]).round(2) if ok else None, "with joint-space fallbacks:", sum(r["fb"] > 1 for r in ok))
for k in ("sx", "sy", "sa", "cx", "cy", "ca"): print("   ok range", k, round(min(r[k] for r in ok), 3), round(max(r[k] for r in ok), 3), " 5-95%:", np.percentile([r[k] for r in ok], [5, 95]).round(3))
json.dump(rows_all, open(d + "_rows.json", "w"))
