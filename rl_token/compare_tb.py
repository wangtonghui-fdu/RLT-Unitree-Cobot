"""Compare per-step VLA scalars of two runs (tensorboard events): bitwise equality / max relative difference."""
import sys, glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
def load(d):
    out = {}
    for f in sorted(glob.glob(f"{d}/runs/**/events.out.tfevents.*", recursive=True)):
        ea = EventAccumulator(f, size_guidance={"scalars": 0}); ea.Reload()
        for tag in ea.Tags()["scalars"]:
            for e in ea.Scalars(tag):
                out.setdefault(tag, {}).setdefault(e.step, e.value)   # first occurrence per step (run a before b)
    return out
A, B = load(sys.argv[1]), load(sys.argv[2])
steps = range(1, int(sys.argv[3]) + 1)
tags = ["training/loss", "training/vla_loss", "training/depth_loss", "training/future_depth_loss", "training/future_video_loss", "training/grad_norm"]
for t in tags:
    a = [A.get(t, {}).get(s) for s in steps]; b = [B.get(t, {}).get(s) for s in steps]
    eq = sum(1 for x, y in zip(a, b) if x is not None and x == y)
    rel = max((abs(x - y) / max(abs(x), 1e-12) for x, y in zip(a, b) if x is not None and y is not None), default=float("nan"))
    print(f"{t:32s} identical {eq}/{len(a)}  max rel diff {rel:.2e}  first5 A {[round(x,6) if x else x for x in a[:5]]}  B {[round(y,6) if y else y for y in b[:5]]}")
print("rlt tags in A:", sorted(k for k in A if k.startswith("rlt/"))[:12])
