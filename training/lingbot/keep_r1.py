"""keep_r1.py (11-20) = keep_ic.py without the RLT files (runs with RLT off never have rlt/rlt_module.pt). Every 120 s:
1. a checkpoint global_step_N in OUT/checkpoints is complete when model/, optimizer/, extra_state/, rlt/rlt_module.pt and
   a full hf_ckpt (index + every shard it lists, nothing written in the last 90 s) exist;
2. complete N with N % KEEP_EVERY == 0 (or N == MAX_STEP) -> copy hf_ckpt + rlt/rlt_module.pt to
   DST/checkpoints/global_step_N/ (sizes verified, then COPY_OK), so DST/checkpoints/global_step_N/hf_ckpt is directly
   usable by the official deploy server (it reads DST/lingbotvla_cli.yaml three levels up);
3. DST/lingbotvla_cli.yaml = OUT/lingbotvla_cli.yaml with data.norm_stats_file set to NORM (the official deploy ignores
   the robot config's norm_stats and falls back to robotwin stats when this is empty);
4. prune (only with PRUNE=1): delete OUT/checkpoints/global_step_N for complete N older than the newest KEEP_LAST complete
   ones, never one that still waits for its copy. Log: one line per action.
Stops when OUT/SUPERVISOR_DONE exists and nothing is left to copy.
usage: keep_ic.py OUT DST NORM KEEP_EVERY KEEP_LAST MAX_STEP"""
import json, os, re, shutil, sys, time

OUT, DST, NORM = sys.argv[1:4]
KEEP_EVERY, KEEP_LAST, MAX_STEP = (int(x) for x in sys.argv[4:7])
PRUNE = os.environ.get("PRUNE") == "1"
CK = os.path.join(OUT, "checkpoints")


def log(msg):
    print(f"[keep] {time.strftime('%F %T')} {msg}", flush=True)


def complete(d):
    h = os.path.join(d, "hf_ckpt")
    need = [os.path.join(d, x) for x in ("model", "optimizer", "extra_state")]
    if not all(os.path.exists(p) for p in need):
        return False
    idx = os.path.join(h, "model.safetensors.index.json")
    if not os.path.isfile(idx):
        return False
    try:
        shards = set(json.load(open(idx))["weight_map"].values())
    except Exception:
        return False
    files = [os.path.join(h, s) for s in shards]
    if not all(os.path.isfile(f) and os.path.getsize(f) > 0 for f in files):
        return False
    newest = max(os.path.getmtime(os.path.join(h, f)) for f in os.listdir(h))
    return time.time() - newest > 90


def copy_ckpt(n, d):
    dst = os.path.join(DST, "checkpoints", f"global_step_{n}")
    if os.path.isfile(os.path.join(dst, "COPY_OK")):
        return True
    tmp = dst + ".partial"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    shutil.copytree(os.path.join(d, "hf_ckpt"), os.path.join(tmp, "hf_ckpt"))
    for root, _, fs in os.walk(os.path.join(d, "hf_ckpt")):
        for f in fs:
            a = os.path.join(root, f)
            b = os.path.join(tmp, "hf_ckpt", os.path.relpath(a, os.path.join(d, "hf_ckpt")))
            assert os.path.getsize(a) == os.path.getsize(b), (a, b)
    open(os.path.join(tmp, "COPY_OK"), "w").write(time.strftime("%F %T") + "\n")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.replace(tmp, dst)
    log(f"copied global_step_{n} -> {dst}")
    return True


def write_cli_yaml():
    src = os.path.join(OUT, "lingbotvla_cli.yaml")
    dst = os.path.join(DST, "lingbotvla_cli.yaml")
    if not os.path.isfile(src) or os.path.isfile(dst):
        return
    s = open(src).read()
    s2, k = re.subn(r"^(\s*)norm_stats_file:.*$", lambda m: f"{m.group(1)}norm_stats_file: {NORM}", s, count=1, flags=re.M)
    assert k == 1, "norm_stats_file key not found"
    os.makedirs(DST, exist_ok=True)
    open(dst, "w").write(s2)
    log(f"wrote {dst} (norm_stats_file = {NORM})")


log(f"start OUT={OUT} DST={DST} every={KEEP_EVERY} keep_last={KEEP_LAST} max={MAX_STEP} prune={PRUNE}")
while True:
    write_cli_yaml()
    steps = []
    if os.path.isdir(CK):
        for x in os.listdir(CK):
            m = re.fullmatch(r"global_step_(\d+)", x)
            if m:
                steps.append(int(m.group(1)))
    steps.sort()
    done = [n for n in steps if complete(os.path.join(CK, f"global_step_{n}"))]
    pending = []
    for n in done:
        if n % KEEP_EVERY == 0 or n == MAX_STEP:
            try:
                copy_ckpt(n, os.path.join(CK, f"global_step_{n}"))
            except Exception as e:
                log(f"copy global_step_{n} failed: {e!r}")
                pending.append(n)
    if PRUNE:
        for n in done[:-KEEP_LAST] if len(done) > KEEP_LAST else []:
            if n in pending:
                continue
            p = os.path.join(CK, f"global_step_{n}")
            log(f"prune {p}")
            shutil.rmtree(p, ignore_errors=True)
    if os.path.exists(os.path.join(OUT, "SUPERVISOR_DONE")) and not pending:
        if steps and steps[-1] not in done and time.time() - os.path.getmtime(os.path.join(OUT, "SUPERVISOR_DONE")) < 1800:
            time.sleep(60)   # final hf_ckpt (async) may still be written
            continue
        last = done[-1] if done else None
        if last is not None and last % KEEP_EVERY != 0 and last != MAX_STEP:
            copy_ckpt(last, os.path.join(CK, f"global_step_{last}"))
        log("supervisor done, keeper exits")
        break
    time.sleep(120)
