"""patch_cache.py CODE_DIR: adds the teacher-target cache (LINGBOT_TEACHER_CACHE=<dir>, off by default) to a tree already patched
by patch_opt.py: lingbotvla/utils/teacher_cache.py + a lookup in front of the teacher block of the train loop (any miss -> the
original online path for that micro-batch).  LINGBOT_TEACHER_CACHE_VERIFY=N compares cached and online targets for N micro-batches."""
import sys, os, shutil
root = sys.argv[1]; here = os.path.dirname(os.path.abspath(__file__))
shutil.copy(os.path.join(here, "teacher_cache.py"), os.path.join(root, "lingbotvla/utils/teacher_cache.py"))
p = os.path.join(root, "tasks/vla/train_lingbotvla.py"); s = open(p).read()
if "TCACHE_PATCH" in s: print("already patched"); sys.exit()
a0 = '                        with torch.autocast("cuda", dtype=torch.bfloat16):\n                            depth_targets, cls_token = get_depth_target('
a1 = '                            future_video_target_rgb = future_pil_images\n'
assert s.count(a0) == 1 and s.count(a1) == 1
i0 = s.index(a0); i1 = s.index(a1) + len(a1); block = s[i0:i1]
ind = "".join("    " + l if l.strip() else l for l in block.splitlines(True))
new = '''                        _tc_hit = None  # TCACHE_PATCH
                        _tc_cmp = None
                        if _TEACHER_CACHE is not None and use_future_depth and use_future_video:
                            _tc_hit = _TEACHER_CACHE.lookup(pil_images, future_pil_images, future_video_effective_fps)
                            if _tc_hit is not None and _TEACHER_CACHE.verify_left > 0:
                                _tc_cmp, _tc_hit = _tc_hit, None
                        if _tc_hit is not None:
                            depth_targets, future_depth_targets, future_video_targets, future_video_current_dino = _tc_hit
                            future_video_current_rgb = pil_images
                            future_video_target_rgb = future_pil_images
                        else:
''' + ind + '''                            if _tc_cmp is not None:
                                _TEACHER_CACHE.report(_tc_cmp, (depth_targets, future_depth_targets, future_video_targets, future_video_current_dino))
'''
s = s[:i0] + new + s[i1:]
b0 = "        data_iterator = iter(train_dataloader)\n"
assert s.count(b0) == 1
s = s.replace(b0, b0 + '''        if _TEACHER_CACHE is None and os.environ.get("LINGBOT_TEACHER_CACHE"):  # TCACHE_PATCH
            try:
                from lingbotvla.utils.teacher_cache import TeacherCache
                _TEACHER_CACHE = TeacherCache(os.environ["LINGBOT_TEACHER_CACHE"], rank=args.train.local_rank, video_cfg=args.train.align_params.get("video", {}))
            except Exception as _tc_e:                                   # a missing / broken cache must never stop the training
                print(f"[TCACHE] not used ({type(_tc_e).__name__}: {_tc_e}); teachers run online", flush=True)
                os.environ.pop("LINGBOT_TEACHER_CACHE", None)
''')
c0 = "            _tm = dict(teacher=0.0, fwd=0.0, bwd=0.0)\n"
assert s.count(c0) == 1
d0 = "    use_future_video = use_depth_align and args.train.align_params.get('use_future_video', False)\n"
assert s.count(d0) == 1
s = s.replace(d0, d0 + "    _TEACHER_CACHE = None  # TCACHE_PATCH\n")
e0 = """                      f"| micro {len(micro_batches)} mem_peak_GiB {torch.cuda.max_memory_allocated() / 2**30:.1f}", flush=True)\n"""
assert s.count(e0) == 1
s = s.replace(e0, """                      f"| micro {len(micro_batches)} mem_peak_GiB {torch.cuda.max_memory_allocated() / 2**30:.1f}"
                      + (f" | tcache hit {_TEACHER_CACHE.n_hit} miss {_TEACHER_CACHE.n_miss}" if _TEACHER_CACHE is not None else ""), flush=True)\n""")
f0 = """            train_metrics = environ_meter.step(delta_time, global_step=global_step)\n"""
assert s.count(f0) == 1
s = s.replace(f0, f0 + """            if _TEACHER_CACHE is not None and args.train.local_rank == 0 and global_step % 200 == 0:  # TCACHE_PATCH
                print(f"[TCACHE] step {global_step}: micro-batch hits {_TEACHER_CACHE.n_hit} misses {_TEACHER_CACHE.n_miss}", flush=True)\n""")
if not os.path.exists(p + ".bak_tcache"): shutil.copy(p, p + ".bak_tcache")
open(p, "w").write(s); print("patched train loop + teacher_cache.py")
