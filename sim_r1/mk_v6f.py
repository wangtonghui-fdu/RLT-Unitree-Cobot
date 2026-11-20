"""mk_v6f.py (11-20): gen_dual_v6.py -> gen_dual_v6f.py (the user: "one video is too long, why do the arms move so slowly").
Same poses / grasps / order as v6; only the timing changes:
  * speed caps 0.30 m/s / 2.0 rad/s -> --tcap 0.70 m/s / --vcap 3.5 rad/s (joint-space fallback 1.2 -> --vjoint 2.4 rad/s)
  * the idle arm no longer waits: the right goes to its pre-handover pose while the left picks the charger up ("bg" legs with their
    own clock), the left goes to the strip while the right carries the plug over the socket (start delayed, clearance of the two
    synchronised paths checked), right aligns while the left presses, both rise together
  * shorter nominal times, insertion 3.0 -> --t_insert 1.8 s
  * ok additionally needs: no arm-arm contact frame and no charger-left-arm contact after the handover"""
import re, sys
src = open("gen_dual_v6.py").read()


def rep(old, new, count=1):
    global src
    assert src.count(old) == count, (src.count(old), old[:80])
    src = src.replace(old, new)


rep('"""gen_dual_v6.py (11-20): v6 =', '"""gen_dual_v6f.py (11-20): v6f = v6 with FAST timing (mk_v6f.py: higher speed caps, both arms move at the same time,\nshorter phases; poses, grasps and order unchanged).\ngen_dual_v6.py (11-20): v6 =')
rep('ap.add_argument("--overview", default="")\n', 'ap.add_argument("--overview", default="")\nap.add_argument("--vcap", type=float, default=3.5)       # peak joint speed of a Cartesian move (rad/s)\n'
    'ap.add_argument("--tcap", type=float, default=0.70)      # peak TCP speed (m/s)\nap.add_argument("--vjoint", type=float, default=2.4)     # peak joint speed of a joint-space move (rad/s)\n'
    'ap.add_argument("--t_insert", type=float, default=1.8)   # the slow final push (s)\n')
rep("def mj_time(Q, idx, L_, vcap=2.0, tcap=0.30):", "def mj_time(Q, idx, L_, vcap=a.vcap, tcap=a.tcap):")
rep("/ 1.2, margin=0.1)", "/ a.vjoint, margin=0.1)")

# ---- the procedure ----
i0 = src.index("SEGS = [\n"); i1 = src.index("\n]\n", i0) + 3
src = src[:i0] + '''T_CARRY, T_HOVER, D_HOVER = 1.3, 1.2, 0.25                           # right carry / left to the strip (starts D_HOVER later)
SEGS = [      # leg spec: callable / name = this segment's move; ("bg", name, nominal s[, delay]) = own clock, runs on; "wait" = let it finish
    ("settle", 0.2, {}, {}),
    ("L_approach", 0.9, {"left": lambda q: leg_or_joint("left", q, P_PRE, R_PICK, full(L["q_preL"], Q_HOME), "approach"), "right": ("bg", "right_pre", 1.5)}, {"right": Q_APP_R}),
    ("L_descend", 0.6, {"left": lambda q: cart_leg("left", q, P_GRASP, R_PICK, "vertical descent", tight=True)}, {}),
    ("L_grasp", 0.4, {}, {"left": Q_SQUEEZE_L}),
    ("L_lift", 0.5, {"left": lambda q: cart_leg("left", q, P_GRASP + np.array([0, 0, 0.12]), R_PICK, "vertical lift")}, {}),
    ("L_to_middle", 1.1, {"left": "to_handover", "right": "wait"}, {}),
    ("handover_ready", 0.15, {}, {}),
    ("R_insert_fingers", 0.8, {"right": "right_in"}, {}),
    ("R_close", 0.4, {}, {"right": Q_SQUEEZE_R}),
    ("L_release", 0.3, {}, {"left": Q_REL}),
    ("L_rise", 0.5, {"left": "left_up"}, {}),
    ("R_over_socket", T_CARRY, {"left": ("bg", "left_hover", T_HOVER, D_HOVER), "right": "right_carry"}, {"left": Q_CLOSED}),
    ("L_to_strip", 0.05, {"left": "wait"}, {}),
    ("L_hold", 0.5, {"left": "left_hold", "right": "right_correct"}, {}),
    ("R_descend", 0.5, {"right": lambda q: cart_leg("right", q, *ins_pose(H_MID), "descend", tight=True)}, {}),
    ("R_insert", a.t_insert, {"right": lambda q: cart_leg("right", q, *ins_pose(H_END), "insert", tight=True)}, {}),
    ("seated", 0.3, {}, {}),
    ("R_release", 0.5, {}, {"right": Q_REL}),                         # 0.3 s knocks the seated plug 1.6 deg over
    ("R_rise", 0.5, {"right": "up", "left": "up"}, {}),
    ("home", 1.5, {"right": "home", "left": "home"}, {"right": Q_WIDE, "left": Q_WIDE}),
    ("done", 0.2, {}, {}),
]
SEG_NAMES = [s_[0] for s_ in SEGS]
''' + src[i1:]
rep('home="双手回位", done="完成")\n', 'home="双手回位", done="完成")\nPH_CN.update(L_approach="左手移到充电头上方，右手同时就位", handover_ready="交接就位", R_over_socket="右手翻转移到插孔上方，左手伸向插排",\n'
    '             L_hold="左手扶住插排，右手对准", R_rise="双手抬起")\n')

# ---- choose_hold: the left moves while the right carries -> check the two synchronised paths ----
rep("def choose_hold(q):", "def choose_hold(q, carry):")
rep('pR, RR = ins_pose(H_END); qRi, pe_, re_, _ = K["right"].solve_anchored(q, pR, RR, margin=0.1)',
    'q_end = q.copy(); q_end[AIDX["right"]] = carry["Q"][-1][AIDX["right"]]; TR_ = np.ceil(max(T_CARRY, carry["T_min"] + 0.05) * 100 - 1e-6) / 100; aR = leg_path(carry["Q"])\n'
    '    pR, RR = ins_pose(H_END); qRi, pe_, re_, _ = K["right"].solve_anchored(q_end, pR, RR, margin=0.1)')
rep('''        c = min([ac_clear(x, q, Q_CLOSED, 0.008) for x in lg1["Q"][::6]] + [ac_clear(x, q, Q_CLOSED, 0.008) for x in lg2["Q"][::4]] +''',
    '''        TL_ = max(T_HOVER, lg1["T_min"] + 0.05); aL = leg_path(lg1["Q"]); sync = []
        for tt in np.arange(0.0, max(TR_, D_HOVER + TL_) + 1e-9, 1 / 60):          # both arms on their own clocks, as they will be executed
            uL = minjerk(min(max((tt - D_HOVER) / TL_, 0.0), 1.0)); uR = minjerk(min(max(tt / TR_, 0.0), 1.0))
            sync.append(ac_clear(aL(uL), aR(uR), Q_REL + (Q_CLOSED - Q_REL) * uL, 0.008))
        c = min(sync + [ac_clear(x, q_end, Q_CLOSED, 0.008) for x in lg2["Q"][::4]] +''')
rep('''print(f"   hold candidate {k_} {(dpress, tau, az, sg)} (0 = nominal): margin {m:.2f}, clearance {c * 1000:.1f} mm, socket at px {px[0, :2].round(0)}", flush=True)''',
    '''print(f"   hold candidate {k_} {(dpress, tau, az, sg)} (0 = nominal): margin {m:.2f}, clearance {c * 1000:.1f} mm (moving together {min(sync) * 1000:.1f}), socket at px {px[0, :2].round(0)}", flush=True)''')

# ---- named legs ----
rep('''        TR = charger_T() @ G_R; dR = -TR[:3, 1]
        return leg_or_joint("right", q, TR[:3, 3] - 0.08 * dR,''', '''        TR = T_CH_H @ G_R; dR = -TR[:3, 1]                               # the left brings the charger exactly to T_CH_H -> known in advance
        return leg_or_joint("right", q, TR[:3, 3] - 0.08 * dR,''')
rep('''        CUR["T_tcpR_ch"] = np.linalg.inv(tcp_T("right")) @ charger_T(); inhand_report("right after handover", CUR["T_tcpR_ch"], np.linalg.inv(G_R))
        p, R = ins_pose(H_HI); return leg_or_joint("right", q, p, R, full(Q_HOME, L["q_insR"]), "over the socket")
    if key == "left_hover": return choose_hold(q)''', '''        if CUR.get("carry_leg") is None:
            CUR["T_tcpR_ch"] = np.linalg.inv(tcp_T("right")) @ charger_T(); inhand_report("right after handover", CUR["T_tcpR_ch"], np.linalg.inv(G_R))
            p, R = ins_pose(H_HI); CUR["carry_leg"] = leg_or_joint("right", q, p, R, full(Q_HOME, L["q_insR"]), "over the socket")
        return CUR["carry_leg"]
    if key == "left_hover": return choose_hold(q, named_leg("right", "right_carry", q))''')

# ---- per-arm clocks ----
i0 = src.index('ARM = {s: dict(path=None, g0=Q_WIDE, g1=Q_WIDE) for s in SIDES}'); i1 = src.index('q_start = full(Q_HOME, Q_HOME); CMD["q"] = q_start.copy()')
src = src[:i0] + '''ARM = {s: dict(path=None, bg=False, t0=0.0, T=1.0, g0=Q_WIDE, g1=Q_WIDE, gt0=0.0, gT=1.0) for s in SIDES}       # each arm (and its gripper) has its own clock
CMD = dict(q=None)


def enter(i, t_seg):
    name, d, legs, grips = SEGS[i]; q = CMD["q"].copy(); T_need = d; fg = []
    for s in SIDES:
        f = legs.get(s); A = ARM[s]; busy = A["bg"] and t_seg < A["t0"] + A["T"] - 1e-6; g_now = float(np.mean(q[GIDX[s]]))
        if isinstance(f, str) and f == "wait": T_need = max(T_need, A["t0"] + A["T"] - t_seg) if busy else T_need; continue      # let its background move finish
        if busy:
            if f is not None or s in grips: raise SystemExit(f"{name}: the {s} arm is still on its background move")
            continue
        if f is None:
            A.update(path=(lambda qh: (lambda u: qh))(q.copy()), bg=False, t0=t_seg, T=1.0, g0=g_now, g1=grips.get(s, g_now), gt0=t_seg, gT=None); fg.append(s); continue
        bg = isinstance(f, tuple); key = f[1] if bg else f
        lg = named_leg(s, key, q) if isinstance(key, str) else key(q)
        if bg:
            Tb = np.ceil(max(f[2], lg["T_min"] + 0.05) * 100 - 1e-6) / 100; dl = f[3] if len(f) > 3 else 0.0
            A.update(path=leg_path(lg["Q"]), bg=True, t0=t_seg + dl, T=Tb, g0=g_now, g1=grips.get(s, g_now), gt0=t_seg + dl, gT=Tb)
            print(f"   {name}: {s} arm in the background, {dl:.2f} + {Tb:.2f} s (nominal {f[2]:.2f})", flush=True); BG_LOG.append((name, s, round(dl, 2), float(Tb)))
        else:
            A.update(path=leg_path(lg["Q"]), bg=False, t0=t_seg, T=None, g0=g_now, g1=grips.get(s, g_now), gt0=t_seg, gT=None); fg.append(s); T_need = max(T_need, lg["T_min"] + 0.05)
    T_need = float(np.ceil(T_need * 100 - 1e-6) / 100)
    for s in fg:
        if ARM[s]["T"] is None: ARM[s]["T"] = T_need
        ARM[s]["gT"] = T_need
    if T_need > d + 1e-9: print(f"   {name}: {d:.2f} -> {T_need:.2f} s", flush=True)
    SEGS[i] = (name, T_need, legs, grips); CUR["i"] = i


BG_LOG = []


def minjerk(s): return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def plan(t):
    t0 = 0.0
    for i, seg in enumerate(SEGS):
        if t < t0 + seg[1] or i == len(SEGS) - 1:
            if CUR["i"] != i: enter(i, t0)
            q = CMD["q"].copy()
            for s in SIDES:
                A = ARM[s]; u = minjerk(min(max((t - A["t0"]) / A["T"], 0.0), 1.0)); ug = minjerk(min(max((t - A["gt0"]) / A["gT"], 0.0), 1.0))
                q[AIDX[s]] = A["path"](u)[AIDX[s]]; q[GIDX[s]] = A["g0"] + (A["g1"] - A["g0"]) * ug
            return SEGS[i][0], np.clip(q, LIM[:, 0], LIM[:, 1])
        t0 += seg[1]


''' + src[i1:]

# ---- monitors ----
rep('''for k in range(10 ** 8):
    t = k * a.dt''', '''N_CL = [0, 9.0]                                                     # frames with the charger touching the LEFT arm after the handover, smallest separation
CLEAR_PH = {}                                                       # smallest left-right hull clearance per phase (mm, 12 = none within 12 mm)


def charger_left_contact():
    d = 9.0
    for c_ in scene.get_contacts():
        n_ = [x.entity.name for x in c_.bodies]
        if "charger" in n_ and any(SIDE_OF.get(x_) == "L" for x_ in n_): d = min([d] + [p.separation for p in c_.points])
    return d


for k in range(10 ** 8):
    t = k * a.dt''')
rep('''        rel, tilt = charger_state(); qp = robot.get_qpos(); dc = arm_contact()
''', '''        rel, tilt = charger_state(); qp = robot.get_qpos(); dc = arm_contact()
        if SEG_NAMES.index(ph) > SEG_NAMES.index("L_rise"):
            dl_ = charger_left_contact(); N_CL[1] = min(N_CL[1], dl_)
            if dl_ < 0.0005:
                N_CL[0] += 1; print(f"   CHARGER-LEFT CONTACT t={t:.2f} {ph} sep {dl_ * 1000:.1f} mm", flush=True) if N_CL[0] < 5 else None
        CLEAR_PH[ph] = min(CLEAR_PH.get(ph, 99.0), round(AC.clearance(qp)[0] * 1000, 1))
''')
rep("seated = abs(rel[2]) < 0.0015 and np.hypot(rel[0], rel[1]) < 0.001 and tilt < 3\n",
    "seated = abs(rel[2]) < 0.0015 and np.hypot(rel[0], rel[1]) < 0.001 and tilt < 3\nclean = N_CONTACT[0] == 0 and N_CL[0] == 0\n")
rep('meta = dict(seed=a.seed, ok=bool(seated),', 'meta = dict(seed=a.seed, ok=bool(seated and clean), seated=bool(seated), charger_left_contact_frames=N_CL[0], charger_left_min_sep_mm=round(min(N_CL[1], 0.02) * 1000, 2),\n'
    '            clear_by_phase_mm=CLEAR_PH, bg=BG_LOG, caps=dict(vcap=a.vcap, tcap=a.tcap, vjoint=a.vjoint, t_insert=a.t_insert), duration_s=round(sum(s_[1] for s_ in SEGS), 2),')
rep('version="dual_v6"', 'version="dual_v6f"')
rep('if seated and a.render and not a.snap:', 'if seated and clean and a.render and not a.snap:')
rep('print("RESULT", json.dumps({k: meta[k] for k in ("seed", "ok", "frames",', 'print("RESULT", json.dumps({k: meta[k] for k in ("seed", "ok", "seated", "duration_s", "charger_left_contact_frames", "charger_left_min_sep_mm", "frames",')
open("gen_dual_v6f.py", "w").write(src)
print("written", len(src.splitlines()), "lines")

# ---- (11-20, user: "can the plug and the socket be randomly tilted and moved?") randomisation ranges as arguments ----
src = open("gen_dual_v6f.py").read()
rep('ap.add_argument("--t_insert", type=float, default=1.8)   # the slow final push (s)\n',
    'ap.add_argument("--t_insert", type=float, default=1.8)   # the slow final push (s)\n'
    'ap.add_argument("--rand_strip", default="-0.015,0.015,-0.01,0.01,-5,5")     # strip offset from the layout: x0,x1,y0,y1 (m), yaw0,yaw1 (deg)\n'
    'ap.add_argument("--rand_charger", default="-0.015,0.015,-0.01,0.01,-5,5")   # charger offset from the layout, same format\n')
rep('RAND = dict(strip_xy=[L["strip"][0] + J_(-0.015, 0.015), L["strip"][1] + J_(-0.01, 0.01)], strip_yaw_deg=L["strip"][2] + J_(-5, 5),\n'
    '            charger_xy=[L["charger"][0] + J_(-0.015, 0.015), L["charger"][1] + J_(-0.01, 0.01)], charger_yaw_deg=L["charger"][2] + J_(-5, 5),\n',
    'RS_, RC_ = ([float(v) for v in x.split(",")] for x in (a.rand_strip, a.rand_charger))\n'
    'RAND = dict(strip_xy=[L["strip"][0] + J_(RS_[0], RS_[1]), L["strip"][1] + J_(RS_[2], RS_[3])], strip_yaw_deg=L["strip"][2] + J_(RS_[4], RS_[5]),\n'
    '            charger_xy=[L["charger"][0] + J_(RC_[0], RC_[1]), L["charger"][1] + J_(RC_[2], RC_[3])], charger_yaw_deg=L["charger"][2] + J_(RC_[4], RC_[5]),\n')
rep('SYAW, CYAW = np.radians(RAND["strip_yaw_deg"]), np.radians(RAND["charger_yaw_deg"])\n',
    'SYAW, CYAW = np.radians(RAND["strip_yaw_deg"]), np.radians(RAND["charger_yaw_deg"])\n'
    'print("RAND", json.dumps({k_: RAND[k_] for k_ in ("strip_xy", "strip_yaw_deg", "charger_xy", "charger_yaw_deg")}), flush=True)\n')
open("gen_dual_v6f.py", "w").write(src)
print("rand args added")
