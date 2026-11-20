"""probe_dual6.py (11-20, user: arms ~35 cm above the table; plug handed over with its tail towards the RIGHT ARM; strip at
centre-right-front; the left hand then holds the strip and its wrist camera watches the plug go in).
Model r1_column_dual_yup_s110.urdf (shoulders 1.10 m), table 0.75 m.  Posture family J2 < 1.6 (upper arm forward at about
shoulder height, forearm hanging), elbow at most 10 cm above the shoulder.
 left : straight down over the charger (flat, prongs LEFT), vertical descent, close, vertical lift, carry to the handover
        keeping it flat with its axis left-right (yaw 0 .. -20 deg)
 right: comes in horizontally from the RIGHT along the charger's axis, 40 mm open (upper finger between the left's fingers
        on the top face, lower finger under it), closes on the tail
 left : opens, rises; right: turns the plug prongs-down (gripper vertical) over the middle socket of the crosswise strip
 left : closes its gripper and holds the strip down 5-7 cm left of the socket, gripper tilted towards the socket so that
        its wrist camera sees the socket and the plug; right: lowers, inserts slowly
Every move Cartesian; left-vs-right hull clearance (arm_collision.ArmCollision).
modes: H  URDF NPROC TZ HOME                 handovers
       B1 URDF NPROC TZ HOME HO.json         charger resting poses (left pick) per handover
       B2 URDF NPROC TZ HOME HO.json         strip poses (right insertion + left hold with camera view) per handover"""
import sys, os, itertools, json, time, numpy as np, sapien
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r1_ik import R1IK, _log
from arm_collision import ArmCollision

MODE, URDF, NPROC, TZ = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]); QH = [float(v) for v in sys.argv[5].split(",")]
ARG6 = sys.argv[6] if len(sys.argv) > 6 else ""
CALIB = os.environ.get("CALIB", "/tmp/r1sim/r1_sim/camera_calib.json")
RZ = lambda t: np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
STRIP_TOP, SLOT = 0.0201, (0.0070, 0.0064)
R_REST = np.stack([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], 1).astype(float)   # flat, prongs +x (left), x_ch = -y, z_ch down
R_INS = np.stack([[1, 0, 0], [0, 0, 1], [0, -1, 0]], 1).astype(float)      # prongs down, x_ch along the strip
Y_LGRIP, Z_LGRIP, Y_RGRIP = 0.014, -0.026, 0.036
GQ = dict(open=-0.010, rel=-0.015, holdL=0.0005, holdR=0.008, app=0.0015, closed=0.0245)
CLR_HOLD, CLR_IN, CLR_MOVE = 0.002, 0.002, 0.004
ELB_UP = 0.10; L_TIP = 0.030


def rexp(w):
    t = np.linalg.norm(w)
    if t < 1e-12: return np.eye(3)
    k = w / t; Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(t) * Kx + (1 - np.cos(t)) * Kx @ Kx


def T_of(p, R): M = np.eye(4); M[:3, :3] = R; M[:3, 3] = p; return M


def G_left(fs):
    x = np.array([fs, 0, 0.0]); y = np.array([0, 0, -1.0]); return T_of([0, Y_LGRIP, Z_LGRIP], np.stack([x, y, np.cross(x, y)], 1))


def G_right(fz):
    x = np.array([0, 0, float(fz)]); y = np.array([0, 1.0, 0]); return T_of([0, Y_RGRIP, 0], np.stack([x, y, np.cross(x, y)], 1))


W = {}


def init():
    sc = sapien.Scene(); ld = sc.create_urdf_loader(); ld.fix_root_link = True; rb = ld.load(URDF)
    names = [j.name for j in rb.get_active_joints()]
    W.update(NQ=len(names), LIM=rb.get_qlimits(), K={s: R1IK(rb, prefix=f"{s}_") for s in ("left", "right")},
             I={s: np.array([names.index(f"{s}_J{j}") for j in range(1, 8)]) for s in ("left", "right")},
             G={s: np.array([i for i, n in enumerate(names) if n.startswith(f"{s}_dex_")]) for s in ("left", "right")},
             AC=ArmCollision(URDF), rs=np.random.RandomState(os.getpid()))
    pin = W["K"]["left"].pin; ln = [l.name for l in rb.get_links()]
    W["CH"] = {s: [ln.index(f"{s}_{n}") for n in ("Link2", "Link4")] for s in ("left", "right")}; W["pin"] = pin
    W["DB"] = {s: ln.index(f"{s}_dex_base_link") for s in ("left", "right")}
    C = json.load(open(CALIB)); T_l7_tcp = np.eye(4); T_l7_tcp[:3, 3] = [0, -0.1677, 0]
    T_l7_db = np.eye(4); T_l7_db[:3, :3] = np.diag([1.0, -1, -1]); T_l7_db[:3, 3] = [0, -0.0477, 0]
    W["T_DB_CAM"] = np.linalg.inv(T_l7_db) @ T_l7_tcp @ np.array(C["wrist"]["T_tcp_cam"]); W["KW"] = np.array(C["wrist"]["K"], float)


def margin(q, s): i = W["I"][s]; L = W["LIM"]; return float(np.min(np.minimum(q[i] - L[i, 0], L[i, 1] - q[i])))


def elbow_ok(q, s):
    W["pin"].compute_forward_kinematics(q); a, b = (W["pin"].get_link_pose(i).p for i in W["CH"][s]); return b[2] < a[2] + ELB_UP


def fam_ok(q, s): return q[W["I"][s]][1] < 1.6


def cam_px(q, s, P):
    """Pixel (u, v) and depth of world points P (n x 3) in arm s's wrist camera (pinhole, 640 x 480)."""
    W["pin"].compute_forward_kinematics(q); Tc = W["pin"].get_link_pose(W["DB"][s]).to_transformation_matrix() @ W["T_DB_CAM"]
    pc = (np.linalg.inv(Tc) @ np.c_[P, np.ones(len(P))].T).T[:, :3]; K = W["KW"]
    return np.c_[K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2], K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2], pc[:, 2]]


def compose(qL, qR, gL, gR):
    q = np.zeros(W["NQ"]); q[W["I"]["left"]] = qL[W["I"]["left"]]; q[W["I"]["right"]] = qR[W["I"]["right"]]
    q[W["G"]["left"]] = gL; q[W["G"]["right"]] = gR; return q


def clr_static(qL, qR, gL, gR): return W["AC"].clearance(compose(qL, qR, gL, gR))[0]


def clr_path(s, Q, q_other, g_self, g_other, every=4):
    worst = 9.0
    for k in list(range(0, len(Q), every)) + [len(Q) - 1]:
        qL, qR = (Q[k], q_other) if s == "left" else (q_other, Q[k]); gL, gR = (g_self, g_other) if s == "left" else (g_other, g_self)
        worst = min(worst, clr_static(qL, qR, gL, gR))
    return worst


def line(p0, R0, p1, R1):
    w = _log(R0.T @ R1); L = np.linalg.norm(p1 - p0); n = int(max(30, L / 0.003, np.linalg.norm(w) / np.radians(1.5)))
    return [(p0 + (p1 - p0) * u, R0 @ rexp(u * w)) for u in np.linspace(0, 1, n + 1)[1:]]


def leg(s, q, p1, R1, tight=False, q_goal=None):
    K = W["K"][s]; p0, R0 = K.fk(q); p1 = np.asarray(p1, float)
    Q, pe, re, mg, dq = K.track_cart(q, line(p0, R0, p1, R1), margin=0.08, q_goal=q_goal, goal_gain=0.0 if q_goal is None else 0.3)
    pf, Rf = K.fk(Q[-1]); tf = (5e-4, 5e-3) if tight else (1e-3, 1e-2)
    ok = np.linalg.norm(pf - p1) < tf[0] and np.linalg.norm(_log(R1 @ Rf.T)) < tf[1] and dq < 0.06 and (pe < 5e-4 and re < 5e-3 if tight else pe < 3e-3 and re < 2e-2)
    Q = np.r_[q[None], Q]; i = W["I"][s]
    return Q, bool(ok and mg > 0.025), float(mg), float(np.abs(np.diff(Q[:, i], axis=0)).sum())


def jpath(qa, qb, n=40): return np.array([qa + (qb - qa) * u for u in np.linspace(0, 1, n + 1)])


def ik(s, T, seeds=(), n_rand=6, mmin=0.08):
    K, I, L = W["K"][s], W["I"][s], W["LIM"]; best = None
    for q0 in [np.asarray(x, float) for x in seeds] + [None] * n_rand:
        if q0 is None: q0 = np.zeros(W["NQ"]); q0[I] = W["rs"].uniform(L[I, 0], L[I, 1])
        q, pe, re = K.solve_from(q0, T[:3, 3], T[:3, :3], iters=200)
        if pe > 3e-4 or re > 3e-3: continue
        q, pe, re, m = K.solve_anchored(q, T[:3, 3], T[:3, :3], margin=0.15)
        if pe > 3e-4 or re > 3e-3 or m < mmin or not elbow_ok(q, s) or not fam_ok(q, s): continue
        if best is None or m > best[1]: best = (q, m)
    return best


def home():
    q = np.zeros(W["NQ"]); q[W["I"]["left"]] = QH; q[W["I"]["right"]] = QH; return q


# ------------------------------------------------------------------ H
def work_H(item):
    hx, hy, hz, yaw, fz, fs = item
    if not W: init()
    qh = home(); Tch = T_of([hx, hy, TZ + hz], RZ(np.radians(yaw)) @ R_REST)
    TL, TR = Tch @ G_left(fs), Tch @ G_right(fz)
    bl = ik("left", TL, seeds=[qh])
    if not bl: return dict(fail="ik_left")
    br = ik("right", TR, seeds=[qh])
    if not br: return dict(fail="ik_right")
    qL, mL = bl; qR, mR = br
    c_hold = clr_static(qL, qR, GQ["holdL"], GQ["holdR"])
    if c_hold < CLR_HOLD: return dict(fail="hold")
    dR = -TR[:3, 1]; bpre = ik("right", T_of(TR[:3, 3] - 0.08 * dR, TR[:3, :3]), seeds=[qR], n_rand=2)
    if not bpre: return dict(fail="pre")
    Qin, ok, mgin, _ = leg("right", bpre[0], TR[:3, 3], TR[:3, :3], tight=True, q_goal=qR)
    if not ok: return dict(fail="in")
    c_in = clr_path("right", Qin, qL, GQ["app"], GQ["holdL"])
    if c_in < CLR_IN: return dict(fail="clr_in")
    pL, RL = W["K"]["left"].fk(qL); up = None
    for dv in ([0, 0, 0.08], [0, 0, 0.06], [0.03, 0, 0.06], [0, 0.03, 0.06]):
        Qup, ok, mup, _ = leg("left", qL, pL + dv, RL)
        if not ok: continue
        c_up = clr_path("left", Qup, Qin[-1], GQ["rel"], GQ["holdR"])
        if c_up >= CLR_MOVE: up = (Qup, c_up, dv); break
    if up is None: return dict(fail="left_up")
    Qup, c_up, dv_up = up; I = W["I"]
    return dict(item=list(item), ok=True, T_ch=Tch.round(6).tolist(), qL=qL[I["left"]].round(4).tolist(), qR_grasp=Qin[-1][I["right"]].round(4).tolist(),
                qR_pre=bpre[0][I["right"]].round(4).tolist(), qL_up=Qup[-1][I["left"]].round(4).tolist(), left_up=list(dv_up), mL=round(mL, 3), mR=round(mR, 3),
                clr=dict(hold=round(c_hold, 4), into=round(c_in, 4), up=round(c_up, 4)),
                score=round(min(mL, mR, bpre[1]) + 2 * min(c_hold, c_in, c_up) - 0.004 * abs(yaw), 4))


# ------------------------------------------------------------------ B1 left pick
def work_B1(item):
    h, cx, cy, psi = item
    if not W: init()
    H = HO[h]; qh = home(); out = dict(h=h, charger=[cx, cy, psi], ok=False); fs = H["item"][5]
    CH = T_of([cx, cy, TZ + 0.0135], RZ(np.radians(psi)) @ R_REST); TG = CH @ G_left(fs)
    b = ik("left", T_of(TG[:3, 3] + [0, 0, 0.10], TG[:3, :3]), seeds=[qh], n_rand=6)
    if not b: return out
    qa, m0 = b
    Qa, oka, _, _ = leg("left", qh, TG[:3, 3] + [0, 0, 0.10], TG[:3, :3], q_goal=qa)          # home -> above the charger, Cartesian if possible
    if oka: qa = Qa[-1]
    Q1, ok, m1, _ = leg("left", qa, TG[:3, 3], TG[:3, :3], tight=True)
    if not ok: return out
    Q2, ok, m2, _ = leg("left", Q1[-1], TG[:3, 3] + [0, 0, 0.12], TG[:3, :3])
    if not ok: return out
    qLH = qh.copy(); qLH[W["I"]["left"]] = H["qL"]; TL = T_of(*W["K"]["left"].fk(qLH))
    Q3, ok, m3, tr3 = leg("left", Q2[-1], TL[:3, 3], TL[:3, :3], q_goal=qLH)
    if not ok: return out
    c = min(clr_path("left", Q, qh, GQ["holdL"], GQ["open"]) for Q in (Q1, Q2, Q3))
    out.update(ok=bool(c >= CLR_MOVE), margin=round(min(m0, m1, m2, m3), 3), travel=round(tr3, 2), clr=round(c, 4), cart_approach=bool(oka),
               q_preL=qa[W["I"]["left"]].round(4).tolist(), end_dq=round(float(np.abs(Q3[-1][W["I"]["left"]] - np.array(H["qL"])).max()), 3))
    return out


# ------------------------------------------------------------------ B2 right insertion + left hold with the camera on the socket
def hold_candidates(S, sy_r):
    """Left 'hold the strip' poses: closed finger tips on the strip top dpress left (+x_strip) of the socket, gripper tilted tau
    from vertical towards the socket (azimuth az about the vertical), finger axis horizontal; both rolls."""
    out = []
    for dpress, tau, az, sg in itertools.product((0.050, 0.058, 0.066), (35, 45, 55), (0, 25, -25, 50, -50), (1, -1)):
        pc = S + RZ(sy_r) @ np.array([dpress, 0, 0]); h = RZ(sy_r + np.radians(az)) @ np.array([-1.0, 0, 0])
        t = np.radians(tau); d = np.sin(t) * h - np.cos(t) * np.array([0, 0, 1.0]); fx = sg * np.cross([0, 0, 1.0], h)
        R = np.stack([fx, -d, np.cross(fx, -d)], 1); out.append(((dpress, tau, az, sg), pc - d * L_TIP + np.array([0, 0, 0.004]), R))
    return out


T_TCP_DB_L = np.eye(4); T_TCP_DB_L[:3, :3] = np.diag([-1.0, -1, 1]); T_TCP_DB_L[:3, 3] = [0, 0.12, 0]   # left tcp -> left dex base (gripper_mount rpy 0 0 pi; origins 0.0477 / 0.1677 on the axis)


def view_pose(p, R, S):
    """Same test as view_score, from the left TCP pose alone (the wrist camera is rigid to the gripper)."""
    Tc = T_of(p, R) @ T_TCP_DB_L @ W["T_DB_CAM"]; P = np.array([S, S + [0, 0, 0.03], S + [0, 0, 0.06]]); K = W["KW"]
    pc = (np.linalg.inv(Tc) @ np.c_[P, np.ones(3)].T).T[:, :3]
    if (pc[:, 2] < 0.05).any() or (pc[:, 2] > 0.40).any(): return None
    px = np.c_[K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2], K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]]
    if not (80 < px[0, 0] < 560 and 60 < px[0, 1] < 440 and 60 < px[1, 0] < 580 and 30 < px[1, 1] < 450): return None
    return float(np.hypot(px[0, 0] - 320, px[0, 1] - 260)), px.round(0).tolist()


def view_score(qP, S):
    px = cam_px(qP, "left", np.array([S, S + [0, 0, 0.03], S + [0, 0, 0.06]]))
    if (px[:, 2] < 0.05).any() or (px[:, 2] > 0.40).any(): return None
    if not (80 < px[0, 0] < 560 and 60 < px[0, 1] < 440 and 60 < px[1, 0] < 580 and 30 < px[1, 1] < 450): return None
    return float(np.hypot(px[0, 0] - 320, px[0, 1] - 260)), px.round(0).tolist()


def work_B2(item):
    h, sx, sy, syaw, twin = item
    if not W: init()
    H = HO[h]; qh = home(); I = W["I"]; K = W["K"]; out = dict(h=h, strip=[sx, sy, syaw], twin=twin, ok=False, stage=0)
    qR = qh.copy(); qR[I["right"]] = H["qR_grasp"]; qLup = qh.copy(); qLup[I["left"]] = H["qL_up"]
    TR = T_of(*K["right"].fk(qR)); Tch = np.array(H["T_ch"]); T_tcp_ch = np.linalg.inv(TR) @ Tch
    sy_r = np.radians(syaw); S = np.array([sx, sy, TZ - 0.0008]) + RZ(sy_r) @ np.array([SLOT[0], SLOT[1], STRIP_TOP])
    Rc = RZ(sy_r + twin * np.pi) @ R_INS
    P_hi = T_of(S + [0, 0, float(os.environ.get("HHI", 0.09))], Rc) @ np.linalg.inv(T_tcp_ch); P1 = T_of(S + [0, 0, 0.045], Rc) @ np.linalg.inv(T_tcp_ch); P2 = T_of(S + [0, 0, -0.0005], Rc) @ np.linalg.inv(T_tcp_ch)
    bP = ik("right", P_hi, seeds=[qR], n_rand=4)
    if not bP: return out
    out["stage"] = 1
    Qc, ok, mc, trc = leg("right", qR, P_hi[:3, 3], P_hi[:3, :3], q_goal=bP[0]); joint_carry = False
    if not ok: Qc = jpath(qR, bP[0], 60); mc = bP[1]; trc = float(np.abs(bP[0][I["right"]] - qR[I["right"]]).sum()); joint_carry = True
    if clr_path("right", Qc, qLup, GQ["holdR"], GQ["rel"]) < CLR_MOVE: return out
    out["stage"] = 2
    Qd1, ok1, m1, _ = leg("right", Qc[-1], P1[:3, 3], P1[:3, :3], tight=True)
    if not ok1: return out
    Qd2, ok2, m2, tr2 = leg("right", Qd1[-1], P2[:3, 3], P2[:3, :3], tight=True)
    if not ok2: return out
    out["stage"] = 3; Qins = np.concatenate([Qd1, Qd2]); best = None; n_ik = n_view = n_path = 0
    cands = [(view_pose(p, R, S), par, p, R) for par, p, R in hold_candidates(S, sy_r)]
    cands = sorted([c for c in cands if c[0] is not None], key=lambda c: c[0][0])[:10]      # best-centred views first
    for vs0, par, p, R in cands:
        n_view += 1
        b = ik("left", T_of(p, R), seeds=[qLup], n_rand=6, mmin=0.06)
        if not b: continue
        n_ik += 1; qP, mP = b
        vs = view_score(qP, S)
        if vs is None: continue
        bh = ik("left", T_of(p + [0, 0, 0.05], R), seeds=[qP], n_rand=0, mmin=0.04)
        if not bh: continue
        Qh, okh, mh, _ = leg("left", qLup, p + [0, 0, 0.05], R, q_goal=bh[0])
        if not okh: Qh = jpath(qLup, bh[0], 60)
        Qp, okp, mp, _ = leg("left", Qh[-1], p, R, tight=True)
        if not okp: continue
        c1 = min(clr_path("left", Qh, Qc[-1], GQ["closed"], GQ["holdR"]), clr_path("left", Qp, Qc[-1], GQ["closed"], GQ["holdR"]))
        if c1 < CLR_MOVE: continue
        c2 = min(clr_path("right", Qins, Qp[-1], GQ["holdR"], GQ["closed"]), clr_static(Qp[-1], Qins[-1], GQ["closed"], GQ["rel"]))
        if c2 < CLR_MOVE: continue
        n_path += 1; sc = -vs[0] / 400 + 0.5 * min(mP, mp) + 5 * min(c1, c2) + (0.05 if okh else 0)
        if best is None or sc > best[0]:
            best = (sc, dict(hold=dict(par=list(par), p=p.round(4).tolist(), R=R.round(5).tolist(), q=Qp[-1][I["left"]].round(4).tolist(), margin=round(min(mP, mp), 3),
                                       cart_hover=bool(okh), px=vs[1], clr=[round(c1, 4), round(c2, 4)])))
    out["hold_counts"] = [n_ik, n_view, n_path]
    if best is None: return out
    out.update(ok=True, stage=4, joint_carry=joint_carry, margin=round(min(mc, m1, m2), 3), travel=round(trc + tr2, 2), q_insR=Qd2[-1][I["right"]].round(4).tolist(), **best[1])
    return out


HO = json.load(open(ARG6)) if MODE in ("B1", "B2") else []

if __name__ == "__main__":
    t0 = time.time(); from collections import Counter
    if MODE == "H":
        items = list(itertools.product((0.06, 0.10, 0.14, 0.18), (-0.25, -0.28, -0.31, -0.34, -0.37, -0.40), (0.06, 0.08, 0.11, 0.14, 0.18, 0.22), (0, -5, -10, -15, -20), (1, -1), (1, -1)))
        print(len(items), "handover candidates", flush=True)
        with Pool(NPROC) as pool: res = pool.map(work_H, items, chunksize=2)
        print("fails:", Counter(r["fail"] for r in res if not r.get("ok")).most_common(), flush=True)
        rows = sorted([r for r in res if r.get("ok")], key=lambda r: -r["score"]); json.dump(rows, open("ho6.json", "w"))
        print(len(rows), "handovers; by yaw:", sorted(Counter(r["item"][3] for r in rows).items()), flush=True)
        for r in rows[:12]: print(json.dumps({k: r[k] for k in ("item", "mL", "mR", "clr", "score")}))
    elif MODE == "B1":
        items = [(h, cx, cy, psi) for h in range(len(HO)) for cx in (0.22, 0.26, 0.30, 0.34, 0.38) for cy in (-0.24, -0.27, -0.30, -0.33, -0.36) for psi in (-8, 0, 8)]
        print(len(items), "B1 items", flush=True)
        with Pool(NPROC) as pool: rows = list(pool.imap_unordered(work_B1, items, chunksize=2))
        json.dump(rows, open("b1_6.json", "w"))
        print("ok per handover:", sorted(Counter(r["h"] for r in rows if r["ok"]).items()))
    elif MODE == "B2":
        items = [(h, sx, sy, yaw, tw) for h in range(len(HO)) for sx in (-0.22, -0.19, -0.16, -0.13, -0.10) for sy in (-0.30, -0.33, -0.36, -0.39, -0.42) for yaw in (-6, 0, 6) for tw in (0, 1)]
        if os.environ.get("COARSE"): items = [(h, sx, sy, 0, tw) for h in range(len(HO)) for sx in (-0.12, -0.14, -0.16, -0.18) for sy in (-0.30, -0.32, -0.34, -0.36, -0.38) for tw in (0, 1)]
        if os.environ.get("FINE"):
            cx_, cy_ = [float(v) for v in os.environ["FINE"].split(",")]
            items = [(h, round(cx_ + dx, 3), round(cy_ + dy, 3), yaw, tw) for h in range(len(HO)) for dx in (-0.02, -0.01, 0, 0.01, 0.02) for dy in (-0.01, 0, 0.01) for yaw in (-6, 0, 6) for tw in (0, 1)]
        print(len(items), "B2 items", flush=True)
        rows = []
        with Pool(NPROC) as pool:
            for k_, r_ in enumerate(pool.imap_unordered(work_B2, items, chunksize=1)):
                rows.append(r_)
                if k_ % 40 == 0: print(f"  {k_}/{len(items)} done, ok so far {sum(1 for x in rows if x['ok'])} ({time.time() - t0:.0f} s)", flush=True)
        json.dump(rows, open(os.environ.get("B2_OUT", "b2_6.json"), "w"))
        print("stage reached:", sorted(Counter(r["stage"] for r in rows).items()))
        print("hold counts (ik, view, path) summed over stage>=3:", np.sum([r["hold_counts"] for r in rows if "hold_counts" in r], axis=0).tolist() if any("hold_counts" in r for r in rows) else None)
        print("ok per handover:", sorted(Counter(r["h"] for r in rows if r["ok"]).items()))
    print(f"done in {time.time() - t0:.0f} s", flush=True)
