"""r1_ik.py: limit-aware damped-least-squares IK for the R1-7a TCP on SAPIEN's pinocchio model (numeric Jacobian,
clamped to joint limits every step, random in-limit restarts).  ik(model, p, R, q0) -> (q, pos_err, rot_err) or None."""
import numpy as np, sapien


def _log(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1); a = np.arccos(c)
    if a < 1e-9: return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if np.pi - a < 1e-6:                                            # 180 deg: axis from the symmetric part
        B = (R + np.eye(3)) / 2; ax = B[:, np.argmax(np.diag(B))]; return ax / np.linalg.norm(ax) * a
    return w / (2 * np.sin(a)) * a


class R1IK:
    def __init__(self, robot, link="tcp", prefix=""):
        """prefix (11-19, column dual-arm model): "left_" / "right_" -> joints <prefix>J1..7 and link <prefix>tcp."""
        if prefix and link == "tcp": link = prefix + "tcp"
        self.pin = robot.create_pinocchio_model(); self.li = [l.name for l in robot.get_links()].index(link)
        names = [j.name for j in robot.get_active_joints()]
        self.mask = np.array([n.startswith(prefix + "J") for n in names]); self.lim = robot.get_qlimits()

    def fk(self, q):
        self.pin.compute_forward_kinematics(np.asarray(q, float)); M = self.pin.get_link_pose(self.li).to_transformation_matrix()
        return M[:3, 3].copy(), M[:3, :3].copy()

    def _err(self, q, p, R):
        pc, Rc = self.fk(q); return np.r_[p - pc, 0.15 * _log(R @ Rc.T)]          # 1 rad ~ 15 cm

    def solve_from(self, q, p, R, iters=300, lam=1e-3):
        idx = np.where(self.mask)[0]; lo, hi = self.lim[idx, 0], self.lim[idx, 1]
        q = q.copy(); q[idx] = np.clip(q[idx], lo, hi)
        for _ in range(iters):
            e = self._err(q, p, R)
            if np.linalg.norm(e[:3]) < 2e-4 and np.linalg.norm(e[3:]) < 2e-4: break
            J = np.zeros((6, len(idx)))
            for k, j in enumerate(idx):
                dq = q.copy(); dq[j] += 1e-5; J[:, k] = -(self._err(dq, p, R) - e) / 1e-5
            step = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), e)
            n = np.linalg.norm(step)
            if n > 0.3: step *= 0.3 / n
            q[idx] = np.clip(q[idx] + step, lo, hi)
        pc, Rc = self.fk(q); return q, np.linalg.norm(p - pc), np.linalg.norm(_log(R @ Rc.T))

    def ik(self, p, R, q0, tries=24, seed=0, pos_tol=5e-4, rot_tol=5e-3):
        rs = np.random.RandomState(seed); best = None; idx = np.where(self.mask)[0]
        for k in range(tries):
            qi = q0.copy()
            if k: qi[idx] = rs.uniform(self.lim[idx, 0], self.lim[idx, 1])
            q, pe, re = self.solve_from(qi, np.asarray(p, float), np.asarray(R, float))
            if pe < pos_tol and re < rot_tol:
                d = np.linalg.norm(q[idx] - q0[idx])
                if best is None or d < best[0]: best = (d, q, pe, re)
                if k == 0: break                                     # converged from q0: the continuous solution
        return None if best is None else best[1:]


def track_ns(self, q, p, R, iters=30, lam=1e-3, ns_gain=0.5, ns_max=0.004):
    """One control tick of Cartesian tracking with joint-limit avoidance in the null space: the primary DLS solves the
    pose as solve_from does; one secondary step (at most ns_max rad per joint per tick) moves the redundant DOF towards
    mid-range, weighted so joints near a limit are pushed hardest.  Returns (q, pos_err, rot_err, limit_margin)."""
    idx = np.where(self.mask)[0]; lo, hi = self.lim[idx, 0], self.lim[idx, 1]; mid, rng = (lo + hi) / 2, (hi - lo)
    q = q.copy(); p = np.asarray(p, float); R = np.asarray(R, float)
    for it in range(iters):
        e = self._err(q, p, R)
        J = np.zeros((6, len(idx)))
        for k, j in enumerate(idx):
            dq = q.copy(); dq[j] += 1e-5; J[:, k] = -(self._err(dq, p, R) - e) / 1e-5
        Jp = J.T @ np.linalg.inv(J @ J.T + lam * np.eye(6))
        step = Jp @ e
        if it == 0:                                                   # secondary task once per tick
            d = (q[idx] - mid) / (rng / 2)                              # -1..1
            g = -d * (1 + 8 * np.clip(np.abs(d) - 0.7, 0, None) / 0.3)  # pull to mid, much harder in the outer 30 %
            ns = (np.eye(len(idx)) - Jp @ J) @ (ns_gain * g)
            mx = np.abs(ns).max()
            if mx > ns_max: ns *= ns_max / mx
            step = step + ns
        elif np.linalg.norm(e[:3]) < 1e-4 and np.linalg.norm(e[3:]) < 1e-4: break
        n = np.linalg.norm(step)
        if n > 0.3: step *= 0.3 / n
        q[idx] = np.clip(q[idx] + step, lo, hi)
    pc, Rc = self.fk(q)
    return q, np.linalg.norm(p - pc), np.linalg.norm(_log(R @ Rc.T)), float(np.min(np.minimum(q[idx] - lo, hi - q[idx])))


R1IK.track_ns = track_ns


def solve_anchored(self, q_anchor, p, R, margin=0.12, iters=400, anchor_gain=0.05, step_max=0.02):
    """Deterministic IK that stays in the anchor's posture family (11-19): local DLS from q_anchor, then null-space steps
    (pose held by the primary task) that push every joint at least `margin` rad inside its limits, with a weak pull back
    towards the anchor so the result varies continuously with (p, R).  Returns (q, pos_err, rot_err, limit_margin)."""
    idx = np.where(self.mask)[0]; lo, hi = self.lim[idx, 0], self.lim[idx, 1]
    p = np.asarray(p, float); R = np.asarray(R, float)
    q, _, _ = self.solve_from(q_anchor, p, R, iters=400)
    qa = np.asarray(q_anchor, float)[idx]
    for it in range(iters):
        e = self._err(q, p, R)
        J = np.zeros((6, len(idx)))
        for k, j in enumerate(idx):
            dq = q.copy(); dq[j] += 1e-5; J[:, k] = -(self._err(dq, p, R) - e) / 1e-5
        Jp = J.T @ np.linalg.inv(J @ J.T + 1e-4 * np.eye(6))
        dl, du = q[idx] - lo, hi - q[idx]
        g = np.clip(margin + 0.03 - dl, 0, None) - np.clip(margin + 0.03 - du, 0, None) + anchor_gain * (qa - q[idx])
        ns = (np.eye(len(idx)) - Jp @ J) @ g
        mx = np.abs(ns).max()
        if mx > step_max: ns *= step_max / mx
        q[idx] = np.clip(q[idx] + Jp @ e + ns, lo, hi)
        if it > 20 and min(dl.min(), du.min()) >= margin and np.linalg.norm(e[:3]) < 1e-4 and np.linalg.norm(e[3:]) < 1e-4 and mx < 1e-3: break
    q, pe, re = self.solve_from(q, p, R, iters=100)
    return q, pe, re, float(np.min(np.minimum(q[idx] - lo, hi - q[idx])))


R1IK.solve_anchored = solve_anchored


def track_cart(self, q, poses, margin=0.10, iters=60, step_max=0.01, verbose=False, band=0.22, q_goal=None, goal_gain=0.0):
    """v5 (11-19, teleop-like motion): follow a dense sequence of TCP poses the way an IK teleop does: each pose is solved
    from the previous joint vector (DLS = least joint motion); once per sample, joints closer than `margin` to a limit get a
    small push back (<= step_max rad, in the null space so the pose is kept) - no posture pull, so the arm never drifts or
    flips on its own.  Optional q_goal: a null-space pull towards that posture, ramping up along the move (the posture the
    move must end in, e.g. an insertion posture with room to the limits).  Returns (Q [n x nq], max pos err, max rot err,
    min limit margin, max per-sample |dq|)."""
    idx = np.where(self.mask)[0]; lo, hi = self.lim[idx, 0], self.lim[idx, 1]
    q = np.asarray(q, float).copy(); out, pe_m, re_m, mg_m, dq_m = [], 0.0, 0.0, 9.0, 0.0
    for n_, (p, R) in enumerate(poses):
        p = np.asarray(p, float); R = np.asarray(R, float); q_prev = q.copy()
        for it in range(iters):
            e = self._err(q, p, R)
            if it and np.linalg.norm(e[:3]) < 5e-5 and np.linalg.norm(e[3:]) < 5e-5: break
            J = np.zeros((6, len(idx)))
            for k, j in enumerate(idx):
                dq = q.copy(); dq[j] += 1e-5; J[:, k] = -(self._err(dq, p, R) - e) / 1e-5
            Jp = J.T @ np.linalg.inv(J @ J.T + 1e-4 * np.eye(6))
            st = Jp @ e
            if it == 0:
                dl, du = q[idx] - lo, hi - q[idx]
                g = np.clip(band - dl, 0, None) - np.clip(band - du, 0, None)      # starts pushing `band` rad before a limit
                g = g * (1 + 4 * (np.clip(margin - dl, 0, None) + np.clip(margin - du, 0, None)) / margin)   # much harder inside `margin`
                if q_goal is not None: g = g + goal_gain * (n_ + 1) / len(poses) * (np.asarray(q_goal, float)[idx] - q[idx])
                ns = (np.eye(len(idx)) - Jp @ J) @ g; mx = np.abs(ns).max()
                if mx > step_max: ns *= step_max / mx
                st = st + ns
            n2 = np.linalg.norm(st)
            if n2 > 0.2: st *= 0.2 / n2
            q[idx] = np.clip(q[idx] + st, lo, hi)
        pc, Rc = self.fk(q); pe, re = np.linalg.norm(p - pc), np.linalg.norm(_log(R @ Rc.T)); d_ = float(np.abs(q[idx] - q_prev[idx]).max())
        if verbose and (pe > 5e-4 or re > 5e-3 or d_ > 0.06): print(f"   track_cart sample {n_}/{len(poses)}: pos err {pe*1000:.2f} mm rot err {re:.4f} step {d_:.3f} q {np.round(q[idx], 3)}")
        pe_m = max(pe_m, pe); re_m = max(re_m, re)
        mg_m = min(mg_m, float(np.min(np.minimum(q[idx] - lo, hi - q[idx])))); dq_m = max(dq_m, d_)
        out.append(q.copy())
    return np.array(out), pe_m, re_m, mg_m, dq_m


R1IK.track_cart = track_cart
