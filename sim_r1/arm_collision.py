"""arm_collision.py (11-20, user: "the two grippers crash into each other"): left-arm vs right-arm clearance for the column
dual-arm model, using PhysX itself on the URDF collision shapes (convex hulls; conservative for the Dex1 fingers).
Separate scene; left_* links and right_* links get disjoint self-collision ids so only left-vs-right pairs (and nothing
against the stand) produce contacts; contact offset `offset` -> contacts are reported up to that distance apart.
clearance(q) -> smallest separation (m) between any left and right link (offset if none within `offset`; < 0 = overlap),
plus the closest link pair.  usage as a script: arm_collision.py URDF LOG.json  (replays a gen_dual_col log)."""
import sys, json, numpy as np, sapien


class ArmCollision:
    def __init__(self, urdf, offset=0.012):
        self.offset = offset
        self.scene = sapien.Scene(); self.scene.set_timestep(1e-5)
        ld = self.scene.create_urdf_loader(); ld.fix_root_link = True; self.robot = ld.load(urdf)
        self.names = [j.name for j in self.robot.get_active_joints()]
        self.side = {}
        for l in self.robot.get_links():
            l.disable_gravity = True
            s = "L" if l.name.startswith("left_") else "R" if l.name.startswith("right_") else "S"
            bits = {"L": 1 << 29, "R": 1 << 28, "S": (1 << 29) | (1 << 28)}[s]
            for sh in l.get_collision_shapes():
                g = sh.get_collision_groups(); g[2] = bits; sh.set_collision_groups(g); sh.set_contact_offset(offset); sh.set_rest_offset(0.0)
            self.side[l.entity.name] = (s, l.name)
        for j in self.robot.get_active_joints(): j.set_drive_properties(0, 0)

    def clearance(self, q):
        self.robot.set_qpos(np.asarray(q, float)); self.robot.set_qvel(np.zeros(len(self.names)))
        self.scene.step()
        best, pair = self.offset, None
        for c in self.scene.get_contacts():
            a, b = (self.side.get(x.entity.name, ("?", "")) for x in c.bodies)
            if {a[0], b[0]} != {"L", "R"}: continue
            for p in c.points:
                if p.separation < best: best, pair = p.separation, (a[1], b[1])
        return float(best), pair

    def path_clearance(self, Q, every=1):
        worst = (self.offset, None, -1)
        for k in range(0, len(Q), every):
            d, pr = self.clearance(Q[k])
            if d < worst[0]: worst = (d, pr, k)
        return worst


if __name__ == "__main__":
    ac = ArmCollision(sys.argv[1]); L = json.load(open(sys.argv[2]))
    nm = ac.names; IL = [nm.index(f"left_J{j}") for j in range(1, 8)]; IR = [nm.index(f"right_J{j}") for j in range(1, 8)]
    bad = []
    for r in L:
        q = np.zeros(len(nm)); q[IL] = r["q_left"]; q[IR] = r["q_right"]
        d, pr = ac.clearance(q)
        if d < 0.005: bad.append((r["t"], r["phase"], round(d * 1000, 1), pr))
    print(len(bad), "frames with left-right clearance < 5 mm (of", len(L), ")")
    last = None
    for b in bad:
        if last is None or b[1] != last: print("  ", b)
        last = b[1]
    print("by phase:", {p: sum(1 for b in bad if b[1] == p) for p in dict.fromkeys(b[1] for b in bad)})
