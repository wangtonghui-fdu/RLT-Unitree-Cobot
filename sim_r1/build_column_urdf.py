"""build_column_urdf.py (11-19): Unitree R1-7a dual-arm column stand ("双臂版 R1-7a + 立柱") as one URDF, Dex1-1 gripper on both arms.

Stand (vendor parts list): 100x100x1200 mm aluminium profile column, 750x750x10 mm chassis plate (X-shaped, as in the product
photo), 4 casters 140x160x90 mm, chassis adapter D130x80 mm.  Stacked: casters 0..0.14 m, plate 0.14..0.15, adapter
0.15..0.23, column 0.23..1.43.  Arms: R1-7a right (sim URDF r1_7a_dex1 = SDK V1.1, J6 +-0.524) and left (SDK
R1_Robotic_Arm_L_V1.0-1027: J1/J3/J5 axes reversed), base flange on a 10 mm plate on the column's side faces, J1 axis
horizontal pointing outwards, SHOULDER_Z above the floor (photo: flange centre ~7 cm under the column top).
Mount: base z (J1 axis) horizontal and outwards, base y down at J1 = 0 (upper arm hanging, as on Unitree's R1 humanoid,
unitree_ros r1_a7: J1 ~ shoulder pitch, J2 ~ roll, J3 ~ yaw, J3's range -0.35..3.49 = yaw +-1.92 around pi/2).
SIDE (11-19, elbow_check.py): with the upper arm hanging, the elbow of the SDK "R" arm can only bend towards +x_base (J3 in
-0.35..3.49 never reaches -pi/2), and x_base = y_base x z_base points BACKWARDS when the R arm hangs on the robot's right side
(v1 of this model: every forearm bent backwards).  Elbows bend forwards only with the R-URDF arm on the robot's LEFT side
and the L-URDF arm on the RIGHT side (= vendor naming seen from in front of the robot?) - so that is how the arms are
mounted here; link names say the physical side (right_* = robot's right, -x), the comment in the URDF says the source.
World: floor z = 0, robot faces -y (same convention as the table-top sim: the work area is at -y), right arm at -x.
usage: build_column_urdf.py [OUT_DIR]   (default third_party/r1_sim/column_dual)
"""
import sys, os, copy, shutil, numpy as np, xml.etree.ElementTree as ET

TP = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
# --yup (11-19, user: "the arm comes in L and R versions"): flange turned 180 deg about the J1 axis -> base y UP, the upper arm
# hangs at J2 = pi (the other end of J2's 0..pi range); then the SDK R arm sits on the robot's RIGHT and the L arm on the LEFT.
YUP = "--yup" in sys.argv
if YUP: sys.argv.remove("--yup")
# --shoulder=Z (11-20, user: "lower the arms, about 35 cm above the table"): J1 axis height above the floor (default: 7 cm under
# the column top = 1.36 m); the arms slide on the column's T-slots.  Output name gets a _sNNN suffix (cm).
SH_ARG = next((x for x in sys.argv if x.startswith("--shoulder=")), None)
if SH_ARG: sys.argv.remove(SH_ARG)
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(TP, "r1_sim", "column_dual")
R_URDF = os.path.join(TP, "r1_sim", "urdf", "r1_7a_dex1.urdf"); R_MESH = os.path.join(TP, "r1_sim", "meshes")
L_DIR = os.path.join(TP, "r1-7a_sdk", "urdf", "ex", "R1_Robotic_Arm_L_V1.0-1027", "R1_Robotic_Arm_L_V1.0-1027")
L_URDF = os.path.join(L_DIR, "urdf", "R1_Robotic_Arm_L_V1.0-1027.urdf"); L_MESH = os.path.join(L_DIR, "meshes")

COL = 0.10; COL_H = 1.20; CASTER_H = 0.14; PLATE_T = 0.010; ADAPT_H = 0.08
Z_PLATE = CASTER_H; Z_ADAPT = Z_PLATE + PLATE_T; Z_COL = Z_ADAPT + ADAPT_H; Z_TOP = Z_COL + COL_H     # 0.14 / 0.15 / 0.23 / 1.43
SHOULDER_Z = float(SH_ARG.split("=")[1]) if SH_ARG else Z_TOP - 0.07   # arm flange centre (photo estimate 1.36 m; --shoulder= overrides)
MOUNT_T = 0.010                                                     # aluminium mounting plate between column face and flange
RIGHT_MOUNT = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]], float) # robot's right (-x): base x -> +y, y -> -z (down), z -> -x (out)
LEFT_MOUNT = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)  # robot's left (+x):  base x -> -y, y -> -z (down), z -> +x (out)
if YUP:
    RIGHT_MOUNT = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], float)  # robot's right: base x -> -y (fwd), y -> +z (up), z -> -x (out)
    LEFT_MOUNT = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)     # robot's left:  base x -> +y, y -> +z (up), z -> +x (out)


def rpy(R):
    p = np.arcsin(-np.clip(R[2, 0], -1, 1))
    if abs(np.cos(p)) > 1e-9: r, y = np.arctan2(R[2, 1], R[2, 2]), np.arctan2(R[1, 0], R[0, 0])
    else: r, y = np.arctan2(-R[1, 2], R[1, 1]), 0.0
    return f"{r:.9f} {p:.9f} {y:.9f}"


def S(*v): return " ".join(f"{x:.6g}" for x in v)


def visual(link, geom, xyz=(0, 0, 0), rp="0 0 0", rgba=(0.75, 0.76, 0.78, 1), name=None, collide=True):
    for tag in (("visual", "collision") if collide else ("visual",)):
        v = ET.SubElement(link, tag); ET.SubElement(v, "origin", xyz=S(*xyz), rpy=rp); g = ET.SubElement(v, "geometry")
        kind, dims = geom
        if kind == "box": ET.SubElement(g, "box", size=S(*dims))
        else: ET.SubElement(g, "cylinder", radius=S(dims[0]), length=S(dims[1]))
        if tag == "visual":
            m = ET.SubElement(v, "material", name=name or f"c_{'_'.join(str(int(c * 255)) for c in rgba)}"); ET.SubElement(m, "color", rgba=S(*rgba))


def inertial(link, m, xyz=(0, 0, 0), I=0.01):
    i = ET.SubElement(link, "inertial"); ET.SubElement(i, "origin", xyz=S(*xyz), rpy="0 0 0"); ET.SubElement(i, "mass", value=S(m))
    ET.SubElement(i, "inertia", ixx=S(I), ixy="0", ixz="0", iyy=S(I), iyz="0", izz=S(I))


def fixed(robot, name, parent, child, xyz, rp="0 0 0"):
    j = ET.SubElement(robot, "joint", name=name, type="fixed"); ET.SubElement(j, "origin", xyz=S(*xyz), rpy=rp)
    ET.SubElement(j, "parent", link=parent); ET.SubElement(j, "child", link=child)


def arm_elems(urdf, prefix, mesh_sub, keep=lambda n: True, drop_root=False):
    """Links/joints of an arm URDF, names prefixed, mesh paths -> ../meshes/<mesh_sub>/<file>."""
    r = ET.parse(urdf).getroot(); out = []
    for e in r:
        if e.tag not in ("link", "joint"): continue
        if not keep(e.get("name")): continue
        e = copy.deepcopy(e); e.set("name", prefix + e.get("name"))
        for t in ("parent", "child"):
            x = e.find(t)
            if x is not None: x.set("link", prefix + x.get("link"))
        for m in e.iter("mesh"): m.set("filename", f"../meshes/{mesh_sub}/{os.path.basename(m.get('filename'))}")
        for m in e.iter("material"):
            if m.get("name"): m.set("name", prefix + m.get("name"))
        out.append(e)
    return out


def main():
    os.makedirs(os.path.join(OUT, "urdf"), exist_ok=True)
    shutil.rmtree(os.path.join(OUT, "meshes"), ignore_errors=True)
    for sub, src, pat in (("sdkR", R_MESH, lambda f: not f.startswith("dex_")), ("sdkL", L_MESH, lambda f: f.endswith(".STL")), ("dex", R_MESH, lambda f: f.startswith("dex_"))):
        d = os.path.join(OUT, "meshes", sub); os.makedirs(d, exist_ok=True)
        for f in os.listdir(src):
            if pat(f) and f.endswith(".STL"): shutil.copy2(os.path.join(src, f), d)
    robot = ET.Element("robot", name="r1_7a_column_dual")

    # ---- stand ----
    alu, dark, black = (0.78, 0.79, 0.81, 1), (0.16, 0.16, 0.17, 1), (0.05, 0.05, 0.05, 1)
    base = ET.SubElement(robot, "link", name="stand_base")                    # root: chassis plate + casters + adapter
    L_BAR = 0.91; W_BAR = 0.15                                                # X plate inside a 750 x 750 square
    for a in (45, -45):
        visual(base, ("box", (L_BAR, W_BAR, PLATE_T)), (0, 0, Z_PLATE + PLATE_T / 2), f"0 0 {np.radians(a):.9f}", (0.93, 0.93, 0.94, 1), "white")
    visual(base, ("cylinder", (0.065, ADAPT_H)), (0, 0, Z_ADAPT + ADAPT_H / 2), rgba=alu, name="alu")
    rc = L_BAR / 2 - 0.07                                                     # casters under the bar tips
    for sx in (-1, 1):
        for sy in (-1, 1):
            cx, cy = sx * rc / np.sqrt(2), sy * rc / np.sqrt(2)
            visual(base, ("box", (0.09, 0.09, 0.008)), (cx, cy, CASTER_H - 0.004), rgba=dark, name="dark")
            visual(base, ("box", (0.05, 0.05, 0.03)), (cx, cy, CASTER_H - 0.023), rgba=dark, name="dark", collide=False)   # swivel
            for side in (-1, 1):                                              # fork cheeks
                visual(base, ("box", (0.006, 0.07, 0.085)), (cx + side * 0.024, cy - 0.01, 0.0675), rgba=dark, name="dark", collide=False)
            visual(base, ("cylinder", (0.05, 0.035)), (cx, cy - 0.025, 0.05), "0 1.570796327 0", black, "black")   # wheel D100
    inertial(base, 15.8, (0, 0, 0.15), 0.8)
    col = ET.SubElement(robot, "link", name="column")
    visual(col, ("box", (COL, COL, COL_H)), (0, 0, COL_H / 2), rgba=alu, name="alu")
    for k in range(4):                                                        # T-slot grooves (visual only)
        ang = k * np.pi / 2; n = np.array([np.cos(ang), np.sin(ang)])
        for off in (-0.025, 0.025):
            t = np.array([-n[1], n[0]]) * off; p = n * (COL / 2 + 0.0005) + t
            visual(col, ("box", (0.008 if abs(n[0]) < 0.5 else 0.002, 0.008 if abs(n[0]) > 0.5 else 0.002, COL_H - 0.01)),
                   (p[0], p[1], COL_H / 2), rgba=dark, name="dark", collide=False)
    visual(col, ("box", (COL + 0.002, COL + 0.002, 0.006)), (0, 0, COL_H + 0.003), rgba=dark, name="dark")   # end cap
    inertial(col, 12.0, (0, 0, COL_H / 2), 1.5)
    fixed(robot, "column_joint", "stand_base", "column", (0, 0, Z_COL))
    for side, sx in (("right", -1), ("left", 1)):
        mp = ET.SubElement(robot, "link", name=f"{side}_mount_plate")
        visual(mp, ("box", (MOUNT_T, 0.12, 0.12)), (sx * MOUNT_T / 2, 0, 0), rgba=dark, name="dark")
        inertial(mp, 0.3, (sx * MOUNT_T / 2, 0, 0), 1e-4)
        fixed(robot, f"{side}_mount_plate_joint", "column", f"{side}_mount_plate", (sx * COL / 2, 0, SHOULDER_Z - Z_COL))

    if not YUP:   # ---- arms: SDK L-URDF on the robot's right, SDK R-URDF (= our single-arm sim, Dex1 included) on the robot's left ----
        sides = (("right", L_URDF, RIGHT_MOUNT, -1, "SDK R1_Robotic_Arm_L_V1.0-1027 + Dex1-1 (see SIDE in build_column_urdf.py)"),
                 ("left", R_URDF, LEFT_MOUNT, 1, "SDK R1_Robotic_Arm V1.1 (r1_7a_dex1.urdf, J6 +-0.524) + Dex1-1"))
    else:         # ---- --yup: SDK R-URDF on the robot's right, SDK L-URDF on the robot's left (vendor naming = physical side) ----
        sides = (("right", R_URDF, RIGHT_MOUNT, -1, "SDK R1_Robotic_Arm V1.1 (r1_7a_dex1.urdf, J6 +-0.524) + Dex1-1, flange base y up"),
                 ("left", L_URDF, LEFT_MOUNT, 1, "SDK R1_Robotic_Arm_L_V1.0-1027 + Dex1-1, flange base y up"))
    for side, src, MNT, sx, note in sides:
        robot.append(ET.Comment(f" {side}_* : robot's {side} side ({'-x' if sx < 0 else '+x'}), {note} "))
        if src == L_URDF:
            for e in arm_elems(L_URDF, f"{side}_", "sdkL"): robot.append(e)
            for e in arm_elems(R_URDF, f"{side}_", "dex", keep=lambda n: n.startswith("dex_") or n in ("gripper_mount", "tcp_joint", "tcp")):
                # 11-20 (user: one wrist camera up, one down): the L arm's Link7 frame is the R arm's mirrored AND turned 180 deg
                # about the gripper axis (Link7 y), so the R arm's mount transform Rx(pi) put the Dex1 (and its camera) upside
                # down on the L arm.  Turn it back: Ry(pi) Rx(pi) = Rz(pi); the origin lies on the axis; the tcp frame stays.
                if e.tag == "joint" and e.get("name") == f"{side}_gripper_mount": e.find("origin").set("rpy", "0 0 3.14159265")
                robot.append(e)
        else:
            for e in arm_elems(R_URDF, f"{side}_", "sdkR"):
                if e.tag == "link" and e.get("name").startswith(f"{side}_dex_"):
                    for m in e.iter("mesh"): m.set("filename", m.get("filename").replace("/sdkR/", "/dex/"))
                robot.append(e)
        fixed(robot, f"{side}_arm_mount", f"{side}_mount_plate", f"{side}_base_link", (sx * MOUNT_T, 0, 0), rpy(MNT))

    ET.indent(robot, "  ")
    out = os.path.join(OUT, "urdf", ("r1_column_dual_yup" if YUP else "r1_column_dual") + (f"_s{int(round(SHOULDER_Z * 100))}" if SH_ARG else "") + ".urdf"); ET.ElementTree(robot).write(out, encoding="unicode")
    print(out, "shoulder z", SHOULDER_Z, "right rpy", rpy(RIGHT_MOUNT), "left rpy", rpy(LEFT_MOUNT))


if __name__ == "__main__":
    main()
