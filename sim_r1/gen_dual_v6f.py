"""gen_dual_v6f.py (11-20): v6f = v6 with FAST timing (mk_v6f.py: higher speed caps, both arms move at the same time,
shorter phases; poses, grasps and order unchanged).
gen_dual_v6.py (11-20): v6 = the user's review of v5: arms lowered on the column (shoulders ~35 cm above the table: URDF
r1_column_dual_yup_s110, table 0.75 m; upper arm forward at shoulder height, forearm hanging), objects in the middle of the table;
the left hands the plug over flat with its TAIL TOWARDS THE RIGHT ARM (the right comes in from the right, upper finger between
the left's fingers, lower finger under the plug); the strip lies crosswise at centre-right-front; after the handover the LEFT
HOLDS THE STRIP (closed gripper on its top 5-7 cm left of the socket, tilted so its wrist camera sees the socket and the plug)
while the right pushes the plug in vertically.  Layout from probe_dual6.py / mk_layout_v6.py.
--data 1: dataset mode (3 cameras only, no overview; small preview.mp4).  Everything up to the end is recorded (return included).
v5 = v4's procedure with the scene fixed after the user's review: table flush against the column
(1.0 x 0.6 m, objects 15+ cm in from the near edge), no strip cable, left gripper turned the right way round (URDF) and both
wrist cameras mounted on <side>_dex_base_link (mirror images), head camera on a bracket 20 cm in front of the column top looking
down past the column (nothing of the stand / shoulders in view), ready posture with the hands low at the sides (off the objects
in the head view).  --snap N renders only every N-th frame as f_XXXX.jpg (quick review of all camera views).
v4 = the user's "standard" procedure (charger flat, prongs LEFT; left straight down, vertical descent,
grasp, vertical lift, to the middle; right comes in horizontally from the charger's back end, upper finger between the left's
fingers on the top face, lower finger under it; left lets go, rises, goes home; right turns it prongs-down with the gripper
vertical right over the socket and pushes it in slowly).  Table 0.82 m; layout from probe_dual4.py / mk_dual_layout4.py.
Older text (gen_dual_col.py): the user's Cobot Magic insert-charge procedure (datasets_ms/agilex-aloha-insert-charge-100demos,
"将充电头插入插排中") on the R1-7a column dual-arm model (r1_column_dual_yup.urdf), both arms with Dex1-1:
  left arm picks the charger lying flat on the table (its left side), pinching the narrow x-faces near the prong end;
  lifts it and turns it prongs-down to a handover point in front of the chest; the right arm pinches the broad faces at the
  back end; the left lets go, backs off, closes its gripper and holds the power strip down beside the target socket; the right
  carries the charger over the middle socket and plugs it in (prongs down), lets go; both arms return to the ready posture.
Layout from probe_dual.py (LAYOUT json).  Every arm move is a teleop-like Cartesian move (r1_ik.track_cart) unless it cannot
be tracked (then joint-space min-jerk into the anchored posture, logged).  Physics 1200 Hz / control 120 Hz / frames 30 Hz.
Charger physical (friction grasps, handover, insertion into RoboDojo's strip, static); arm-arm collisions are NOT simulated
(one articulation, self-collision off) - probe_dual checks a kinematic arm gap instead.
Cameras like the Cobot data: camera_top (on the column top, looking down at the table), camera_wrist_left, camera_wrist_right
(calibrated wrist camera, distorted), plus an overview for the showcase video.
Writes OUT/episode.hdf5 (state/action 16 = left J1..7, left gripper open 0..1, right J1..7, right gripper), OUT/demo.mp4,
OUT/meta.json, OUT/log.json.
usage: gen_dual_col.py URDF CALIB ASSETS LAYOUT OUT --seed N [--nominal 1] [--render 1] [--spp 16] [--ffmpeg PATH] [--preview 1]"""
import argparse, io, json, os, sys, time, subprocess, itertools, numpy as np, sapien, h5py
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r1_ik import R1IK, _log
from arm_collision import ArmCollision

ap = argparse.ArgumentParser()
for k in ("urdf", "calib", "assets", "layout", "out"):
    ap.add_argument(k)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--nominal", type=int, default=1)
ap.add_argument("--render", type=int, default=1)
ap.add_argument("--preview", type=int, default=0)
ap.add_argument("--snap", type=int, default=0)
ap.add_argument("--data", type=int, default=0)
ap.add_argument("--spp", type=int, default=16)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--dt", type=float, default=1 / 1200)
ap.add_argument("--ctrl_hz", type=int, default=120)
ap.add_argument("--ffmpeg", default="")
ap.add_argument("--head_pos", default="")
ap.add_argument("--head_target", default="")
ap.add_argument("--overview", default="")
ap.add_argument("--vcap", type=float, default=3.5)       # peak joint speed of a Cartesian move (rad/s)
ap.add_argument("--tcap", type=float, default=0.70)      # peak TCP speed (m/s)
ap.add_argument("--vjoint", type=float, default=2.4)     # peak joint speed of a joint-space move (rad/s)
ap.add_argument("--t_insert", type=float, default=1.8)   # the slow final push (s)
ap.add_argument("--rand_strip", default="-0.015,0.015,-0.01,0.01,-5,5")     # strip offset from the layout: x0,x1,y0,y1 (m), yaw0,yaw1 (deg)
ap.add_argument("--rand_charger", default="-0.015,0.015,-0.01,0.01,-5,5")   # charger offset from the layout, same format
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
L = json.load(open(a.layout)); TABLE_Z = L.get("table_z", 0.90)
rng = np.random.RandomState(a.seed); J_ = (lambda lo, hi: 0.0) if a.nominal else (lambda lo, hi: rng.uniform(lo, hi))
RZ = lambda t: np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
QZ = lambda t: [np.cos(t / 2), 0, 0, np.sin(t / 2)]
RS_, RC_ = ([float(v) for v in x.split(",")] for x in (a.rand_strip, a.rand_charger))
RAND = dict(strip_xy=[L["strip"][0] + J_(RS_[0], RS_[1]), L["strip"][1] + J_(RS_[2], RS_[3])], strip_yaw_deg=L["strip"][2] + J_(RS_[4], RS_[5]),
            charger_xy=[L["charger"][0] + J_(RC_[0], RC_[1]), L["charger"][1] + J_(RC_[2], RC_[3])], charger_yaw_deg=L["charger"][2] + J_(RC_[4], RC_[5]),
            light=1.0 + J_(-0.25, 0.25), ambient=1.0 + J_(-0.2, 0.2), light_dir=[0.3 + J_(-0.3, 0.3), 0.4 + J_(-0.3, 0.3), -1])
SYAW, CYAW = np.radians(RAND["strip_yaw_deg"]), np.radians(RAND["charger_yaw_deg"])
print("RAND", json.dumps({k_: RAND[k_] for k_ in ("strip_xy", "strip_yaw_deg", "charger_xy", "charger_yaw_deg")}), flush=True)
Q_HOME = np.array(L["q_home"]); INSTRUCTION = "将充电头插入插排中"
if not a.head_target: a.head_target = f"0.0,-0.38,{TABLE_Z}"
if not a.head_pos: a.head_pos = f"0.0,-0.20,{TABLE_Z + 0.84}"
if not a.overview: a.overview = f"-0.60,-1.55,{TABLE_Z + 0.80};0.0,-0.25,{TABLE_Z + 0.16}"

px = sapien.physx
c = px.get_shape_config(); c.contact_offset, c.rest_offset = 0.002, 0.0; px.set_shape_config(c)
c = px.get_body_config(); c.solver_position_iterations, c.solver_velocity_iterations = 40, 4; px.set_body_config(c)
c = px.get_scene_config(); c.enable_tgs, c.enable_pcm = True, True; px.set_scene_config(c)
if a.render:
    sapien.render.set_camera_shader_dir("rt"); sapien.render.set_ray_tracing_samples_per_pixel(a.spp)
    sapien.render.set_ray_tracing_path_depth(8); sapien.render.set_ray_tracing_denoiser("oidn")
C = json.load(open(a.calib))
_hp, _ht = np.array([float(v) for v in a.head_pos.split(",")]), np.array([float(v) for v in a.head_target.split(",")])
_z = (_ht - _hp) / np.linalg.norm(_ht - _hp); _x = np.cross(_z, [0, 0, 1.0]); _x /= np.linalg.norm(_x); _y = np.cross(_z, _x)
HEAD_T = np.eye(4); HEAD_T[:3, 0], HEAD_T[:3, 1], HEAD_T[:3, 2], HEAD_T[:3, 3] = _x, _y, _z, _hp
scene = sapien.Scene(); scene.set_timestep(a.dt)
LS = RAND["light"] * 0.8; scene.set_ambient_light([0.22 * RAND["ambient"]] * 3)
scene.add_directional_light(RAND["light_dir"], [1.5 * LS, 1.5 * LS, 1.45 * LS], shadow=True, shadow_map_size=4096)
scene.add_point_light([0.0, -0.55, 2.35], [2.4 * LS] * 3, shadow=True); scene.add_point_light([-1.0, -1.3, 2.0], [1.0 * LS] * 3)


def material(rgb, rough=0.5, metal=0.0, spec=0.5):
    m = sapien.render.RenderMaterial(); m.base_color = [*rgb, 1.0]; m.roughness = rough; m.metallic = metal; m.specular = spec
    return m


TABLE_M, WHITE = material([0.55, 0.40, 0.26], 0.8), material([0.90, 0.90, 0.88], 0.5)   # wood top (user 11-20: set the white objects apart)
GLOSS_WHITE, DARK, PRONG = material([0.875] * 3, 0.22, 0.1), material([0.216] * 3, 0.3), material([0.80] * 3, 0.3, 1.0)
FLOOR, WALL, LEG = material([0.62, 0.62, 0.60], 0.9), material([0.62, 0.64, 0.67], 0.95), material([0.25, 0.25, 0.27], 0.5, 0.6)
CABLE = material([0.92, 0.92, 0.90], 0.6)
PM_PLASTIC = scene.create_physical_material(0.5, 0.4, 0.0); PM_PAD = scene.create_physical_material(1.2, 1.0, 0.0)
b = scene.create_actor_builder(); b.add_box_visual(half_size=[4, 4, 0.005], material=FLOOR); b.add_box_collision(half_size=[4, 4, 0.005])
b.build_static(name="floor").set_pose(sapien.Pose([0, 0, -0.005]))
b = scene.create_actor_builder(); b.add_box_visual(half_size=[4, 0.01, 1.6], material=WALL); b.build_static(name="wall").set_pose(sapien.Pose([0, 1.8, 1.6]))
TABLE_C, TABLE_H = np.array([0.0, -0.37]), np.array([0.50, 0.30])  # v5: 1.0 x 0.6 m, near edge 2 cm from the column's front face
b = scene.create_actor_builder()
b.add_box_visual(sapien.Pose([0, 0, -0.012]), half_size=[*TABLE_H, 0.012], material=TABLE_M); b.add_box_collision(sapien.Pose([0, 0, -0.012]), half_size=[*TABLE_H, 0.012])
for sx_ in (-1, 1):
    for sy_ in (-1, 1):
        b.add_box_visual(sapien.Pose([sx_ * (TABLE_H[0] - 0.04), sy_ * (TABLE_H[1] - 0.04), -(TABLE_Z + 0.024) / 2]), half_size=[0.02, 0.02, (TABLE_Z - 0.024) / 2], material=LEG)
b.build_static(name="table").set_pose(sapien.Pose([*TABLE_C, TABLE_Z]))
b = scene.create_actor_builder()
_bc = _hp - _z * 0.016; _zb = _bc[2] + 0.035                         # v5 bracket: post up from the column top, bar forward ABOVE the camera
_post_h = (_zb - 1.43) / 2                                           # (everything behind / above the lens: never in view)
b.add_box_visual(half_size=[0.012, 0.012, _post_h], material=LEG); b.build_static(name="head_post").set_pose(sapien.Pose([0, 0, 1.43 + _post_h]))
b = scene.create_actor_builder(); b.add_box_visual(half_size=[0.012, abs(_bc[1]) / 2 + 0.012, 0.008], material=LEG)
b.build_static(name="head_bar").set_pose(sapien.Pose([0, _bc[1] / 2, _zb]))
b = scene.create_actor_builder(); b.add_box_visual(half_size=[0.010, 0.010, 0.012], material=LEG); b.build_static(name="head_drop").set_pose(sapien.Pose([_bc[0], _bc[1], _bc[2] + 0.02]))
_Mb = np.eye(4); _Mb[:3, 0], _Mb[:3, 1], _Mb[:3, 2], _Mb[:3, 3] = _x, _z, -_y, _bc
b = scene.create_actor_builder(); b.add_box_visual(half_size=[0.045, 0.0125, 0.0125], material=DARK); b.build_static(name="head_cam_body").set_pose(sapien.Pose(_Mb))

AS = a.assets
S_BODY, S_PLATE, S_BTN = (f"{AS}/socket/{n}.obj" for n in ("P_3b880443a15aabc4", "P_89e168adb9a23b3c", "P_80c68b0c268ffcdd"))
C_BODY = f"{AS}/charger/P_68f3f7542f1797e0.obj"
C_PRONGS = [f"{AS}/charger/{n}.obj" for n in ("P_9dbe0132b89ad8e0", "P_204036384b927f57")]
C_MARKS = [f"{AS}/charger/{n}.obj" for n in ("P_8a0ac7b7ff6ab698", "P_371b217d37787e98")]
sb = scene.create_actor_builder()
sb.add_visual_from_file(S_BODY, material=GLOSS_WHITE); sb.add_nonconvex_collision_from_file(S_BODY, material=PM_PLASTIC)
sb.add_visual_from_file(S_PLATE, material=DARK)
sb.add_visual_from_file(S_BTN, material=GLOSS_WHITE); sb.add_convex_collision_from_file(S_BTN, material=PM_PLASTIC)
STRIP_TOP = 0.0201
# the strip's cable (visual only, like the Cobot set): a few segments leaving the far end and curling away
strip = sb.build_static(name="power_strip")
STRIP_POSE = sapien.Pose([*RAND["strip_xy"], TABLE_Z - 0.0008], QZ(SYAW)); strip.set_pose(STRIP_POSE)
SLOT_CENTRE = (0.0070, 0.0064)
TARGET = STRIP_POSE * sapien.Pose([SLOT_CENTRE[0], SLOT_CENTRE[1], STRIP_TOP])
R_REST = np.stack([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], 1).astype(float)   # flat on a broad face, prongs +x (left), x_ch = -y
_Mc = np.eye(4); _Mc[:3, :3] = RZ(CYAW) @ R_REST; _Mc[:3, 3] = [*RAND["charger_xy"], TABLE_Z + 0.0135]; CH_TABLE = sapien.Pose(_Mc)
pb = scene.create_actor_builder(); pb.set_physx_body_type("dynamic")
PATCH = dict(patch_radius=0.012, min_patch_radius=0.006)
pb.add_visual_from_file(C_BODY, material=GLOSS_WHITE); pb.add_convex_collision_from_file(C_BODY, material=PM_PLASTIC, **PATCH)
for f in C_PRONGS: pb.add_visual_from_file(f, material=PRONG); pb.add_convex_collision_from_file(f, material=PM_PLASTIC)
for f in C_MARKS: pb.add_visual_from_file(f)
pb.set_mass_and_inertia(0.03, sapien.Pose([0, 0.024, 0]), [7.6e-6, 6.2e-6, 1.0e-5])
charger = pb.build(name="charger")

loader = scene.create_urdf_loader(); loader.fix_root_link = True
robot = loader.load(a.urdf)
names = [j.name for j in robot.get_active_joints()]; joints = robot.get_active_joints(); links = {l.name: l for l in robot.get_links()}
for l in robot.get_links():
    l.disable_gravity = True
    for sh in l.get_collision_shapes():
        g = sh.get_collision_groups(); g[2] = (1 << 29) if l.name.startswith("left_") else (1 << 28) if l.name.startswith("right_") else (1 << 29) | (1 << 28)
        sh.set_collision_groups(g)                                    # left-vs-right pairs collide (and are monitored), nothing else in the robot
    if "dex_Link" in l.name and l.name[-2:] in ("_2", "_3"):
        for s_ in l.get_collision_shapes(): s_.set_physical_material(PM_PAD)
    if not ("Link" in l.name or "base_link" in l.name) or l.name == "stand_base": continue
    for comp in l.entity.components:
        if isinstance(comp, sapien.render.RenderBodyComponent):
            for sh in comp.render_shapes:
                for part in (sh.parts if hasattr(sh, "parts") else [sh]):
                    m = part.material
                    if "dex_" in l.name: m.base_color, m.metallic, m.roughness = [0.04, 0.04, 0.045, 1], 0.0, 0.45
                    elif l.name.endswith("base_link") or l.name.endswith("Link1"): m.base_color, m.metallic, m.roughness = [0.08, 0.08, 0.09, 1], 0.3, 0.4
                    else: m.base_color, m.metallic, m.roughness = [0.66, 0.67, 0.69, 1], 0.55, 0.35
SIDES = ("left", "right")
AIDX = {s: np.array([names.index(f"{s}_J{j}") for j in range(1, 8)]) for s in SIDES}
GIDX = {s: np.array([i for i, n in enumerate(names) if n.startswith(f"{s}_dex_")]) for s in SIDES}
for j, n in zip(joints, names):
    if "_J" in n: j.set_drive_properties(4e4, 4e2, 300.0)
    else: j.set_drive_properties(3e3, 60.0, 15.0)
K = {s: R1IK(robot, prefix=f"{s}_") for s in SIDES}
TCP = {s: links[f"{s}_tcp"] for s in SIDES}; L7 = {s: links[f"{s}_Link7"] for s in SIDES}
LIM = robot.get_qlimits()
Q_WIDE, Q_SQUEEZE_L, Q_SQUEEZE_R, Q_REL, Q_CLOSED = -0.010, 0.008, 0.016, -0.015, 0.0245   # Dex1 finger q (gap 63 / 27 / 11 / 73 / 0 mm)
Q_APP_R = 0.0015                                                     # right comes in only 40 mm open (wider hits the left palm, probe_dual3)
R_CH0 = np.stack([[1, 0, 0], [0, 0, 1], [0, -1, 0]], 1).astype(float)


def rexp(w):
    t = np.linalg.norm(w)
    if t < 1e-12: return np.eye(3)
    k = w / t; Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(t) * Kx + (1 - np.cos(t)) * Kx @ Kx


def T_of(p, R): M = np.eye(4); M[:3, :3] = R; M[:3, 3] = p; return M


# ---- grasps / key poses (probe_dual6.py conventions) ----
_T_l7_tcp = np.eye(4); _T_l7_tcp[:3, 3] = [0, -0.1677, 0]; _T_l7_db = np.eye(4); _T_l7_db[:3, :3] = np.diag([1.0, -1, -1]); _T_l7_db[:3, 3] = [0, -0.0477, 0]
T_DB_CAM0 = np.linalg.inv(_T_l7_db) @ _T_l7_tcp @ np.array(C["wrist"]["T_tcp_cam"]); K_WRIST = np.array(C["wrist"]["K"], float); L_TIP = 0.030
_LN = [l_.name for l_ in robot.get_links()]; I_DB_L = _LN.index("left_dex_base_link")
R_INS = np.stack([[1, 0, 0], [0, 0, 1], [0, -1, 0]], 1).astype(float)     # prongs down, x_ch along the strip
Y_LGRIP, Z_LGRIP, Y_RGRIP = L["y_lgrip"], L["z_lgrip"], L["y_rgrip"]
_xl = np.array([L["fs_left"], 0, 0.0]); _yl = np.array([0, 0, -1.0])
G_L = T_of([0, Y_LGRIP, Z_LGRIP], np.stack([_xl, _yl, np.cross(_xl, _yl)], 1))   # left: straight down, fingers across the 42 mm width
_xr = np.array([0, 0, float(L["fz_right"])]); _yr = np.array([0, 1.0, 0])
G_R = T_of([0, Y_RGRIP, 0], np.stack([_xr, _yr, np.cross(_xr, _yr)], 1))       # right: along the body towards the prongs, fingers on top/bottom
_TG = CH_TABLE.to_transformation_matrix() @ G_L; P_GRASP, R_PICK = _TG[:3, 3], _TG[:3, :3]; P_PRE = P_GRASP + np.array([0, 0, 0.10])
T_CH_H = np.array(L["T_ch_H"], float)
R_CH_INS = RZ(SYAW + L["twin"] * np.pi) @ R_INS
H_HI, H_PRE, H_MID, H_END = L.get("h_hi", 0.09), 0.045, 0.020, -0.0005
CUR = dict(T_tcpL_ch=np.linalg.inv(G_L), T_tcpR_ch=np.linalg.inv(G_R))
Q_APP_R = 0.0015                                                     # right comes in 40 mm open (charger 27 mm thick)


def fk(s, q): return K[s].fk(q)


def line_poses(p0, R0, p1, R1):
    w = _log(R0.T @ R1); L_ = np.linalg.norm(p1 - p0); n = int(max(30, L_ / 0.0015, np.linalg.norm(w) / np.radians(0.75)))
    return [(p0 + (p1 - p0) * s, R0 @ rexp(s * w)) for s in np.linspace(0, 1, n + 1)[1:]], L_


_TAU = np.linspace(0, 1, 2001); _S = 10 * _TAU ** 3 - 15 * _TAU ** 4 + 6 * _TAU ** 5


def mj_time(Q, idx, L_, vcap=a.vcap, tcap=a.tcap):
    s = np.linspace(0, 1, len(Q)); sm = (s[1:] + s[:-1]) / 2
    f = 30 * np.interp(sm, _S, _TAU) ** 2 * (1 - np.interp(sm, _S, _TAU)) ** 2
    dqds = np.abs(np.diff(Q[:, idx], axis=0)).max(1) * (len(Q) - 1)
    return max(float((dqds * f).max()) / vcap, 1.875 * L_ / tcap)


def cart_leg(s, qa, p1, R1, what="", tight=False, q_goal="auto", min_margin=0.025, gain=0.3):
    idx = AIDX[s]; p0, R0 = fk(s, qa); p1 = np.asarray(p1, float); R1 = np.asarray(R1, float)
    poses, L_ = line_poses(p0, R0, p1, R1)
    if isinstance(q_goal, str):
        Q1 = K[s].track_cart(qa, poses, margin=0.08)[0]
        qg, pe_, re_, m_ = K[s].solve_anchored(Q1[-1], p1, R1, margin=0.17)
        q_goal = qg if (pe_ < 3e-4 and re_ < 3e-3) else None
    Q, pe, re, _, dq = K[s].track_cart(qa, poses, margin=0.08, q_goal=q_goal, goal_gain=0.0 if q_goal is None else gain)
    pf, Rf = fk(s, Q[-1]); pe_f, re_f = np.linalg.norm(pf - p1), np.linalg.norm(_log(R1 @ Rf.T))
    tol = (5e-4, 5e-3) if tight else (3e-3, 2e-2)
    tol_f = (5e-4, 5e-3) if tight else (1e-3, 1e-2)                  # end pose: exact for the precise moves, ~0.5 deg otherwise
    if pe > tol[0] or re > tol[1] or dq > 0.06 or pe_f > tol_f[0] or re_f > tol_f[1]:
        raise SystemExit(f"{s} {what}: Cartesian move not trackable (pos {pe * 1000:.1f} mm, rot {re:.3f}, step {dq:.3f}, end {pe_f * 1000:.2f} mm / {re_f:.4f})")
    Q = np.r_[np.asarray(qa, float)[None], Q]
    M = np.minimum(Q[:, idx] - LIM[idx, 0], LIM[idx, 1] - Q[:, idx]).min(1); mg = float(M[len(M) // 5:].min() if M[0] < min_margin else M.min())
    if mg < min_margin: raise SystemExit(f"{s} {what}: within {mg:.3f} rad of a limit")
    return dict(Q=Q, T_min=mj_time(Q, idx, L_), margin=mg)


def joint_leg(s, qa, qb):
    idx = AIDX[s]; Q = np.array([qa + (qb - qa) * u for u in np.linspace(0, 1, 121)])
    return dict(Q=Q, T_min=1.875 * float(np.abs(qb[idx] - qa[idx]).max()) / a.vjoint, margin=0.1)


def leg_or_joint(s, qa, p1, R1, q_anchor, what):
    try: return cart_leg(s, qa, p1, R1, what=what)
    except SystemExit as e_:
        qs, pe, re, m = K[s].solve_anchored(q_anchor, np.asarray(p1, float), np.asarray(R1, float), margin=0.12)
        if pe > 3e-4 or re > 3e-3: raise SystemExit(f"{s} {what}: no IK ({e_})")
        print(f"   {e_} -> joint space (margin {m:.2f})", flush=True); return joint_leg(s, qa, qs)


def full(qL7, qR7, gL=Q_WIDE, gR=Q_WIDE):
    q = np.zeros(len(names)); q[AIDX["left"]] = qL7; q[AIDX["right"]] = qR7; q[GIDX["left"]] = gL; q[GIDX["right"]] = gR; return q


def charger_T(): return charger.pose.to_transformation_matrix()


def tcp_T(s): return TCP[s].entity.pose.to_transformation_matrix()


def ins_pose(h):
    M = T_of(TARGET.p + np.array([0, 0, h]), R_CH_INS) @ np.linalg.inv(CUR["T_tcpR_ch"]); return M[:3, 3], M[:3, :3]


# ---- the procedure: (name, nominal s, {arm: callable(q_now) -> leg | None}, {arm: gripper target}) ----
T_CARRY, T_HOVER, D_HOVER = 1.3, 1.2, 0.25                           # right carry / left to the strip (starts D_HOVER later)
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
PH_CN = dict(settle="准备", L_approach="左手移到充电头正上方", L_descend="左手竖直下降", L_grasp="左手夹住", L_lift="左手竖直抬起", L_to_middle="左手送到中间",
             R_approach="右手从右侧对准插头尾部", R_insert_fingers="右手插入（上指在左手两指间、下指托底）", R_close="右手夹紧", L_release="左手松开", L_rise="左手抬起",
             R_over_socket="右手翻转、移到插孔正上方", L_to_strip="左手伸向插排", L_hold="左手扶住插排", R_lower="右手下降到插孔上方", R_align="右手对准",
             R_descend="右手下降", R_insert="右手缓慢插入", seated="到底", R_release="右手松开", R_rise="右手抬起", L_leave="左手离开插排", home="双手回位", done="完成")
PH_CN.update(L_approach="左手移到充电头上方，右手同时就位", handover_ready="交接就位", R_over_socket="右手翻转移到插孔上方，左手伸向插排",
             L_hold="左手扶住插排，右手对准", R_rise="双手抬起")


def inhand_report(tag, T_now, T_nom):
    D = np.linalg.inv(T_nom) @ T_now; ang = np.degrees(np.linalg.norm(_log(D[:3, :3])))
    print(f"   in-hand {tag}: {np.round(D[:3, 3] * 1000, 1)} mm, {ang:.1f} deg from nominal", flush=True)


AC = ArmCollision(a.urdf); RS_HOLD = np.random.RandomState(12345)


def ac_clear(qL, qR, gL, gR):
    q = np.zeros(len(names)); q[AIDX["left"]] = qL[AIDX["left"]]; q[AIDX["right"]] = qR[AIDX["right"]]; q[GIDX["left"]] = gL; q[GIDX["right"]] = gR
    return AC.clearance(q)[0]


def left_cam_px(q, P):
    """Pixels (pinhole) and depth of world points in the LEFT wrist camera at joint vector q."""
    pin_ = K["left"].pin; pin_.compute_forward_kinematics(np.asarray(q, float)); Tc = pin_.get_link_pose(I_DB_L).to_transformation_matrix() @ T_DB_CAM0
    pc = (np.linalg.inv(Tc) @ np.c_[P, np.ones(len(P))].T).T[:, :3]
    return np.c_[K_WRIST[0, 0] * pc[:, 0] / pc[:, 2] + K_WRIST[0, 2], K_WRIST[1, 1] * pc[:, 1] / pc[:, 2] + K_WRIST[1, 2], pc[:, 2]]


def choose_hold(q, carry):
    """Left 'hold the strip' pose on this (randomised) strip (probe_dual6 B2: nominal parameters first): closed finger tips on the
    strip top dpress left of the socket, gripper tilted tau towards the socket (azimuth az), so that the left wrist camera sees
    the socket and the plug; Cartesian hover -> hold; >= 4 mm hull clearance to the right arm now and when the plug is seated."""
    S = TARGET.p.copy(); nom = tuple(L["hold"]["par"])
    cands = [nom] + sorted([c for c in itertools.product((0.050, 0.058, 0.066), (35, 45, 55), (0, 25, -25, 50, -50), (1, -1)) if c != nom],
                           key=lambda c: abs(c[0] - nom[0]) * 20 + abs(c[1] - nom[1]) / 20 + abs(c[2] - nom[2]) / 25 + (c[3] != nom[3]) * 3)
    q_end = q.copy(); q_end[AIDX["right"]] = carry["Q"][-1][AIDX["right"]]; TR_ = np.ceil(max(T_CARRY, carry["T_min"] + 0.05) * 100 - 1e-6) / 100; aR = leg_path(carry["Q"])
    pR, RR = ins_pose(H_END); qRi, pe_, re_, _ = K["right"].solve_anchored(q_end, pR, RR, margin=0.1); CN = dict(ik=0, view=0, path=0, clr=0)
    for k_, (dpress, tau, az, sg) in enumerate(cands):
        pc = S + RZ(SYAW) @ np.array([dpress, 0, 0]); h = RZ(SYAW + np.radians(az)) @ np.array([-1.0, 0, 0]); t_ = np.radians(tau)
        d = np.sin(t_) * h - np.cos(t_) * np.array([0, 0, 1.0]); fx = sg * np.cross([0, 0, 1.0], h); R = np.stack([fx, -d, np.cross(fx, -d)], 1)
        p = pc - d * L_TIP + np.array([0, 0, 0.004]); best_ = None
        for an in [full(L["hold"]["q"], Q_HOME), q, None, None]:
            if an is None:
                q0 = q.copy(); q0[AIDX["left"]] = RS_HOLD.uniform(LIM[AIDX["left"], 0], LIM[AIDX["left"], 1]); an, pe, re = K["left"].solve_from(q0, p, R, iters=200)
                if pe > 3e-4 or re > 3e-3: continue
            qP_, pe, re, m_ = K["left"].solve_anchored(an, p, R, margin=0.12)
            if pe <= 3e-4 and re <= 3e-3 and qP_[AIDX["left"]][1] < 1.6 and (best_ is None or m_ > best_[1]): best_ = (qP_, m_)
            if best_ and best_[1] > 0.10: break
        if best_ is None or best_[1] < 0.05: CN["ik"] += 1; continue
        qP, m = best_; px = left_cam_px(qP, np.array([S, S + [0, 0, 0.03], S + [0, 0, 0.06]]))
        if (px[:, 2] < 0.05).any() or (px[:, 2] > 0.40).any() or not (80 < px[0, 0] < 560 and 60 < px[0, 1] < 440 and 60 < px[1, 0] < 580 and 30 < px[1, 1] < 450):
            CN["view"] += 1; continue
        try: lg1 = cart_leg("left", q, p + np.array([0, 0, 0.05]), R, "to the strip", q_goal=qP)
        except SystemExit:
            qHv, pe, re, mh = K["left"].solve_anchored(qP, p + np.array([0, 0, 0.05]), R, margin=0.1)
            if pe > 3e-4 or re > 3e-3: CN["path"] += 1; continue
            lg1 = joint_leg("left", q, qHv)
        try: lg2 = cart_leg("left", lg1["Q"][-1], p, R, "hold", tight=True)
        except SystemExit: CN["path"] += 1; continue
        TL_ = max(T_HOVER, lg1["T_min"] + 0.05); aL = leg_path(lg1["Q"]); sync = []
        for tt in np.arange(0.0, max(TR_, D_HOVER + TL_) + 1e-9, 1 / 60):          # both arms on their own clocks, as they will be executed
            uL = minjerk(min(max((tt - D_HOVER) / TL_, 0.0), 1.0)); uR = minjerk(min(max(tt / TR_, 0.0), 1.0))
            sync.append(ac_clear(aL(uL), aR(uR), Q_REL + (Q_CLOSED - Q_REL) * uL, 0.008))
        c = min(sync + [ac_clear(x, q_end, Q_CLOSED, 0.008) for x in lg2["Q"][::4]] +
                [ac_clear(lg2["Q"][-1], qRi, Q_CLOSED, 0.008), ac_clear(lg2["Q"][-1], qRi, Q_CLOSED, Q_REL)])
        if c < 0.004: CN["clr"] += 1; continue
        print(f"   hold candidate {k_} {(dpress, tau, az, sg)} (0 = nominal): margin {m:.2f}, clearance {c * 1000:.1f} mm (moving together {min(sync) * 1000:.1f}), socket at px {px[0, :2].round(0)}", flush=True)
        CUR["hold_leg"] = lg2; return lg1
    raise SystemExit(f"choose_hold: no feasible hold pose {CN}")


def named_leg(s, key, q):
    if key == "to_handover":
        CUR["T_tcpL_ch"] = np.linalg.inv(tcp_T("left")) @ charger_T(); inhand_report("left after lift", CUR["T_tcpL_ch"], np.linalg.inv(G_L))
        TL = T_CH_H @ np.linalg.inv(CUR["T_tcpL_ch"])
        return cart_leg("left", q, TL[:3, 3], TL[:3, :3], "to the middle", q_goal=full(L["q_handL"], Q_HOME))
    if key == "right_pre":
        TR = T_CH_H @ G_R; dR = -TR[:3, 1]                               # the left brings the charger exactly to T_CH_H -> known in advance
        return leg_or_joint("right", q, TR[:3, 3] - 0.08 * dR, TR[:3, :3], full(Q_HOME, L["q_preR"]), "to the tail")
    if key == "right_in":
        TR = charger_T() @ G_R; return cart_leg("right", q, TR[:3, 3], TR[:3, :3], "fingers in", tight=True, q_goal=full(Q_HOME, L["q_handR"]))
    if key == "left_up":
        p, R = fk("left", q)
        for dv, mm_ in ((np.array(L["left_up"]), 0.025), (np.array([0, 0, 0.06]), 0.025), (np.array([0.03, 0, 0.06]), 0.025), (np.array(L["left_up"]), 0.004)):
            try: return cart_leg("left", q, p + dv, R, "rise", min_margin=mm_)
            except SystemExit as e_: print("  ", e_, flush=True)
        raise SystemExit("left: cannot rise")
    if key == "right_carry":
        if CUR.get("carry_leg") is None:
            CUR["T_tcpR_ch"] = np.linalg.inv(tcp_T("right")) @ charger_T(); inhand_report("right after handover", CUR["T_tcpR_ch"], np.linalg.inv(G_R))
            p, R = ins_pose(H_HI); CUR["carry_leg"] = leg_or_joint("right", q, p, R, full(Q_HOME, L["q_insR"]), "over the socket")
        return CUR["carry_leg"]
    if key == "left_hover": return choose_hold(q, named_leg("right", "right_carry", q))
    if key == "left_hold": return CUR["hold_leg"]
    if key == "right_correct":
        CUR["T_tcpR_ch"] = np.linalg.inv(tcp_T("right")) @ charger_T(); inhand_report("right before insertion", CUR["T_tcpR_ch"], np.linalg.inv(G_R))
        p, R = ins_pose(H_PRE); return cart_leg("right", q, p, R, "align", q_goal=q)
    if key == "up":
        p, R = fk(s, q)
        for dv in (np.array([0, 0, 0.06]), np.array([0, 0, 0.04]), 0.05 * R[:, 1], 0.04 * R[:, 1] + np.array([0, 0, 0.02])):
            try: return cart_leg(s, q, p + dv, R, "up")
            except SystemExit as e_: print("  ", e_, flush=True)
        qh_ = q.copy(); qh_[AIDX[s]] = q_start[AIDX[s]]; return joint_leg(s, q, qh_)
    if key == "home":
        try: return cart_leg(s, q, *fk(s, q_start), "home", q_goal=q_start)
        except SystemExit as e_: print("  ", e_, "-> joint space", flush=True); qh_ = q.copy(); qh_[AIDX[s]] = q_start[AIDX[s]]; return joint_leg(s, q, qh_)
    raise KeyError(key)


def leg_path(Q):
    n = len(Q) - 1
    def at(u):
        x = min(max(u, 0.0), 1.0) * n; k = min(int(x), n - 1); f = x - k; return Q[k] + (Q[k + 1] - Q[k]) * f
    return at


CUR.update(i=-1)
ARM = {s: dict(path=None, bg=False, t0=0.0, T=1.0, g0=Q_WIDE, g1=Q_WIDE, gt0=0.0, gT=1.0) for s in SIDES}       # each arm (and its gripper) has its own clock
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


q_start = full(Q_HOME, Q_HOME); CMD["q"] = q_start.copy()
robot.set_qpos(q_start); robot.set_qvel(np.zeros(len(names)))
for j, v in zip(joints, q_start): j.set_drive_target(float(v))
charger.set_pose(CH_TABLE)


# ---- cameras ----
def cv_to_sapien(T):
    T = np.array(T, float); M = np.eye(4); M[:3, :3] = np.stack([T[:3, 2], -T[:3, 0], -T[:3, 1]], 1); M[:3, 3] = T[:3, 3]
    return sapien.Pose(M)


def look(p, t):
    p, t = np.array(p, float), np.array(t, float); f = (t - p) / np.linalg.norm(t - p); l = np.cross([0, 0, 1.0], f); l /= np.linalg.norm(l); u = np.cross(f, l)
    M = np.eye(4); M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = f, l, u, p; return sapien.Pose(M)


cams = {}
if a.render:
    def add_cam(name, K_, size, T=None, mount=None):
        w, h = size
        cm = scene.add_camera(name, w, h, 1.0, 0.01, 10) if mount is None else scene.add_mounted_camera(name, mount.entity, sapien.Pose(), w, h, 1.0, 0.01, 10)
        cm.set_perspective_parameters(0.01, 10, K_[0][0], K_[1][1], K_[0][2], K_[1][2], 0.0)
        if mount is None: cm.entity.set_pose(cv_to_sapien(T))
        else: cm.local_pose = cv_to_sapien(T)
        return cm
    cams["top"] = add_cam("top", C["head"]["K"], C["head"]["size"], T=HEAD_T)
    T_l7_tcp = np.eye(4); T_l7_tcp[:3, 3] = [0, -0.1677, 0]
    T_l7_db = np.eye(4); T_l7_db[:3, :3] = np.diag([1.0, -1, -1]); T_l7_db[:3, 3] = [0, -0.0477, 0]     # the R arm's gripper mount (calibrated arm)
    T_DB_CAM = np.linalg.inv(T_l7_db) @ T_l7_tcp @ np.array(C["wrist"]["T_tcp_cam"])                  # wrist camera in the Dex1 base frame
    Kw, (ww, hw), PADX, PADY = np.array(C["wrist"]["K"], float), C["wrist"]["size"], 160, 120
    Kwide = Kw.copy(); Kwide[0, 2] += PADX; Kwide[1, 2] += PADY
    for s in SIDES:
        cams[f"wrist_{s}"] = add_cam(f"wrist_{s}", Kwide.tolist(), (ww + 2 * PADX, hw + 2 * PADY), T=T_DB_CAM, mount=links[f"{s}_dex_base_link"])
    k1, k2, p1_, p2_ = C["wrist"]["dist"][:4]; uu, vv = np.meshgrid(np.arange(ww) + 0.0, np.arange(hw) + 0.0)
    xd, yd = (uu - Kw[0, 2]) / Kw[0, 0], (vv - Kw[1, 2]) / Kw[1, 1]; x, y = xd.copy(), yd.copy()
    for _ in range(30):
        r2 = x * x + y * y; rad = 1 + k1 * r2 + k2 * r2 * r2
        x = (xd - (2 * p1_ * x * y + p2_ * (r2 + 2 * x * x))) / rad; y = (yd - (p1_ * (r2 + 2 * y * y) + 2 * p2_ * x * y)) / rad
    WMAP = (Kwide[0, 0] * x + Kwide[0, 2], Kwide[1, 1] * y + Kwide[1, 2])
    _op, _ot = [np.array([float(v) for v in s.split(",")]) for s in a.overview.split(";")]
    cams["overview"] = scene.add_camera("overview", 960, 720, np.radians(36), 0.02, 20); cams["overview"].entity.set_pose(look(_op, _ot))


def remap(im, mp):
    U, V = np.clip(mp[0], 0, im.shape[1] - 1.001), np.clip(mp[1], 0, im.shape[0] - 1.001); u0, v0 = U.astype(int), V.astype(int)
    du, dv = (U - u0)[..., None], (V - v0)[..., None]; f = im.astype(float)
    return (f[v0, u0] * (1 - du) * (1 - dv) + f[v0, u0 + 1] * du * (1 - dv) + f[v0 + 1, u0] * (1 - du) * dv + f[v0 + 1, u0 + 1] * du * dv).astype(np.uint8)


def grab(cm): cm.take_picture(); return (cm.get_picture("Color")[..., :3] * 255).clip(0, 255).astype(np.uint8)


def jpeg(img): b_ = io.BytesIO(); Image.fromarray(img).save(b_, format="JPEG", quality=92); return b_.getvalue()


def opening(qf): return float(np.clip((0.0245 - np.mean(qf)) / 0.0445, 0, 1))


FONT = next((ImageFont.truetype(f, 22) for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",) if os.path.exists(f)), None)
FONT_S = next((ImageFont.truetype(f, 15) for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",) if os.path.exists(f)), None)


def compose(ov, top, wl, wr, ph, t):
    """1920x720: overview 960x720 | [camera_top, info] over [wrist left, wrist right] (480x360 each)."""
    G = Image.new("RGB", (1920, 720), (20, 20, 22)); G.paste(Image.fromarray(ov), (0, 0))
    for im, (x0, y0), lab in ((top, (960, 0), "camera_top"), (wl, (960, 360), "camera_wrist_left"), (wr, (1440, 360), "camera_wrist_right")):
        G.paste(Image.fromarray(im).resize((480, 360), Image.BILINEAR), (x0, y0)); d = ImageDraw.Draw(G)
        d.rectangle([x0, y0, x0 + 170, y0 + 20], fill=(0, 0, 0)); d.text((x0 + 4, y0 + 1), lab, fill=(255, 255, 255), font=FONT_S)
    d = ImageDraw.Draw(G)
    d.text((1460, 30), "R1-7a 立柱双臂 · 交接插充电头", fill=(230, 230, 230), font=FONT); d.text((1460, 100), "左手夹起 → 交接 → 左手扶插排 → 右手插入", fill=(170, 170, 170), font=FONT_S); d.text((1460, 70), f"指令：{INSTRUCTION}", fill=(200, 200, 200), font=FONT_S)
    d.text((1460, 130), f"{t:5.1f} s", fill=(255, 230, 80), font=FONT); d.text((1460, 170), PH_CN.get(ph, ph), fill=(255, 230, 80), font=FONT)
    return np.asarray(G)


if a.preview:
    scene.update_render(); ims = [grab(cams["overview"]), grab(cams["top"]), remap(grab(cams["wrist_left"]), WMAP), remap(grab(cams["wrist_right"]), WMAP)]
    Image.fromarray(compose(*ims, "settle", 0.0)).save(f"{a.out}/preview.png"); print("preview saved"); sys.exit()

# ---- run ----
ctrl_every = max(1, int(round(1.0 / (a.ctrl_hz * a.dt)))); frame_steps = int(round(1.0 / (a.fps * a.dt)))
log, qcmd, t_render = [], q_start.copy(), 0.0
REC = dict(top=[], wrist_left=[], wrist_right=[], action=[], state=[], phase=[])
ff = None; wall = time.time(); ph = "settle"


def charger_state():
    M = charger_T(); Tl = np.linalg.inv(TARGET.to_transformation_matrix()) @ M
    return Tl[:3, 3], np.degrees(np.arccos(np.clip(M[2, 1], -1, 1)))


def vec16(q, gq):
    return np.r_[q[AIDX["left"]], opening(gq[GIDX["left"]]), q[AIDX["right"]], opening(gq[GIDX["right"]])]


SIDE_OF = {l.entity.name: ("L" if l.name.startswith("left_") else "R" if l.name.startswith("right_") else "S") for l in robot.get_links()}
N_CONTACT = [0, 9.0]                                                # frames with a left-right contact, smallest separation


def arm_contact():
    d = 9.0
    for c_ in scene.get_contacts():
        a_, b_ = (SIDE_OF.get(x.entity.name, "?") for x in c_.bodies)
        if {a_, b_} == {"L", "R"}: d = min([d] + [p.separation for p in c_.points])
    return d


N_CL = [0, 9.0]                                                     # frames with the charger touching the LEFT arm after the handover, smallest separation
CLEAR_PH = {}                                                       # smallest left-right hull clearance per phase (mm, 12 = none within 12 mm)


def charger_left_contact():
    d = 9.0
    for c_ in scene.get_contacts():
        n_ = [x.entity.name for x in c_.bodies]
        if "charger" in n_ and any(SIDE_OF.get(x_) == "L" for x_ in n_): d = min([d] + [p.separation for p in c_.points])
    return d


for k in range(10 ** 8):
    t = k * a.dt
    if t > sum(sg[1] for sg in SEGS) + 1e-9: break
    if k % ctrl_every == 0:
        ph, qcmd = plan(t); CMD["q"] = qcmd.copy()
        for j, v in zip(joints, qcmd): j.set_drive_target(float(v))
    scene.step()
    if k % frame_steps == 0:
        rel, tilt = charger_state(); qp = robot.get_qpos(); dc = arm_contact()
        if SEG_NAMES.index(ph) > SEG_NAMES.index("L_rise"):
            dl_ = charger_left_contact(); N_CL[1] = min(N_CL[1], dl_)
            if dl_ < 0.0005:
                N_CL[0] += 1; print(f"   CHARGER-LEFT CONTACT t={t:.2f} {ph} sep {dl_ * 1000:.1f} mm", flush=True) if N_CL[0] < 5 else None
        CLEAR_PH[ph] = min(CLEAR_PH.get(ph, 99.0), round(AC.clearance(qp)[0] * 1000, 1))
        if dc < 0.002: N_CONTACT[0] += 1; N_CONTACT[1] = min(N_CONTACT[1], dc); print(f"   ARM CONTACT t={t:.2f} {ph} sep {dc * 1000:.1f} mm", flush=True) if N_CONTACT[0] < 5 else None
        log.append(dict(t=round(t, 4), phase=ph, charger_mm=[round(v * 1000, 2) for v in rel], tilt_deg=round(tilt, 2), charger_z=round(float(charger.pose.p[2]), 4),
                        q_left=[round(float(v), 4) for v in qp[AIDX["left"]]], q_right=[round(float(v), 4) for v in qp[AIDX["right"]]],
                        g=[round(opening(qp[GIDX[s]]), 3) for s in SIDES]))
        REC["state"].append(vec16(qp, qp)); REC["action"].append(vec16(qcmd, qcmd)); REC["phase"].append(ph)
        if a.render and (not a.snap or (k // frame_steps) % a.snap == 0):
            t1 = time.time(); scene.update_render()
            top, wl, wr = grab(cams["top"]), remap(grab(cams["wrist_left"]), WMAP), remap(grab(cams["wrist_right"]), WMAP)
            if a.snap: Image.fromarray(compose(grab(cams["overview"]), top, wl, wr, ph, t)).save(f"{a.out}/f_{int(round(t * 30)):04d}.jpg", quality=88)
            else:
                for key, im in (("top", top), ("wrist_left", wl), ("wrist_right", wr)): REC[key].append(jpeg(im))
            if a.ffmpeg and not a.snap and a.data:                      # dataset mode: small 3-camera preview only
                fr = np.concatenate([top[::2, ::2], wl[::2, ::2], wr[::2, ::2]], 1)
                if ff is None:
                    ff = subprocess.Popen([a.ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "960x240", "-r", str(a.fps), "-i", "-",
                                           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26", f"{a.out}/preview.mp4"], stdin=subprocess.PIPE)
                ff.stdin.write(np.ascontiguousarray(fr).tobytes())
            elif a.ffmpeg and not a.snap:
                fr = compose(grab(cams["overview"]), top, wl, wr, ph, t)
                if ff is None:
                    ff = subprocess.Popen([a.ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1920x720", "-r", str(a.fps), "-i", "-",
                                           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", f"{a.out}/demo.mp4"], stdin=subprocess.PIPE)
                ff.stdin.write(fr.tobytes())
                if k % (frame_steps * 15) == 0: Image.fromarray(fr).save(f"{a.out}/f_{int(round(t * 30)):04d}.jpg", quality=85)
            t_render += time.time() - t1
if ff is not None: ff.stdin.close(); ff.wait()
rel, tilt = charger_state()
seated = abs(rel[2]) < 0.0015 and np.hypot(rel[0], rel[1]) < 0.001 and tilt < 3
clean = N_CONTACT[0] == 0 and N_CL[0] == 0
meta = dict(seed=a.seed, ok=bool(seated and clean), seated=bool(seated), charger_left_contact_frames=N_CL[0], charger_left_min_sep_mm=round(min(N_CL[1], 0.02) * 1000, 2),
            clear_by_phase_mm=CLEAR_PH, bg=BG_LOG, caps=dict(vcap=a.vcap, tcap=a.tcap, vjoint=a.vjoint, t_insert=a.t_insert), duration_s=round(sum(s_[1] for s_ in SEGS), 2), instruction=INSTRUCTION, fps=a.fps, frames=len(REC["state"]), rand=RAND, layout=L, version="dual_v6f", urdf=os.path.basename(a.urdf), arm_contact_frames=N_CONTACT[0], arm_min_sep_mm=round(N_CONTACT[1] * 1000, 2),
            final_charger_in_target_frame_mm=[round(v * 1000, 2) for v in rel], tilt_deg=round(tilt, 2), segs=[(s_[0], s_[1]) for s_ in SEGS],
            wall_s=round(time.time() - wall, 1), render_s=round(t_render, 1), spp=a.spp,
            state_names=[f"left_J{i}" for i in range(1, 8)] + ["left_gripper_open"] + [f"right_J{i}" for i in range(1, 8)] + ["right_gripper_open"],
            action_def="commanded joint targets (rad) + commanded gripper opening (0 closed..1 open), same order as the state")
if seated and clean and a.render and not a.snap:
    with h5py.File(f"{a.out}/episode.hdf5", "w") as f:      # same key names as the single-arm sets (train_smolvla_r1.py / to_lerobot)
        for key, cam in (("top", "head_camera"), ("wrist_left", "left_wrist_camera"), ("wrist_right", "right_wrist_camera")):
            L_ = max(len(x) for x in REC[key]); f.create_dataset(f"observation/{cam}/rgb", data=np.array(REC[key], dtype=f"S{L_}"))
        f.create_dataset("joint_action/vector", data=np.array(REC["action"], np.float32)); f.create_dataset("observation/joint_state", data=np.array(REC["state"], np.float32))
        f.create_dataset("phase", data=np.array(REC["phase"], dtype="S24")); f.attrs["instruction"] = INSTRUCTION; f.attrs["meta"] = json.dumps(meta, ensure_ascii=False)
json.dump(meta, open(f"{a.out}/meta.json", "w"), ensure_ascii=False); json.dump(log, open(f"{a.out}/log.json", "w"))
print("RESULT", json.dumps({k: meta[k] for k in ("seed", "ok", "seated", "duration_s", "charger_left_contact_frames", "charger_left_min_sep_mm", "frames", "final_charger_in_target_frame_mm", "tilt_deg", "arm_contact_frames", "arm_min_sep_mm", "wall_s", "render_s")}), flush=True)
