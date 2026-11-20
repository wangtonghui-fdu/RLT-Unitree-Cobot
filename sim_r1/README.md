# 宇树 R1-7a 立柱双臂仿真示教

SAPIEN（光线追踪渲染 + PhysX，1200 Hz）里的双臂"交接、插充电头"示教生成。

| 文件 | 作用 |
|---|---|
| `build_column_urdf.py` | 由单臂 URDF 生成立柱双臂模型（左右臂镜像安装、肩高可调） |
| `r1_ik.py` | 逆运动学：阻尼最小二乘、限位回避、沿笛卡尔直线逐点跟踪 |
| `arm_collision.py` | 左右臂之间的最小间隙（用 PhysX 的碰撞外壳） |
| `probe_dual6.py`、`mk_layout_v6.py` | 搜索可行的摆放和交接位姿，生成 `layouts/layout_dual_v6.json` |
| `gen_dual_v6.py` | 示教生成器（规范动作、物理插入、三路相机、HDF5 输出） |
| `mk_v6f.py` → `gen_dual_v6f.py` | 快速版：两臂同时动、提高速度上限、随机范围可配，每条约 14 秒 |
| `dry6f.sh`、`an_wide.py` | 只跑物理不渲染，统计哪些随机摆位做得成 |
| `gen_worker_v6f.sh` | 按通过的种子清单批量渲染 |
| `to_lerobot_dual.py` | HDF5 转 LeRobot v3.0 |
| `sheet_wide.py` | 把多条示教的关键时刻拼成一张图用于人工检查 |

动作规范：左手竖直夹起平放的充电头 → 桌面中间交接（右手上指伸进左手两指之间、下指托底）→ 左手扶住排插并用腕部相机对准插孔 → 右手竖直缓慢插入。做不成的随机摆位直接剔除，不改动作去凑。

数据约定：16 维 = 左臂 7 关节 + 左夹爪开度 + 右臂 7 关节 + 右夹爪开度；`state[t]` 是实测值，`action[t]` 是下一帧的指令。

需要自备：宇树 R1-7a 与 Dex1-1 的 URDF 和网格；RoboDojo plug_in_charger 的排插、充电头模型。`camera_calib.json` 是我们的腕部相机标定。
