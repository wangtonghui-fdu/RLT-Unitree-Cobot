# 三个基线的训练配置

同一份仿真数据（128 条、三路相机、16 维状态和动作）。

| | SmolVLA | π0.5 | LingBot-VLA v2 |
|---|---|---|---|
| 目录 | `smolvla/` | `pi05/` | `lingbot/` |
| 全局批量 | 40（2 卡 × 20） | 96（8 卡分片） | 256（4 卡 × 16 × 累积 4） |
| 学习率 | 1e-4，余弦到 2.5e-6 | 2.5e-5，余弦 | 5e-5，恒定 |
| 优化器 | AdamW | AdamW | Muon |
| 关节动作 | 绝对角度 | 相对当前状态的增量 | 相对当前状态的增量 |
| 图像增广 | 无 | 有（openpi 默认） | 无 |
| 步数 | 40000 | 20000 | 不设上限 |

- `smolvla/train_smolvla_r1.py`：直接读示教 HDF5 的 DDP 训练脚本，`smolvla_r1_dual_v6f_2gpu.json` 是配置。
- `pi05/`：往 openpi 里加双臂 R1 的数据配置和输入输出变换（`config_r1_dual_patch.py`、`r1_dual_append.py`）。
- `lingbot/`：机器人配置和训练 yaml；`build_frame_cache.py` / `verify_frame_cache.py` 把视频帧预解码成内存映射数组；`keep_r1.py` 轮换存档；`zs_eval.py` 开环评测；`robot_config_cobot_insert_charge.yaml` 是 Cobot Magic 真机数据的配置。

yaml 和 json 里的路径是占位符，需要改成自己的。
