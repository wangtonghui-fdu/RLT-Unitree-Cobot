# RL Token 复现：宇树 R1-7a 与 Cobot Magic 插充电头

在"把充电头插进排插"这个毫米级任务上复现 RL token（RLT，Xu et al. 2026，Physical Intelligence）的阶段性代码和记录。

**展示页**：<https://wangtonghui-fdu.github.io/RLT-Unitree-Cobot/>（有仿真示教视频和各项结果）

## 现在做到哪了

截至 2026-11-20。**阶段二（在线强化学习）还没有开始，目前不能说复现完成。**

| 内容 | 机器人 | 状态 |
|---|---|---|
| RL token 阶段一模块，与 VLA 微调并行训练 | Cobot Magic 真机数据 | 已完成 |
| LingBot-VLA v2 全量微调 | Cobot Magic 真机数据 | 已完成；真机交接时双臂相撞 |
| "RL token 里有没有毫米级几何"诊断 | Cobot Magic 真机数据 | 已完成 |
| 立柱双臂模型与规范动作示教（128 条） | 宇树 R1-7a 仿真 | 已完成 |
| SmolVLA、π0.5、LingBot-VLA 三个基线训练 | 宇树 R1-7a 仿真 | 进行中，尚无闭环评测 |
| LingBot-VLA 训练提速（8.2 → 6.2 秒/步） | — | 两项已用上，老师输出缓存在做 |
| RL token 阶段二 | 两者 | 未开始 |
| 宇树 R1-7a 真机 | 宇树 R1-7a | 未开始 |

## 目录

| 目录 | 内容 |
|---|---|
| `rl_token/` | RL token 阶段一模块、接入 LingBot-VLA 训练循环的补丁、随机数隔离的单元测试 |
| `representation_probe/` | 特征提取和探针脚本（表征诊断） |
| `sim_r1/` | 立柱双臂 URDF 生成、逆运动学、两臂碰撞检查、示教生成、布局搜索、转 LeRobot 格式 |
| `training/` | 三个模型的训练配置和数据接入代码 |
| `speedups/` | LingBot-VLA 的分步计时、两项提速补丁、老师输出缓存 |
| `tools/` | 多卡一致性自检、训练 NaN 看门脚本 |
| `docs/` | 展示页（GitHub Pages） |

每个目录里有单独的说明。

## 仓库里没有的东西

这些需要自己准备，脚本里对应的路径都是占位符（`/path/to/...`）：

- **宇树 R1-7a 和 Dex1-1 的 URDF 与网格**：从宇树官方获取。
- **排插和充电头模型**：来自 RoboDojo 基准 plug_in_charger 任务的素材。
- **模型代码和预训练权重**：LingBot-VLA v2、openpi（π0.5）、LeRobot（SmolVLA）用各自的官方仓库，这里只有针对它们的补丁和配置。
- **真机数据和我们训练出的权重**。

## 说明

- 脚本是研究过程中的工作代码，注释里保留了当时的日期和修改原因，没有整理成可安装的包。
- 多数脚本原本在特定机器上运行，换环境需要改路径和 Python 环境。
