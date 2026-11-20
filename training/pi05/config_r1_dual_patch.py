"""Append the dual-arm R1 data config + train config (pi05_r1_dual_v6) to openpi's training/config.py and the dual transforms to
policies/r1_policy.py (idempotent).  usage: config_r1_dual_patch.py OPENPI_DIR R1_DUAL_APPEND"""
import sys, pathlib
root = pathlib.Path(sys.argv[1]); pol = root / "src/openpi/policies/r1_policy.py"; p = root / "src/openpi/training/config.py"
ps = pol.read_text()
if "R1DualInputs" not in ps: pol.write_text(ps.rstrip("\n") + "\n" + pathlib.Path(sys.argv[2]).read_text()); print("policy patched")
s = p.read_text()
if "pi05_r1_dual_v6" in s:
    print("config already patched"); sys.exit()
anchor = "# Use `get_config` if you need to get a config by name in your code."
block = '''@dataclasses.dataclass(frozen=True)
class LeRobotR1DualDataConfig(DataConfigFactory):
    """Unitree R1-7a column dual-arm robot + Dex1-1 grippers, handover + plug task (11-20), LeRobot v3.0 from gen_dual_v6.py /
    to_lerobot_dual.py.  16-d state/action: joints become deltas w.r.t. the current state, both gripper openings stay absolute."""

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack = _transforms.Group(inputs=[_transforms.RepackTransform({
            "observation/image": "observation.images.camera_top",
            "observation/left_wrist_image": "observation.images.camera_wrist_left",
            "observation/right_wrist_image": "observation.images.camera_wrist_right",
            "observation/state": "observation.state",
            "actions": "action",
            "prompt": "prompt",
        })])
        data_transforms = _transforms.Group(inputs=[r1_policy.R1DualInputs(model_type=model_config.model_type)],
                                            outputs=[r1_policy.R1DualOutputs()])
        delta_mask = _transforms.make_bool_mask(7, -1, 7, -1)
        data_transforms = data_transforms.push(inputs=[_transforms.DeltaActions(delta_mask)],
                                               outputs=[_transforms.AbsoluteActions(delta_mask)])
        model_transforms = ModelTransformFactory()(model_config)
        return dataclasses.replace(self.create_base_config(assets_dirs, model_config), repack_transforms=repack,
                                   data_transforms=data_transforms, model_transforms=model_transforms,
                                   action_sequence_keys=("action",))


'''
assert s.count(anchor) == 1
s = s.replace(anchor, block + anchor, 1)
anchor2 = "_CONFIGS = ["
cfg = '''_CONFIGS = [
    # dual-arm column robot, handover + plug (11-20)
    TrainConfig(
        name="pi05_r1_dual_v6",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=50),
        data=LeRobotR1DualDataConfig(repo_id="r1_dual_v6", base_config=DataConfig(prompt_from_task=True)),
        weight_loader=weight_loaders.CheckpointWeightLoader("/root/c2r/models/pi05_base/params"),
        batch_size=64,
        fsdp_devices=8,
        num_workers=32,
        num_train_steps=30000,
        save_interval=2500,
        keep_period=5000,
        log_interval=50,
    ),
'''
assert s.count(anchor2) == 1
s = s.replace(anchor2, cfg, 1)
p.write_text(s); print("config patched")
