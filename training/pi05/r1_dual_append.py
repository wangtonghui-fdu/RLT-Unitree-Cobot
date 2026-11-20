

# ---- dual-arm R1-7a column robot (11-20): camera_top -> base_0_rgb, both wrist cameras real; state / actions 16-d
# (left J1..J7 rad, left gripper opening 0..1, right J1..J7, right gripper)
@dataclasses.dataclass(frozen=True)
class R1DualInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        inputs = {
            "state": np.asarray(data["observation/state"]),
            "image": {"base_0_rgb": _parse_image(data["observation/image"]), "left_wrist_0_rgb": _parse_image(data["observation/left_wrist_image"]),
                      "right_wrist_0_rgb": _parse_image(data["observation/right_wrist_image"])},
            "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.True_, "right_wrist_0_rgb": np.True_},
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class R1DualOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :16])}
