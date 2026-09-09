import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "model" / "RealESRGAN_x4plus.pth"
OUTPUT_PATH = PROJECT_DIR / "model" / "RealESRGAN_x4plus.onnx"


model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
checkpoint = torch.load(MODEL_PATH, map_location="cpu")

if "params_ema" in checkpoint:
    state_dict = checkpoint["params_ema"]
elif "params" in checkpoint:
    state_dict = checkpoint["params"]
else:
    state_dict = checkpoint

model.load_state_dict(state_dict, strict=True)
model.eval()

dummy_input = torch.randn(1, 3, 64, 64)

torch.onnx.export(
    model,
    dummy_input,
    OUTPUT_PATH,
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch", 2: "height", 3: "width"},
        "output": {0: "batch", 2: "height", 3: "width"}
    },
    do_constant_folding=True
)

print(f"ONNX model written to: {OUTPUT_PATH}")