from pathlib import Path
import onnxruntime as ort

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "model" / "RealESRGAN_x4plus.onnx"

print(f"Model: {MODEL_PATH}")
print(f"ONNX Runtime: {ort.__version__}")
print("Available providers:")

for provider in ort.get_available_providers():
    print(f"  {provider}")

if "CUDAExecutionProvider" not in ort.get_available_providers():
    raise RuntimeError("CUDAExecutionProvider is not available")

session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CUDAExecutionProvider"]
)

print("\nModel loaded successfully.")
print("Execution providers:")
for provider in session.get_providers():
    print(f"  {provider}")

print("\nInputs:")
for input_info in session.get_inputs():
    print(f"  {input_info.name}: {input_info.shape}, {input_info.type}")

print("\nOutputs:")
for output_info in session.get_outputs():
    print(f"  {output_info.name}: {output_info.shape}, {output_info.type}")