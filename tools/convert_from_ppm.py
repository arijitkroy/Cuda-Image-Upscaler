import sys
from pathlib import Path
from PIL import Image

def convert_from_ppm(input_path: str, output_path: str) -> None:
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(input_file) as image:
        rgb_image = image.convert("RGB")
        extension = output_file.suffix.lower()

        rgb_image.save(output_file, format="PNG")
        print("Conversion completed")
        print(f"Input: {input_file}")
        print(f"Output: {output_file}")
        print(f"Resolution: {rgb_image.width} x {rgb_image.height}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:\npython convert_from_ppm.py <input.ppm> <output.png>")
        print("Example:\npython convert_from_ppm.py output/upscaled.ppm output/upscaled.png")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    try:
        convert_from_ppm(input_path, output_path)
    except Exception as error:
        print(f"Conversion failed: {error}")
        sys.exit(1)