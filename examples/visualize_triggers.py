import sys
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

sys.path.append(str(Path(__file__).parent.parent / "src"))

import trigger

def main():
    input_dir = Path("examples/images")
    output_dir = Path("examples/output")
    output_dir.mkdir(exist_ok=True, parents=True)

    image_paths = list(input_dir.glob("*.JPEG"))

    trigger_options = {
        "Clean": lambda x: x,
        "White Box": trigger.white_box_trigger,
        "Gaussian Noise": trigger.gaussian_noise_trigger
    }

    for img_path in image_paths:
        print(f"Processing {img_path.name}...")
        img = Image.open(img_path).convert("RGB")
        
        results = []
        for _, trigger_fn in trigger_options.items():
            transform_pipeline = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                trigger_fn,
            ])
            img = transform_pipeline(img)
            results.append(img)

        widths, heights = zip(*(i.size for i in results))
        total_width = sum(widths)
        max_height = max(heights)

        combined = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for res_img in results:
            combined.paste(res_img, (x_offset, 0))
            x_offset += res_img.size[0]

        save_path = output_dir / f"compare_{img_path.stem}.png"
        combined.save(save_path)
        print(f"Saved visualization: {save_path}")

if __name__ == "__main__":
    main()
