import os
import json
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from pycocotools import mask as maskUtils
from rich import print
from rich.progress import (
    Progress,
    TimeElapsedColumn,
    TimeRemainingColumn,
    BarColumn,
    TextColumn,
)
from utils import load_model, get_inference_augmentation


def encode_mask(mask):
    rle = maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def visualize_random_segmentation(
    image_info, results, test_dir, save_dir, num_samples=2
):
    import random
    import matplotlib.pyplot as plt
    from pycocotools.mask import decode as decode_mask
    from collections import defaultdict
    import numpy as np
    from PIL import Image
    import os

    chosen_ids = random.sample([info["id"] for info in image_info], num_samples)

    image_to_results = defaultdict(list)
    for res in results:
        image_to_results[res["image_id"]].append(res)

    for image_id in chosen_ids:
        file_name = next(
            info["file_name"] for info in image_info if info["id"] == image_id
        )
        image_path = os.path.join(test_dir, file_name)
        image = np.array(Image.open(image_path).convert("RGB"))

        seg_image = image.copy()
        overlay = np.zeros_like(image, dtype=np.uint8)

        for res in image_to_results[image_id]:
            mask = decode_mask(res["segmentation"])
            color = np.random.randint(0, 255, size=3, dtype=np.uint8)
            for c in range(3):
                overlay[:, :, c] = np.where(mask == 1, color[c], overlay[:, :, c])

        blended = (0.5 * seg_image + 0.5 * overlay).astype(np.uint8)

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(image)
        ax[0].set_title("Original Image")
        ax[0].axis("off")

        ax[1].imshow(blended)
        ax[1].set_title("Instance Segmentation")
        ax[1].axis("off")

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"vis_image_{image_id}.png")
        plt.savefig(save_path)
        plt.close()


def main():
    args = get_inference_augmentation()
    os.makedirs(args.save_dir, exist_ok=True)
    device = args.device
    model = load_model(args.model_path, device, model_type="test")
    transform = transforms.ToTensor()

    with open(os.path.join(args.data_root, "test_image_name_to_ids.json"), "r") as f:
        image_info = json.load(f)

    test_dir = os.path.join(args.data_root, "test_release")
    results = []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.1f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("[green]Predicting...", total=len(image_info))
        for info in image_info:
            file_name = info["file_name"]
            image_id = info["id"]
            image_path = os.path.join(test_dir, file_name)
            image = Image.open(image_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device=device)

            with torch.no_grad():
                outputs = model(image_tensor)[0]

            for i in range(len(outputs["scores"])):

                if device == "cpu":
                    mask = outputs["masks"][i, 0].numpy() > args.score_thre
                else:
                    mask = outputs["masks"][i, 0].cpu().numpy() > args.score_thre
                label = outputs["labels"][i].item()
                bbox = outputs["boxes"][i].tolist()
                encoded_mask = encode_mask(mask)

                results.append(
                    {
                        "image_id": image_id,
                        "bbox": bbox,
                        "score": float(outputs["scores"][i]),
                        "category_id": label,
                        "segmentation": encoded_mask,
                    }
                )
            progress.update(task, advance=1)

    with open(os.path.join(args.save_dir, "test-results.json"), "w") as f:
        json.dump(results, f)
    visualize_random_segmentation(
        image_info, results, test_dir, args.save_dir, num_samples=2
    )
    print(
        f"\n[green]save prediction result in [yellow2]{args.save_dir}[/yellow2] folder"
    )


if __name__ == "__main__":
    main()
