from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
import tempfile
import json
import torch
from torch.amp import GradScaler, autocast
import numpy as np
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn_v2,
    MaskRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import argparse
from rich.progress import (
    Progress,
    TimeElapsedColumn,
    TimeRemainingColumn,
    BarColumn,
    TextColumn,
)


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_training_augmentation():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-tr",
        "--train_root",
        type=str,
        default="data/train",
        help="Path to training images",
    )
    parser.add_argument(
        "-bs", "--batch_size", type=int, default=1, help="Batch size for training"
    )
    parser.add_argument(
        "-as",
        "--accumulation_steps",
        type=int,
        default=4,
        help="Accumulation steps for loss back propagation",
    )
    parser.add_argument("-l", "--lr", type=float, default=0.004, help="Learning rate")
    parser.add_argument(
        "-e", "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training",
    )
    parser.add_argument(
        "-w",
        "--num_workers",
        type=int,
        default=1,
        help="Number of worker threads for DataLoader",
    )
    parser.add_argument(
        "-s",
        "--save_dir",
        type=str,
        default="save_model",
        help="The directory where the weight is saved",
    )
    parser.add_argument(
        "-pt",
        "--partial_training_data",
        type=float,
        default=-1,
        help="Ratio of training data to be used (0~1); if out of range, use all data",
    )
    parser.add_argument(
        "-pv",
        "--partial_validation_data",
        type=float,
        default=-1,
        help="Ratio of validation data to be used (0~1); if out of range, use all data",
    )
    return parser.parse_args()


def get_inference_augmentation():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-dr",
        "--data_root",
        type=str,
        default="data",
        help="Path to the test images",
    )
    parser.add_argument(
        "-mp",
        "--model_path",
        type=str,
        default="save_model/epoch10.pth",
        help="Path to the saved model",
    )
    parser.add_argument(
        "-bs", "--batch_size", type=int, default=1, help="Batch size for test"
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for test",
    )
    parser.add_argument(
        "-nw",
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker threads for DataLoader",
    )
    parser.add_argument(
        "-sd",
        "--save_dir",
        type=str,
        default="save_result",
        help="The directory where the result is saved",
    )
    parser.add_argument(
        "-st",
        "--score_thre",
        type=float,
        default=0.8,
        help="The score threshold to pick the box when predicting the whole number",
    )
    return parser.parse_args()


def load_model(weight_path, device="cuda", model_type="train"):

    model = maskrcnn_resnet50_fpn_v2(
        weights=(
            MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if weight_path is None else None
        )
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 5)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, 5)
    if weight_path is not None:
        model.load_state_dict(
            torch.load(weight_path, map_location=torch.device("cpu"), weights_only=True)
        )
    model.to(device)

    if model_type == "train":
        model.train()
    else:
        model.eval()
    return model


def encode_mask(mask):
    rle = maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def train_one_epoch(model, optimizer, data_loader, device, accumulation_steps):
    model.train()
    scaler = GradScaler(device=device)
    loss_all = 0.0

    optimizer.zero_grad()

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.1f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("[green]Training...", total=len(data_loader))
        for step, (images, targets) in enumerate(data_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with autocast(device_type="cuda"):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                loss_scaled = losses / accumulation_steps

            scaler.scale(loss_scaled).backward()

            if (step + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            loss_all += losses.item()
            progress.update(task, advance=1)

            if step % 10 == 0:
                torch.cuda.empty_cache()

        if len(data_loader) % accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    return loss_all / len(data_loader)


def evaluate_map(model, data_loader, device):
    model.eval()
    results = []
    coco_gt = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": f"class_{i}"} for i in range(1, 5)],
    }
    ann_id = 1
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.1f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("[cyan]Validation...", total=len(data_loader))
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            with torch.no_grad():
                outputs = model(images)

            for i, output in enumerate(outputs):
                image_id = int(targets[i]["image_id"].item())
                h, w = images[i].shape[-2:]
                coco_gt["images"].append(
                    {
                        "id": image_id,
                        "height": h,
                        "width": w,
                        "file_name": f"{image_id}.jpg",
                    }
                )

                gt_masks = targets[i]["masks"].cpu().numpy()
                gt_labels = targets[i]["labels"].cpu().numpy()
                for j in range(len(gt_masks)):
                    mask = gt_masks[j]
                    encoded = maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))
                    encoded["counts"] = encoded["counts"].decode("utf-8")
                    coco_gt["annotations"].append(
                        {
                            "id": ann_id,
                            "image_id": image_id,
                            "category_id": int(gt_labels[j]),
                            "segmentation": encoded,
                            "iscrowd": 0,
                            "area": int(mask.sum()),
                            "bbox": list(maskUtils.toBbox(encoded)),
                        }
                    )
                    ann_id += 1

                pred_masks = output["masks"].cpu().numpy()
                pred_labels = output["labels"].cpu().numpy()
                pred_scores = output["scores"].cpu().numpy()

                for k in range(len(pred_scores)):
                    if pred_scores[k] < 0.5:
                        continue
                    mask = pred_masks[k, 0] > 0.5
                    encoded = maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))
                    encoded["counts"] = encoded["counts"].decode("utf-8")
                    results.append(
                        {
                            "image_id": image_id,
                            "category_id": int(pred_labels[k]),
                            "segmentation": encoded,
                            "score": float(pred_scores[k]),
                        }
                    )
            progress.update(task, advance=1)

    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".json"
    ) as pred_f, tempfile.NamedTemporaryFile(mode="w+", suffix=".json") as gt_f:

        json.dump(results, pred_f)
        json.dump(coco_gt, gt_f)
        pred_f.flush()
        gt_f.flush()

        cocoGt = COCO(gt_f.name)
        cocoDt = cocoGt.loadRes(pred_f.name)

        cocoEval = COCOeval(cocoGt, cocoDt, iouType="segm")
        cocoEval.params.iouThrs = np.array([0.5], dtype=np.float32)
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        map_50 = cocoEval.stats[0]

    return map_50
