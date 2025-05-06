import os
import torch
from torch.utils.data import DataLoader
from dataset import InstanceSegmentationDataset, SegmentationTransform
import numpy as np
from utils import (
    load_model,
    train_one_epoch,
    evaluate_map,
    get_training_augmentation,
    count_trainable_parameters,
)
from rich import print
from torch.optim.lr_scheduler import StepLR


def main():
    args = get_training_augmentation()
    device = args.device

    full_dataset = InstanceSegmentationDataset(args.train_root, transform=None)
    val_size = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    indices = torch.randperm(len(full_dataset))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = torch.utils.data.Subset(
        InstanceSegmentationDataset(
            args.train_root, transform=SegmentationTransform(train=True)
        ),
        train_indices,
    )
    val_dataset = torch.utils.data.Subset(
        InstanceSegmentationDataset(
            args.train_root, transform=SegmentationTransform(train=False)
        ),
        val_indices,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x)),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x)),
    )

    model = load_model(None, device, model_type="train")
    print(f"model parameters: {count_trainable_parameters(model)}")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    train_losses = []
    val_maps = []

    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\n[yellow2]Epoch \[{epoch+1}/{args.epochs}]")
        train_loss = train_one_epoch(
            model, optimizer, train_loader, device, args.accumulation_steps
        )
        scheduler.step()
        val_map = evaluate_map(model, val_loader, device)

        train_losses.append(train_loss)
        val_maps.append(val_map)

        torch.save(
            model.state_dict(), os.path.join(args.save_dir, f"epoch{epoch+1}.pth")
        )
        print(
            f"Train Loss={train_loss:.4f}, lr={scheduler.get_last_lr()[0]:.6f}, Val mAP50={val_map:.4f}"
        )
        print(
            f"save model in [yellow2]{os.path.join(args.save_dir, f'epoch{epoch+1}.pth')}"
        )

    np.save(os.path.join(args.save_dir, "train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(args.save_dir, "val_mAP50.npy"), np.array(val_maps))
    print(
        f"\n[green]save loss and mAP numpy file in [yellow2]{args.save_dir}[/yellow2] folder"
    )


if __name__ == "__main__":
    main()
