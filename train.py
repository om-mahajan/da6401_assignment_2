"""Training entrypoint"""
import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import wandb

from data.pets_dataset import OxfordIIITPetDataset
from models.multitask import MultiTaskPerceptionModel
from losses.iou_loss import IoULoss

def get_transforms(img_size: int = 224):
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_ids']))

    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_ids']))

    return train_transform, val_transform

def train_one_epoch(model, train_loader, optimizer, loss_fns, weights, device):
    model.train()
    total_loss, cls_loss_sum, loc_loss_sum, seg_loss_sum = 0, 0, 0, 0
    
    for images, targets in train_loader:
        images = images.to(device)
        class_targets = targets["class_id"].to(device)
        bbox_targets = targets["bbox"].to(device)
        seg_targets = targets["segmentation"].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        
        loss_cls = loss_fns["classification"](outputs["classification"], class_targets)
        loss_loc = loss_fns["localization"](outputs["localization"], bbox_targets)
        loss_seg = loss_fns["segmentation"](outputs["segmentation"], seg_targets)
        
        loss = (weights["classification"] * loss_cls +
                weights["localization"] * loss_loc +
                weights["segmentation"] * loss_seg)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        cls_loss_sum += loss_cls.item()
        loc_loss_sum += loss_loc.item()
        seg_loss_sum += loss_seg.item()
        
    num_batches = len(train_loader)
    return {
        "train/loss": total_loss / num_batches,
        "train/loss_cls": cls_loss_sum / num_batches,
        "train/loss_loc": loc_loss_sum / num_batches,
        "train/loss_seg": seg_loss_sum / num_batches,
    }

def validate(model, val_loader, loss_fns, weights, device):
    model.eval()
    total_loss, cls_loss_sum, loc_loss_sum, seg_loss_sum = 0, 0, 0, 0
    
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            class_targets = targets["class_id"].to(device)
            bbox_targets = targets["bbox"].to(device)
            seg_targets = targets["segmentation"].to(device)
            
            outputs = model(images)
            
            loss_cls = loss_fns["classification"](outputs["classification"], class_targets)
            loss_loc = loss_fns["localization"](outputs["localization"], bbox_targets)
            loss_seg = loss_fns["segmentation"](outputs["segmentation"], seg_targets)
            
            loss = (weights["classification"] * loss_cls +
                    weights["localization"] * loss_loc +
                    weights["segmentation"] * loss_seg)
            
            total_loss += loss.item()
            cls_loss_sum += loss_cls.item()
            loc_loss_sum += loss_loc.item()
            seg_loss_sum += loss_seg.item()
            
    num_batches = len(val_loader)
    return {
        "val/loss": total_loss / num_batches,
        "val/loss_cls": cls_loss_sum / num_batches,
        "val/loss_loc": loc_loss_sum / num_batches,
        "val/loss_seg": seg_loss_sum / num_batches,
    }

def main(args):
    # Pass entity if provided, otherwise leave it to default wandb settings
    wandb_kwargs = {
        "project": "DA6401-Assignment2",
        "name": args.run_name,
        "config": vars(args)
    }
    if args.wandb_entity:
        wandb_kwargs["entity"] = args.wandb_entity
        
    wandb.init(**wandb_kwargs)
    config = wandb.config
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_transform, val_transform = get_transforms(args.img_size)
    
    train_dataset = OxfordIIITPetDataset(root=args.data_dir, split="train", transform=train_transform)
    val_dataset = OxfordIIITPetDataset(root=args.data_dir, split="val", transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Initialize model with user-provided parameters
    model = MultiTaskPerceptionModel(num_breeds=37, seg_classes=3, in_channels=3, dropout_p=args.dropout_p).to(device)
    
    # Handle transfer learning / backbone freezing
    if args.freeze_backbone:
        print("Freezing the entire VGG11 encoder backbone...")
        for param in model.encoder.parameters():
            param.requires_grad = False
    elif args.partial_unfreeze:
        print("Partially freezing VGG11 encoder: only unfreezing conv4 and conv5...")
        for param in model.encoder.parameters():
            param.requires_grad = False
        
        # Unfreeze conv4 & conv5 based on our VGG11 block structure
        for block in [model.encoder.conv4_1, model.encoder.bn4_1, model.encoder.conv4_2, model.encoder.bn4_2,
                      model.encoder.conv5_1, model.encoder.bn5_1, model.encoder.conv5_2, model.encoder.bn5_2]:
            for param in block.parameters():
                param.requires_grad = True

    loss_fns = {
        "classification": nn.CrossEntropyLoss(),
        "localization": IoULoss(reduction="mean"),
        "segmentation": nn.CrossEntropyLoss(ignore_index=255)
    }
    
    weights = {
        "classification": args.w_cls,
        "localization": args.w_loc,
        "segmentation": args.w_seg
    }
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    best_val_loss = float("inf")
    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fns, weights, device)
        val_metrics = validate(model, val_loader, loss_fns, weights, device)
        
        wandb.log({**train_metrics, **val_metrics, "epoch": epoch})
        
        if val_metrics["val/loss"] < best_val_loss:
            best_val_loss = val_metrics["val/loss"]
            torch.save(model.state_dict(), save_dir / f"{args.run_name}_best.pth")
            print(f"  >> Saved best model with val_loss: {best_val_loss:.4f}")
            
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/oxford-iiit-pet")
    parser.add_argument("--run_name", type=str, default="baseline")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--wandb_entity", type=str, default=None, help="Your W&B team/organization name")
    
    # Model configuration
    parser.add_argument("--dropout_p", type=float, default=0.5, help="Dropout probability")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze VGG11 encoder backbone (Strict feature extractor)")
    parser.add_argument("--partial_unfreeze", action="store_true", help="Only unfreeze the last 2 conv blocks of VGG11 encoder")

    # Loss weights
    parser.add_argument("--w_cls", type=float, default=1.0)
    parser.add_argument("--w_loc", type=float, default=5.0) # IoU needs higher weight generally
    parser.add_argument("--w_seg", type=float, default=1.0)
    
    args = parser.parse_args()
    main(args)