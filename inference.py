"""Inference and evaluation"""
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import numpy as np

from data.pets_dataset import OxfordIIITPetDataset
from models.multitask import MultiTaskPerceptionModel
from train import get_transforms

def iou(boxA, boxB):
    # box format: [xc, yc, w, h]
    xa_c, ya_c, wa, ha = boxA
    xb_c, yb_c, wb, hb = boxB
    
    xa1, ya1, xa2, ya2 = xa_c - wa/2, ya_c - ha/2, xa_c + wa/2, ya_c + ha/2
    xb1, yb1, xb2, yb2 = xb_c - wb/2, yb_c - hb/2, xb_c + wb/2, yb_c + hb/2
    
    x1 = max(xa1, xb1)
    y1 = max(ya1, yb1)
    x2 = min(xa2, xb2)
    y2 = min(ya2, yb2)
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    boxA_area = (xa2 - xa1) * (ya2 - ya1)
    boxB_area = (xb2 - xb1) * (yb2 - yb1)
    
    if_iou = inter_area / float(boxA_area + boxB_area - inter_area + 1e-6)
    return if_iou

def evaluate(model, data_loader, device):
    model.eval()
    
    all_preds_cls = []
    all_targets_cls = []
    
    iou_scores = []
    
    dice_scores = []
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            outputs = model(images)
            
            # Classification
            preds_cls = torch.argmax(outputs["classification"], dim=1).cpu().numpy()
            target_cls = targets["class_id"].numpy()
            all_preds_cls.extend(preds_cls)
            all_targets_cls.extend(target_cls)
            
            # Localization
            preds_loc = outputs["localization"].cpu().numpy()
            target_loc = targets["bbox"].numpy()
            for p_box, t_box in zip(preds_loc, target_loc):
                if np.sum(t_box) > 0: # Only if box exists
                    iou_scores.append(iou(p_box, t_box))
                    
            # Segmentation
            preds_seg = torch.argmax(outputs["segmentation"], dim=1).cpu().numpy()
            target_seg = targets["segmentation"].numpy()
            
            # Dice score for foreground (class 1 and 2 usually, 0 is background)
            for p_seg, t_seg in zip(preds_seg, target_seg):
                intersection = np.sum((p_seg == t_seg) & (t_seg != 255))
                union = np.sum(p_seg != 255) + np.sum(t_seg != 255)
                dice = (2.0 * intersection) / (union + 1e-6)
                dice_scores.append(dice)
                
    macro_f1 = f1_score(all_targets_cls, all_preds_cls, average="macro")
    mean_iou = np.mean(iou_scores)
    mean_dice = np.mean(dice_scores)
    
    return {
        "macro_f1": macro_f1,
        "mAP (mIoU here)": mean_iou,
        "dice_score": mean_dice
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/oxford-iiit-pet")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_transform = get_transforms(args.img_size)
    
    test_dataset = OxfordIIITPetDataset(root=args.data_dir, split="test", transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = MultiTaskPerceptionModel(num_breeds=37, seg_classes=3, in_channels=3).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    
    metrics = evaluate(model, test_loader, device)
    
    print("-" * 40)
    print(f"Evaluation Results from {args.ckpt}")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print("-" * 40)