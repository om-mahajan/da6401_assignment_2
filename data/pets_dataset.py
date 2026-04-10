"""Dataset skeleton for Oxford-IIIT Pet.
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Optional, Tuple, Dict
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import Dataset
import urllib.request
import tarfile

class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader."""
    
    def __init__(self, root: str, split: str = "train", transform: Optional[Callable] = None):
        """
        Initialize the dataset.
        
        Args:
            root: Path to the dataset directory.
            split: One of 'train', 'val', or 'test'.
            transform: Albumentations transforms to apply.
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform
        
        self.images_dir = self.root / "images"
        self.annotations_dir = self.root / "annotations"
        self.trimaps_dir = self.annotations_dir / "trimaps"
        self.xmls_dir = self.annotations_dir / "xmls"
        
        self._download_and_extract()
        self._load_data()
        
    def _download_and_extract(self):
        """Download dataset if not present."""
        self.root.mkdir(parents=True, exist_ok=True)
        images_tar = self.root / "images.tar.gz"
        annotations_tar = self.root / "annotations.tar.gz"
        
        if not self.images_dir.exists():
            print("Downloading images...")
            urllib.request.urlretrieve("https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz", images_tar)
            with tarfile.open(images_tar, "r:gz") as tar:
                tar.extractall(path=self.root)
                
        if not self.annotations_dir.exists():
            print("Downloading annotations...")
            urllib.request.urlretrieve("https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz", annotations_tar)
            with tarfile.open(annotations_tar, "r:gz") as tar:
                tar.extractall(path=self.root)

    def _load_data(self):
        """Parse dataset metadata."""
        list_txt = self.annotations_dir / "trainval.txt" if self.split in ["train", "val"] else self.annotations_dir / "test.txt"
        
        self.samples = []
        self.classes = set()
        
        with open(list_txt, "r") as f:
            lines = f.readlines()
            
        # For simplicity and reproducibility, we randomly shuffle trainval and split 80/20 for val
        np.random.seed(42)
        if self.split in ["train", "val"]:
            np.random.shuffle(lines)
            split_idx = int(0.8 * len(lines))
            lines = lines[:split_idx] if self.split == "train" else lines[split_idx:]

        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
                
            img_name = parts[0]
            class_id = int(parts[1]) - 1 # 0-indexed class
            
            # Check if xml exists (some images don't have bounding boxes)
            xml_path = self.xmls_dir / f"{img_name}.xml"
            if not xml_path.exists():
                continue
                
            self.samples.append({
                "name": img_name,
                "class_id": class_id,
            })
            self.classes.add(class_id)
            
        self.num_classes = len(self.classes)

    def __len__(self) -> int:
        return len(self.samples)
        
    def _parse_voc_bbox(self, xml_path: Path) -> list:
        """Parse VOC XML to get [x_center, y_center, width, height] bbox."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bndbox = root.find("object/bndbox")
        if bndbox is None:
            return [0, 0, 0, 0] # fallback
            
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)
        
        x_center = (xmin + xmax) / 2.0
        y_center = (ymin + ymax) / 2.0
        width = xmax - xmin
        height = ymax - ymin
        
        return [x_center, y_center, width, height]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        sample = self.samples[idx]
        img_name = sample["name"]
        
        # Load image
        img_path = self.images_dir / f"{img_name}.jpg"
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            # Some files are corrupted in the original dataset
            # Fallback to random if corrupted
            return self.__getitem__((idx + 1) % len(self))
            
        image = np.array(image)
        
        # Load segment map
        trimap_path = self.trimaps_dir / f"{img_name}.png"
        trimap = np.array(Image.open(trimap_path)) - 1 # 0, 1, 2
        
        # Load bbox (albumentations expects [x_min, y_min, x_max, y_max, class_id])
        xml_path = self.xmls_dir / f"{img_name}.xml"
        tree = ET.parse(xml_path)
        bndbox = tree.getroot().find("object/bndbox")
        if bndbox is not None:
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
            bboxes = [[xmin, ymin, xmax, ymax]]
            class_ids = [sample["class_id"]]
        else:
            bboxes = []
            class_ids = []
            
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=trimap, bboxes=bboxes, class_ids=class_ids)
            image = transformed["image"]
            trimap = transformed["mask"]
            bboxes = transformed["bboxes"]
            
            # Albumentations returns image as [C, H, W] if ToTensorV2 is used.
            
        # Convert bounding box to [x_center, y_center, width, height]
        if len(bboxes) > 0:
            xmin, ymin, xmax, ymax = bboxes[0]
            x_center = (xmin + xmax) / 2.0
            y_center = (ymin + ymax) / 2.0
            width = xmax - xmin
            height = ymax - ymin
            bbox_tensor = torch.tensor([x_center, y_center, width, height], dtype=torch.float32)
        else:
            bbox_tensor = torch.tensor([0, 0, 0, 0], dtype=torch.float32)
            
        targets = {
            "class_id": torch.tensor(sample["class_id"], dtype=torch.long),
            "bbox": bbox_tensor,
            "segmentation": trimap.clone().detach().to(torch.long) if torch.is_tensor(trimap) else torch.tensor(trimap, dtype=torch.long)
        }
        
        if not torch.is_tensor(image):
            # Fallback if no transform applies ToTensor
            image = torchvision.transforms.functional.to_tensor(image)
            
        return image, targets