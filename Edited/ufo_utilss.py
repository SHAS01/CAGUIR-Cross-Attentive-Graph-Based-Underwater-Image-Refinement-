import os
import sys

import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

def is_image_file(filename):
    return any(filename.lower().endswith(extension) for extension in ['jpeg', 'jpg', 'png', 'gif', 'bmp'])

class TestLoader(Dataset):
    def __init__(self, test_path):
        super(TestLoader, self).__init__()
        self.test_path = test_path

        self.test_list = []
        for root, _, files in os.walk(test_path):
            for file in files:
                if is_image_file(file):
                    self.test_list.append(os.path.join(root, file))

        if len(self.test_list) == 0:
            raise FileNotFoundError(f"No images found in {test_path}")

        print(f"Found {len(self.test_list)} test images in {test_path}")

        self.transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, idx):
        img_path = self.test_list[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, os.path.basename(img_path)

    def __len__(self):
        return len(self.test_list)

class TrainValLoader(Dataset):
    def __init__(self, enhan_images_path, ori_images_path):
        super(TrainValLoader, self).__init__()
        self.enhan_path = enhan_images_path
        self.ori_path = ori_images_path

        # Load all matching image files
        self.image_list = sorted([
            f for f in os.listdir(ori_images_path) if is_image_file(f)
        ])

        if len(self.image_list) == 0:
            raise FileNotFoundError(f"No matching image pairs found between {ori_images_path} and {enhan_images_path}")

        print(f"Found {len(self.image_list)} image pairs (original + enhanced)")

        self.transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_clean = Image.open(os.path.join(self.enhan_path, img_name)).convert('RGB')
        img_ori = Image.open(os.path.join(self.ori_path, img_name)).convert('RGB')

        img_clean = self.transform(img_clean)
        img_ori = self.transform(img_ori)

        return img_clean, img_ori

    def __len__(self):
        return len(self.image_list)

def load_checkpoint(model, checkpoint_path):
    """Load model checkpoint"""
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return False
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint from {checkpoint_path}")
        return True
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return False
