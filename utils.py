import os
import sys

import torch
import torch.utils.data as data

import numpy as np
import random

from PIL import Image
from collections import OrderedDict
from torchvision import transforms
from torch.utils.data import Dataset

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif', 'bmp'])

def train_val_list(enhan_images_path, ori_images_path):
    """Split images into train and validation sets"""
    if not os.path.exists(enhan_images_path):
        raise FileNotFoundError(f"Enhanced images path not found: {enhan_images_path}")
    if not os.path.exists(ori_images_path):
        raise FileNotFoundError(f"Original images path not found: {ori_images_path}")
        
    image_list = sorted(os.listdir(ori_images_path))
    if len(image_list) == 0:
        raise ValueError(f"No images found in {ori_images_path}")
        
    # Calculate 80-20 split
    total_images = len(image_list)
    train_size = int(total_images * 0.8)  # 80% for training
    val_size = total_images - train_size  # 20% for testing
    
    if train_size == 0 or val_size == 0:
        raise ValueError(f"Not enough images found in {ori_images_path} for training and testing split.")
    
    print(f"Total images: {total_images}")
    print(f"Training set size: {train_size} (80%)")
    print(f"Testing set size: {val_size} (20%)")
    
    return image_list[:train_size], image_list[train_size:]

class test_loader(Dataset):
    def __init__(self, test_path):
        super(test_loader, self).__init__()
        self.test_path = test_path
        
        # Get all image files from the directory
        self.test_list = []
        for root, _, files in os.walk(test_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                    self.test_list.append(os.path.join(root, file))
        
        if len(self.test_list) == 0:
            raise FileNotFoundError(f"No images found in {test_path}")
            
        print(f"Found {len(self.test_list)} test images in {test_path}")
        
        self.transform = transforms.Compose([
            transforms.Resize((320,320)),
            transforms.ToTensor(),
            # Adjust size as needed
        ])

    def __getitem__(self, idx):
        img_path = self.test_list[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, os.path.basename(img_path)

    def __len__(self):
        return len(self.test_list)

class train_val_loader(Dataset):
    def __init__(self, enhan_images_path, ori_images_path, mode="train"):
        super(train_val_loader, self).__init__()
        self.enhan_path = enhan_images_path
        self.ori_path = ori_images_path
        self.mode = mode
        
        # Get train and test splits using train_val_list
        train_list, test_list = train_val_list(enhan_images_path, ori_images_path)
        
        # Use appropriate list based on mode
        self.image_list = train_list if mode == "train" else test_list
        
        if len(self.image_list) == 0:
            raise FileNotFoundError(f"No matching image pairs found between {ori_images_path} and {enhan_images_path}")
            
        print(f"Found {len(self.image_list)} images for {mode}ing")
        
        self.transform = transforms.Compose([
            transforms.Resize((320,320)),
            transforms.ToTensor(),
            # Adjust size as needed
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
        checkpoint = torch.load(checkpoint_path)
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