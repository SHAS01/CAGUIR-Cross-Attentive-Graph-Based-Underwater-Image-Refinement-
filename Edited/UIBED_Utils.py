import os
import torch
import torch.utils.data as data

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif', 'bmp'])

class OriginalImageLoader(Dataset):
    """
    Dataset for loading ONLY original images (no enhanced/clean version needed).
    """
    def __init__(self, ori_images_path):
        super(OriginalImageLoader, self).__init__()
        self.ori_images_path = ori_images_path

        # Get all image files
        self.image_list = []
        for root, _, files in os.walk(ori_images_path):
            for file in files:
                if is_image_file(file):
                    self.image_list.append(os.path.join(root, file))

        if len(self.image_list) == 0:
            raise FileNotFoundError(f"No images found in {ori_images_path}")

        print(f"Found {len(self.image_list)} original images in {ori_images_path}")

        self.transform = transforms.Compose([
            transforms.Resize((320,320)),
            transforms.ToTensor()
            # You can add Resize() if necessary
        ])

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, os.path.basename(img_path)  # Returning image tensor and filename

    def __len__(self):
        return len(self.image_list)

def load_checkpoint(model, checkpoint_path):
    """Load model checkpoint."""
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

        # Remove 'module.' if needed
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded checkpoint from {checkpoint_path}")
        return True
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return False
