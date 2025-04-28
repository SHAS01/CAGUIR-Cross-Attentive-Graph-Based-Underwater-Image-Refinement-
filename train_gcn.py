import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from thop import profile
from skimage.metrics import structural_similarity as ssim  # Import SSIM
from loss import Charbonnier_Loss, SSIM_Loss, torchPSNR, torchFSIM, torchUIQM, torchUCIQE  # Ensure these functions are defined

import os
import argparse
import time
import utils
from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler
from loss import Charbonnier_Loss, SSIM_Loss, torchPSNR

from networks import network, graph_network
from graphmethods import build_adjacency_matrices
import torchvision.models as models

def get_least_used_gpu():
    """Returns the GPU device ID with the least memory usage."""
    import pynvml
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    
    print(f"\nNumber of available GPUs: {device_count}")
    print("GPU Information:")
    print("-" * 50)
    
    min_memory_used = float('inf')
    best_device = 0
    
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        name = pynvml.nvmlDeviceGetName(handle)
        memory_used = info.used / (1024**2)  # Convert to MB
        memory_total = info.total / (1024**2)  # Convert to MB
        memory_free = info.free / (1024**2)  # Convert to MB
        
        print(f"GPU {i}: {name.decode('utf-8')}")
        print(f"  Memory Used: {memory_used:.2f} MB")
        print(f"  Memory Free: {memory_free:.2f} MB")
        print(f"  Memory Total: {memory_total:.2f} MB")
        print("-" * 50)
        
        if memory_used < min_memory_used:
            min_memory_used = memory_used
            best_device = i
    
    pynvml.nvmlShutdown()
    return best_device

def train(config):
    # Set CUDA device to the one with least memory usage
    device_id = get_least_used_gpu()
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")
    
    print(f"\nSelected GPU {device_id} for training")
    print(f"CUDA Device Name: {torch.cuda.get_device_name(device_id)}")
    print(f"Current CUDA Device: {torch.cuda.current_device()}")
    
    # Set hidden layer sizes
    encoder_hidden_size = 64
    gnn_hidden_size = 1500
    num_gnn_layers = 2

    # Initialize the graph network with specified hidden sizes and layers
    graph_net = graph_network.graph_net(block_size=config.block_size).to(device)

    print("gpu_id:", device_id)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    device_ids = [i for i in range(torch.cuda.device_count())]
    
    if torch.cuda.device_count() > 1:
        graph_net = nn.DataParallel(graph_net, device_ids=device_ids)

    train_dataset = utils.train_val_loader(config.enhan_images_path, config.ori_images_path, mode="train", batch_size=2, num_workers=2)
    val_dataset = utils.train_val_loader(config.enhan_images_path, config.ori_images_path, mode="val", batch_size=2, num_workers=2)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2, pin_memory=True)

    print("Data loaders created. Starting training...")

    # Training loop
    print("Starting training loop...")
    for epoch in range(config.num_epochs):
        print(f"Epoch {epoch + 1}/{config.num_epochs}")
        graph_net.train()  # Set the model to training mode

        train_loss = 0.0
        for i, (img_clean, img_ori) in enumerate(tqdm(train_loader, desc="Training Batches")):
            print(f"Processing batch {i + 1}...")
            img_clean = img_clean.cuda()
            img_ori = img_ori.cuda()
            print("Images moved to GPU.")

            optimizer.zero_grad()  # Zero the gradients
            enhanced_image = graph_net(img_ori)  # Forward pass
            print("Forward pass completed.")

            loss = criterion_char(img_clean, enhanced_image) + criterion_ssim(img_clean, enhanced_image)
            loss.backward()  # Backward pass
            print("Backward pass completed.")

            optimizer.step()  # Update weights
            print("Weights updated.")

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Average Training Loss for Epoch {epoch + 1}: {avg_train_loss:.4f}")

        # Validation step (optional)
        # You can add validation logic here if needed

    print("Training complete.")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()

    parser.add_argument('--block_size', type=int, default="16")
    parser.add_argument('--net_name', type=str, default="net_C")
    parser.add_argument('--d_net_name', type=str, default="d_net")
    parser.add_argument('--enhan_images_path', type=str, default="Datasets/UIEB/train/target/")
    parser.add_argument('--ori_images_path', type=str, default="/Datasets/UIEB/train/input/")
    parser.add_argument('--lr', type=float, default=1e-6)  # Set learning rate to 1e-6
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=100)  # Set to 100
    parser.add_argument('--train_batch_size', type=int, default=2)  # Set batch size to 2
    parser.add_argument('--val_batch_size', type=int, default=2)  # Set validation batch size to 2
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--checkpoint_path', type=str, default="./trained_model/")
    parser.add_argument('--cudaid', type=str, default="1", help="choose cuda device id 0-7).")

    config = parser.parse_args()

    # Ensure the checkpoint directory exists
    if not os.path.exists(config.checkpoint_path):
        os.makedirs(config.checkpoint_path)  # Create the parent directory if it doesn't exist

    if not os.path.exists(os.path.join(config.checkpoint_path, config.net_name)):
        os.mkdir(os.path.join(config.checkpoint_path, config.net_name))  # Create the subdirectory

    torch.cuda.empty_cache()
    s = time.time()
    train(config)
    e = time.time()
    print(str(e - s))
