import os
import time
import argparse
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm
import numpy as np

import utils
from networks import graph_network
from graphmethods import build_adjacency_matrix

from collections import OrderedDict


def load_checkpoint(model, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if exists
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    print(f"Checkpoint loaded successfully from {checkpoint_path}")


def test(config):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cudaid

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize network
    enhan_net = graph_network.graph_net(config.block_size).to(device)

    # Load checkpoint
    checkpoint_path = os.path.join(
        config.checkpoint_path, config.net_name,
        f"model_run{config.run}_epoch{config.epoch}.pth"
    )
    print(f"Loading checkpoint: {checkpoint_path}")
    load_checkpoint(enhan_net, checkpoint_path)

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        enhan_net = nn.DataParallel(enhan_net)

    # Load original images only
    ori_dataset = utils.OriginalImageLoader(config.ori_images_path)  # <-- only original images

    ori_loader = torch.utils.data.DataLoader(
        ori_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
        pin_memory=True
    )

    # Prepare result directory
    result_dir = os.path.join(config.result_path, config.net_name, config.dataset_name)
    os.makedirs(result_dir, exist_ok=True)

    enhan_net.eval()

    with torch.no_grad():
        for i, (img_ori, img_name) in enumerate(tqdm(ori_loader, desc="Testing", unit="batch")):
            img_ori = img_ori.to(device)

            for j in range(img_ori.size(0)):  # handle batch size > 1
                single_ori = img_ori[j].unsqueeze(0)

                # Build adjacency matrix
                adj_matrix = build_adjacency_matrix(
                    single_ori.squeeze(0).cpu().permute(1, 2, 0).numpy(),
                    config.block_size,
                    k=5
                )
                adj_matrix = torch.tensor(adj_matrix, device=device)

                # Forward pass
                enhanced_img = enhan_net(single_ori, adj_matrix)
                enhanced_img = torch.clamp(enhanced_img, 0, 1)

                # Save enhanced image
                enhanced_img_cpu = enhanced_img.squeeze(0).cpu()
                save_name = os.path.splitext(img_name[j])[0] + "_enhanced.png"
                save_path = os.path.join(result_dir, save_name)
                torchvision.utils.save_image(enhanced_img_cpu, save_path)


if __name__ == '__main__':
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()

    parser.add_argument('--block_size', type=int, default=32)
    parser.add_argument('--net_name', type=str, default="UIEB")
    parser.add_argument('--dataset_name', type=str, default="ls")
    parser.add_argument('--ori_images_path', type=str, default="/workspace/rds/dataset/TEST")
    parser.add_argument('--run', type=int, default=4)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--checkpoint_path', type=str, default="/workspace/rds/trained_model/")
    parser.add_argument('--result_path', type=str, default="./result/")
    parser.add_argument('--cudaid', type=str, default="0")

    config = parser.parse_args()

    os.makedirs(os.path.join(config.result_path, config.net_name, config.dataset_name), exist_ok=True)

    start_time = time.time()
    test(config)
    print(f"Final time: {time.time() - start_time:.2f} seconds")
