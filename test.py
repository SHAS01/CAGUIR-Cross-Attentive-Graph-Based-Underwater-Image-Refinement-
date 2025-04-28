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
from loss import torchPSNR, torchSSIM, torchFSIM, torchUIQM, torchUCIQE

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
    device_ids = list(range(torch.cuda.device_count()))
    if len(device_ids) > 1:
        enhan_net = nn.DataParallel(enhan_net, device_ids=device_ids)

    # Load test data
    test_dataset = utils.train_val_loader(
        config.enhan_images_path, config.ori_images_path, mode="test"
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
        pin_memory=True
    )

    # Prepare results directory
    result_dir = os.path.join(config.result_path, config.net_name, config.dataset_name)
    os.makedirs(result_dir, exist_ok=True)

    # Evaluation
    enhan_net.eval()

    total_psnr = total_ssim = total_fsim = total_uiqm = total_uciqe = 0
    num_samples = 0

    with torch.no_grad():
        for i, (img_clean, img_ori) in enumerate(tqdm(test_loader, desc="Testing", unit="batch")):
            img_clean, img_ori = img_clean.to(device), img_ori.to(device)

            for j in range(img_ori.size(0)):  # handle batch size > 1
                single_ori = img_ori[j].unsqueeze(0)  # Add batch dim
                single_clean = img_clean[j].unsqueeze(0)

                # Build adjacency matrix for this image
                adj_matrix = build_adjacency_matrix(
                    single_ori.squeeze(0).cpu().permute(1, 2, 0).numpy(), config.block_size, k=5
                )
                adj_matrix = torch.tensor(adj_matrix, device=device)

                # Forward pass
                enhanced_img = enhan_net(single_ori, adj_matrix)
                enhanced_img = torch.clamp(enhanced_img, 0, 1)
                single_clean = torch.clamp(single_clean, 0, 1)

                # Compute metrics
                psnr_value = torchPSNR(single_clean, enhanced_img)
                ssim_value = torchSSIM(single_clean, enhanced_img, win_size=3)
                fsim_value = torchFSIM(single_clean, enhanced_img)
                uiqm_value = torchUIQM(single_clean, enhanced_img)
                uciqe_value = torchUCIQE(single_clean, enhanced_img)

                print(f"Image {i * config.batch_size + j + 1}: "
                      f"PSNR={psnr_value.item():.2f}, SSIM={ssim_value.item():.4f}, "
                      f"FSIM={fsim_value.item():.4f}, UIQM={uiqm_value.item():.2f}, "
                      f"UCIQE={uciqe_value.item():.2f}")

                total_psnr += psnr_value.item()
                total_ssim += ssim_value.item()
                total_fsim += fsim_value.item()
                total_uiqm += uiqm_value.item()
                total_uciqe += uciqe_value.item()
                num_samples += 1

                # Save enhanced images
                enhanced_img_cpu = enhanced_img.squeeze(0).cpu()
                save_path = os.path.join(result_dir, f"enhanced_{i * config.batch_size + j}.png")
                torchvision.utils.save_image(enhanced_img_cpu, save_path)

    # Average metrics
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples
    avg_fsim = total_fsim / num_samples
    avg_uiqm = total_uiqm / num_samples
    avg_uciqe = total_uciqe / num_samples

    print("\n****************************** Testing Results ******************************")
    print(f"Average PSNR: {avg_psnr:.4f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average FSIM: {avg_fsim:.4f}")
    print(f"Average UIQM: {avg_uiqm:.4f}")
    print(f"Average UCIQE: {avg_uciqe:.4f}")
    print("---------------------------------------------------------------------------")

    # Save metrics
    metrics_file = os.path.join(result_dir, "test_metrics.txt")
    with open(metrics_file, "w") as f:
        f.write("****************************** Testing Results ******************************\n")
        f.write(f"Average PSNR: {avg_psnr:.4f}\n")
        f.write(f"Average SSIM: {avg_ssim:.4f}\n")
        f.write(f"Average FSIM: {avg_fsim:.4f}\n")
        f.write(f"Average UIQM: {avg_uiqm:.4f}\n")
        f.write(f"Average UCIQE: {avg_uciqe:.4f}\n")
        f.write("---------------------------------------------------------------------------\n")


if __name__ == '__main__':
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()

    # Basic parameters
    parser.add_argument('--block_size', type=int, default=32)
    parser.add_argument('--net_name', type=str, default="UFO")
    parser.add_argument('--dataset_name', type=str, default="ls")
    parser.add_argument('--ori_images_path', type=str, default="/workspace/rds/dataset/UFO/TEST/lrd")
    parser.add_argument('--enhan_images_path', type=str, default="/workspace/rds/dataset/UFO/TEST/hr")
    parser.add_argument('--run', type=int, default=2, help="Model run number")
    parser.add_argument('--epoch', type=int, default=100,help="Epoch number to load")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--checkpoint_path', type=str, default="/workspace/rds/trained_model/")
    parser.add_argument('--result_path', type=str, default="./result/")
    parser.add_argument('--cudaid', type=str, default="0", help="CUDA device ID (0-7)")

    config = parser.parse_args()

    os.makedirs(os.path.join(config.result_path, config.net_name, config.dataset_name), exist_ok=True)

    start_time = time.time()
    test(config)
    print(f"Final time: {time.time() - start_time:.2f} seconds")
