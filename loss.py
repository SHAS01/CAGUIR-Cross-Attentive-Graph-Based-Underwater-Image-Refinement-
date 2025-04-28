import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

class Charbonnier_Loss(nn.Module):
    def __init__(self, eps=1e-3):
        super(Charbonnier_Loss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = torch.add(x, -y)
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss


class VGG_Loss(nn.Module):
    def __init__(self, n_layers=5):
        super(VGG_Loss, self).__init__()

        feature_layers = (2, 7, 12, 21, 30)
        self.weights = (1.0, 1.0, 1.0, 1.0, 1.0)

        vgg = torchvision.models.vgg19(pretrained=True).features

        self.layers = nn.ModuleList()
        prev_layer = 0
        for next_layer in feature_layers[:n_layers]:
            layers = nn.Sequential()
            for layer in range(prev_layer, next_layer):
                layers.add_module(str(layer), vgg[layer])
            self.layers.append(layers.cuda())
            prev_layer = next_layer

        for param in self.parameters():
            param.requires_grad = False

        self.criterion = nn.L1Loss().cuda()

    def forward(self, x, y):
        loss = 0
        for layer, weight in zip(self.layers, self.weights):
            x = layer(x)
            with torch.no_grad():
                y = layer(y)
            loss += weight * self.criterion(x, y)
        return loss


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


class SSIM_Loss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM_Loss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def torchPSNR(tar_img, prd_img):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between target and predicted images.
    """
    mse = torch.mean((tar_img - prd_img) ** 2)
    if mse == 0:
        return torch.tensor(float('inf'))  # Return infinity if MSE is zero
    max_pixel = 1.0  # Assuming the pixel values are normalized to [0, 1]
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr


def torchFSIM(tar_img, prd_img):
    fsim_value = (2 * tar_img * prd_img + 0.01) / (tar_img ** 2 + prd_img ** 2 + 0.01)
    return fsim_value.mean()


def torchSSIM(tar_img, prd_img, win_size=3):
    """
    Compute Structural Similarity Index (SSIM) between target and predicted images.
    """
    tar_img_np = tar_img.detach().cpu().numpy()
    prd_img_np = prd_img.detach().cpu().numpy()

    # Handle batch dimension
    if tar_img_np.ndim == 4:
        tar_img_np = tar_img_np[0]
        prd_img_np = prd_img_np[0]

    # Convert to HWC format for SSIM calculation
    tar_img_np = np.transpose(tar_img_np, (1, 2, 0))
    prd_img_np = np.transpose(prd_img_np, (1, 2, 0))

    # Ensure pixel values are in [0, 1]
    tar_img_np = np.clip(tar_img_np, 0, 1)
    prd_img_np = np.clip(prd_img_np, 0, 1)

    # Compute SSIM with a smaller win_size
    ssim_value = ssim(tar_img_np, prd_img_np, multichannel=True, data_range=1.0, win_size=win_size)
    return torch.tensor(ssim_value)


def torchMSE(tar_img, prd_img):
    return ((prd_img - tar_img) ** 2).mean()


def torchUIQM(tar_img, prd_img):
    tar_img = tar_img.detach().cpu().numpy()
    prd_img = prd_img.detach().cpu().numpy()

    if tar_img.ndim == 4:
        tar_img = tar_img[0]
        prd_img = prd_img[0]

    tar_img = np.transpose(tar_img, (1, 2, 0))
    prd_img = np.transpose(prd_img, (1, 2, 0))

    tar_gray = np.mean(tar_img, axis=2)
    prd_gray = np.mean(prd_img, axis=2)

    luminance_tar = np.mean(tar_gray)
    luminance_prd = np.mean(prd_gray)
    contrast_tar = np.std(tar_gray)
    contrast_prd = np.std(prd_gray)
    structure = np.corrcoef(tar_gray.flatten(), prd_gray.flatten())[0, 1]

    uiqm_value = (luminance_prd / luminance_tar) * (contrast_prd / contrast_tar) * structure
    return uiqm_value


def torchUCIQE(tar_img, prd_img):
    tar_img = tar_img.detach().cpu().numpy()
    prd_img = prd_img.detach().cpu().numpy()

    if tar_img.ndim == 4:
        tar_img = tar_img[0]
        prd_img = prd_img[0]

    tar_img = np.transpose(tar_img, (1, 2, 0))
    prd_img = np.transpose(prd_img, (1, 2, 0))

    tar_img = (tar_img * 255).clip(0, 255).astype(np.uint8)
    prd_img = (prd_img * 255).clip(0, 255).astype(np.uint8)

    tar_hsv = cv2.cvtColor(tar_img, cv2.COLOR_RGB2HSV)
    prd_hsv = cv2.cvtColor(prd_img, cv2.COLOR_RGB2HSV)

    brightness_tar = np.mean(tar_hsv[:, :, 2])
    brightness_prd = np.mean(prd_hsv[:, :, 2])
    colorfulness_tar = np.mean(tar_hsv[:, :, 0])
    colorfulness_prd = np.mean(prd_hsv[:, :, 0])

    uciqe_value = (brightness_prd / brightness_tar) * (colorfulness_prd / colorfulness_tar)
    return uciqe_value