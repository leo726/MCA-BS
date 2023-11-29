from PIL import Image
import torch
import torch.nn as nn
from math import log10
import numpy as np
from torch import Tensor
import os
import matplotlib.image as mp
from pytorch_msssim import ssim
import torchvision.transforms as tf
import cv2

def load_img(filepath, mode):
    img = Image.open(filepath).convert(mode)
    # y, _, _ = img.split()
    return img

def compute_mse(hr,sr):
    mse = float(nn.MSELoss()(hr,sr))
    return mse

def compute_psnr(hr, sr):
     mse = nn.MSELoss()(hr,sr)
     psnr = 10*log10(1/mse)
     return psnr

def cal_ssim(x1,x2):
    ssim_value = ssim(x1,x2).item()
    print(ssim_value)


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
 
 
def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

if __name__ == '__main__':
     jpg_path = '/home/yangling/VQGAN/taming-transformers-master/logs/2023-06-30T15-13-25_BseBoneShadowX-rays_vqgan/images/val/'
     
     i = 0
     total_psnr1 = 0
     total_psnr2 = 0
     total_ssim = 0
     total_mse = 0
     while i < 48:
         if i < 10:
             num = '0'+ str(i)
         else:
             num = str(i)
         hr_path = jpg_path + 'bse_inputs_e-000039_b-0000' + num + '.png'
         sr_path = jpg_path + 'bone_reconstructions_e-000039_b-0000' + num + '.png'
          
         hr = load_img(hr_path,'RGB')
         sr = load_img(sr_path,'RGB')
         hr = np.array(hr) / 255  #[0,1]
         sr = np.array(sr) / 255
         hr = Tensor(hr)
         sr = Tensor(sr)
         psnr1 = compute_psnr(hr, sr)
         print(psnr1)
         total_psnr1 += psnr1
         mse = compute_mse(hr,sr)
         print(mse)
         total_mse += mse

         hr1 = load_img(hr_path, 'YCbCr')
         sr1 = load_img(sr_path, 'YCbCr')
         hr1 = np.array(hr1) / 255  # [0,1]
         sr1 = np.array(sr1) / 255
         hr1 = Tensor(hr1)
         sr1 = Tensor(sr1)
         psnr2 = compute_psnr(hr1, sr1)
         print(psnr2)
         total_psnr2 += psnr2
         
         img1 = cv2.imread(hr_path, 0)
         img2 = cv2.imread(sr_path, 0)
         img1 = np.array(img1)
         img2 = np.array(img2)
         ss = ssim(img1,img2)
         print(ss)
         total_ssim += ss
         
         i += 1
         
         print('*'*40)
          
     print('psnr result:')
     print(total_psnr1/i)
     print(total_psnr2/i)
     print('ssim resutl:')
     print(total_ssim/i)
     print('mse resutl:')
     print(total_mse/i)
    