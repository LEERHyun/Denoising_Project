from model import HybridNAFNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary 
from ptflops import get_model_complexity_info
from torchvision import transforms
import data
from data import SIDD, RENOIR, PolyUDataset, Custom, DnD
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
import lr_scheduler
import os
from model import NAFNet
import glob
from skimage.metrics import structural_similarity as ssim
import lpips


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Init Model
model = HybridNAFNet(img_channel=3,out_channel=3).to(device)
#model = NAFNet().to(device)
#model = HybridNAFNet(img_channel=3,  out_channel = 3, middle_blk_num=12,enc_blk_nums=[4,2,6],dec_blk_nums=[2,2,4],refinement=4).to(device)
model_path = r"C:\Users\Ahhyun\Desktop\Workplace\Model\hybrid_model_18.pth"
model.load_state_dict(torch.load(model_path, map_location=device,weights_only=True))
model.eval()


#PSNR
def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()

# SSIM
def calculate_ssim(img1, img2):
    img1_np = img1.squeeze().cpu().numpy().transpose(1, 2, 0)
    img2_np = img2.squeeze().cpu().numpy().transpose(1, 2, 0)
    return ssim(img1_np, img2_np, data_range=1.0, channel_axis=2)

#LPIPS
lpips_model = lpips.LPIPS(net='alex').to(device)


#Dataload
#dataset_dir = r"C:\Users\Ahhyun\Desktop\Workplace\Dataset\Patched_Dataset"
#dataset = RENOIR(transform=None)
#dataset = PolyUDataset(transform=None)
#dataset = DnD()
dataset = Custom()
batch_size = 2
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


#Init Parameter

total_psnr = 0
total_ssim = 0
total_lpips = 0
count = 0


with torch.no_grad():
    for noisy_images, gt_images in tqdm(test_loader, desc="Evaluating"):
        noisy_images = noisy_images.to(device)
        gt_images = gt_images.to(device)

        outputs = model(noisy_images)
        outputs = torch.clamp(outputs, 0.0, 1.0)  # 이미지 범위 정규화
        gt_images = torch.clamp(gt_images, 0.0, 1.0)

        for i in range(outputs.size(0)):
            out_img = outputs[i:i+1]
            gt_img = gt_images[i:i+1]

            psnr = calculate_psnr(out_img, gt_img) #Psnr
            ssim_val = calculate_ssim(out_img, gt_img) #SSIM
            lpips_val = lpips_model(out_img, gt_img).item() #LPIPS

            total_psnr += psnr
            total_ssim += ssim_val
            total_lpips += lpips_val
            count += 1

# 전체 성능 계산산
avg_psnr = total_psnr / count
avg_ssim = total_ssim / count
avg_lpips = total_lpips / count

print(f"Average PSNR:  {avg_psnr:.2f} dB")
print(f"Average SSIM:  {avg_ssim:.4f}")
print(f"Average LPIPS: {avg_lpips:.4f}")