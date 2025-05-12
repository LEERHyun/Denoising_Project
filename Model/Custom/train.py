from model import HybridNAFNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary 
from ptflops import get_model_complexity_info
from torchvision import transforms
import data
from data import SIDD, SIDD_Medium
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
import lr_scheduler
import os
from model import NAFNet,HybridNAFNet,Restormer
import glob
from skimage.metrics import structural_similarity as ssim
import lpips
import model_edit
from model_edit import HybridNAFNet_Edit
import losses
from losses import PSNRLoss



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = HybridNAFNet_Edit(enc_blk_nums=[4,2,6],middle_blk_num=12,dec_blk_nums=[4,2,2],refinement=4)
#naf = NAFNet()
model.to(device)
#naf.to(device)
inp_shape = (3,256,256)
macs, params = get_model_complexity_info(model, inp_shape, verbose=False, print_per_layer_stat=False)
params = float(params[:-3])
macs = float(macs[:-4])
#torchsummary.summary(model,inp_shape)

print(f"Custom MACS: {macs}, PARAMS:{params}")

#Init Dataset
dataset_dir = r"C:\Users\Ahhyun\Desktop\Workplace\Dataset\SIDD_Patched_Dataset"  

dataset = SIDD_Medium(root_dir=dataset_dir, transform=None)



#split Data
data_size = len(dataset)
train_size = int(0.8*data_size)
val_size = int(0.1*data_size)
test_size = data_size - train_size - val_size

from torch.utils.data import random_split
train_dataset, test_dataset, val_dataset = random_split(dataset, [train_size,test_size,val_size])

batch_size = 2
train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
print("Dataset prepared")

#Parameter
criterion = PSNRLoss()
criterion.cuda()
optimizer = optim.Adam(model.parameters(),betas=(0.9,0.9),lr=0.001)
total_iter = int(train_size/batch_size)
num_epochs = 1000
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max = total_iter,eta_min=1e-7)
checkpoint_dir = r"C:\Users\Ahhyun\Desktop\Workplace\Code\Denoising_Project-main\Model\Custom\checkpoint"

#PSNR
def calculate_psnr(img1, img2):
    "img1 ,img2 range [0,1]"
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


#Checkpoint 여부
use_checkpoint = input("체크포인트를 불러올까요? (y/n): ").strip().lower() == 'y'


#Load Checkpoint
checkpoint_dir = r'C:\Users\Ahhyun\Desktop\Workplace\Code\Denoising_Project-main\Model\Custom\checkpoint'
checkpoint_pattern = os.path.join(checkpoint_dir, 'checkpoint_*.pth')

if use_checkpoint:
    checkpoint_files = glob.glob(checkpoint_pattern)
    if checkpoint_files:
        # 가장 최근에 수정된 체크포인트 파일 선택
        latest_ckpt = max(checkpoint_files, key=os.path.getmtime)
        print(f"Load Checkpoint: {latest_ckpt}")

        checkpoint = torch.load(latest_ckpt, weights_only = True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Start with {start_epoch}.")
    else:
        print("No Checkpoint file Found. Start with new file")
        start_epoch = 0
else:
    print("Start with new file")
    start_epoch = 0

for epoch in tqdm(range(start_epoch,num_epochs), desc="Training Progress"):
    model.train()
    train_loss = 0
    
    for noisy_images, gt_images in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        noisy_images, gt_images = noisy_images.to(device), gt_images.to(device)
        
        # Forward pass
        outputs = model(noisy_images)
        loss = criterion(outputs, gt_images)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_loss = train_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    total_psnr = 0
    total_ssim = 0
    total_lpips = 0
     # Validation step with PSNR
    model.eval()
    total_psnr = 0
    with torch.no_grad():
        for noisy_images, gt_images in tqdm(val_loader, desc="Evaluating"):
            noisy_images, gt_images = noisy_images.to(device), gt_images.to(device)
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
    avg_psnr = total_psnr / val_size
    avg_ssim = total_ssim / val_size
    avg_lpips = total_lpips / val_size
    print(f"Validation PSNR after Epoch {epoch+1}: {avg_psnr:.2f} dB")
    print(f"Validation SSIM: {avg_ssim:.4f}")
    print(f"Validation LPIPS: {avg_lpips:.4f}")
    
    #Scheduler 업데이트
    scheduler.step()
    
    # Checkpoint 저장 
    if (epoch + 1) % 4 == 0:
        checkpoint_path = os.path.join(checkpoint_dir,f'checkpoint_{epoch+1}.pth')
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at Epoch {epoch+1}")
    torch.save(model.state_dict(), "hybrid_model.pth") #Test용용


torch.save(model.state_dict(), "hybrid_model.pth")
print("Final model saved at hybrid_model.pth")


from skimage.metrics import structural_similarity as compare_ssim
import lpips
import torchvision.transforms.functional as TF
import numpy as np

# LPIPS 모델 초기화
loss_fn_alex = lpips.LPIPS(net='alex').to(device)
loss_fn_alex.eval()


model.eval()
total_psnr = 0.0
total_ssim = 0.0
total_lpips = 0.0
num_samples = 0

print("Evaluating on Test Dataset...")

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


# 평균 값 계산
avg_psnr = total_psnr / test_size
avg_ssim = total_ssim / test_size
avg_lpips = total_lpips / test_size

print(f"\n=== Test Results ===")
print(f"Average PSNR: {avg_psnr:.2f} dB")
print(f"Average SSIM: {avg_ssim:.4f}")
print(f"Average LPIPS: {avg_lpips:.4f}")
