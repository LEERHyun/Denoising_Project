from model import HybridNAFNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary 
from ptflops import get_model_complexity_info
from torchvision import transforms
import data
from data import DenoiseDataset
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
import lr_scheduler
import os
from model import NAFNet


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = HybridNAFNet()
naf = NAFNet()
model.to(device)
naf.to(device)
inp_shape = (3,256,256)
macs, params = get_model_complexity_info(model, inp_shape, verbose=False, print_per_layer_stat=False)
params = float(params[:-3])
macs = float(macs[:-4])
torchsummary.summary(model,inp_shape)

print(f"Custom MACS: {macs}, PARAMS:{params}")

#Init Dataset
dataset_dir = r"C:\Users\Ahhyun\Desktop\Workplace\Dataset\Patched_Dataset"  

dataset = DenoiseDataset(root_dir=dataset_dir, transform=None)



#split Data
batch_size = 16
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
criterion = nn.MSELoss()
criterion.cuda()
optimizer = optim.Adam(model.parameters(),betas=(0.9,0.9),lr=0.001)
scheduler = lr_scheduler.MultiStepRestartLR(optimizer,milestones=[30,60,90],gamma=0.1,restarts=[45])
num_epochs = 100
checkpoint_dir = r"C:\Users\Ahhyun\Desktop\Workplace\Code\Denoising_Project-main\Model\Custom\checkpoint"

#PSNR
def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()



for epoch in tqdm(range(num_epochs), desc="Training Progress"):
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
    
     # Validation step with PSNR
    model.eval()
    total_psnr = 0
    with torch.no_grad():
        for noisy_images, gt_images in val_loader:
            noisy_images, gt_images = noisy_images.to(device), gt_images.to(device)
            outputs = model(noisy_images)
            psnr = calculate_psnr(outputs, gt_images)
            total_psnr += psnr
    avg_psnr = total_psnr / len(val_loader)
    print(f"Validation PSNR after Epoch {epoch+1}: {avg_psnr:.2f} dB")
    
    #Scheduler 업데이트
    scheduler.step()
    
    # Checkpoint 저장 
    if (epoch + 1) % 5 == 0:
        checkpoint_path = os.path.join(checkpoint_dir,f'checkpoint_{epoch+1}.pth')
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at Epoch {epoch+1}")
    torch.save(model.state_dict(), "hybrid_model.pth")


torch.save(model.state_dict(), "hybrid_model.pth")
print("Final model saved at hybrid_model.pth")