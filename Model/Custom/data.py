from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

class DenoiseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        #root-dir: Root Directory of Dataset
        #transform(callable, optional): transform
            
        self.root_dir = root_dir
        self.transform = transform
        self.image_pairs = []
        self.transform = transforms.Compose([
        transforms.Resize((256, 256)),  # resize
        transforms.ToTensor()         # totensor 
        ])
        for image_folder in os.listdir(root_dir):
            image_folder_path = os.path.join(root_dir, image_folder)
            if not os.path.isdir(image_folder_path):
                continue

            for patch_folder in os.listdir(image_folder_path):
                patch_folder_path = os.path.join(image_folder_path, patch_folder)
                if not os.path.isdir(patch_folder_path):
                    continue

                gt_path = os.path.join(patch_folder_path, "GT_SRGB_010.png")
                noisy_path = os.path.join(patch_folder_path, "NOISY_SRGB_010.png")
                # GT 및 노이즈 이미지가 모두 있는 경우만
                if os.path.exists(gt_path) and os.path.exists(noisy_path):
                    self.image_pairs.append((gt_path, noisy_path))
                    
    def __len__(self):
        return len(self.image_pairs)
        
    def __getitem__(self, idx):
        gt_path,noisy_path = self.image_pairs[idx]
            
        noisy_img = Image.open(noisy_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")
            
        if self.transform:
            noisy_img = self.transform(noisy_img)
            gt_img = self.transform(gt_img)
        
        return noisy_img, gt_img
    