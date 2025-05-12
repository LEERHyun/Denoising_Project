from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

class SIDD(Dataset):
    def __init__(self, root_dir = r"C:\Users\Ahhyun\Desktop\Workplace\Dataset\SIDD_Patched_Dataset", transform=None):
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

class SIDD_Medium(Dataset):
    def __init__(self, root_dir = r"C:\Users\Ahhyun\Desktop\Workplace\Dataset\SIDD_Patched_Dataset", transform=None):
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

                gt_path_1 = os.path.join(patch_folder_path, "GT_SRGB_010.png")
                noisy_path_1 = os.path.join(patch_folder_path, "NOISY_SRGB_010.png")
                
                gt_path_2 = os.path.join(patch_folder_path, "GT_SRGB_011.png")
                noisy_path_2 = os.path.join(patch_folder_path, "NOISY_SRGB_011.png")
                # GT 및 노이즈 이미지가 모두 있는 경우만
                if os.path.exists(gt_path_1) and os.path.exists(noisy_path_1):
                    self.image_pairs.append((gt_path_1, noisy_path_1))
                    
                if os.path.exists(gt_path_2) and os.path.exists(noisy_path_2):
                    self.image_pairs.append((gt_path_2, noisy_path_2))                    
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
    

class RENOIR(Dataset):
    def __init__(self, root_dir = r"C:\Users\Ahhyun\Desktop\Workplace\Dataset\RENOIR_Patched_Dataset", transform=None):
        self.root_dir = root_dir
        self.image_pairs = []        
        self.transform = transforms.Compose([        
        transforms.Resize((256, 256)),                                   
        transforms.ToTensor()         # totensor 
        ])

        # 모든 Patch 폴더 경로 수집
        for image_folder in os.listdir(root_dir):
            image_folder_path = os.path.join(root_dir, image_folder)
            if not os.path.isdir(image_folder_path):
                continue

            for patch_folder in os.listdir(image_folder_path):
                patch_folder_path = os.path.join(image_folder_path, patch_folder)
                if not os.path.isdir(patch_folder_path):
                    continue

                gt_path = os.path.join(patch_folder_path, "Reference.bmp")
                noisy_path = os.path.join(patch_folder_path, "Noisy.bmp")
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
            gt_img = self.transform(gt_img)
            noisy_img = self.transform(noisy_img)

        return noisy_img, gt_img  # 입력: noisy, 타깃: reference


class PolyUDataset(Dataset):
    def __init__(self, root_dir = r"C:\Users\Ahhyun\Desktop\Workplace\Dataset\PolyU_Patched", transform=None):
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

                gt_path = os.path.join(patch_folder_path, "groundtruth.jpg")
                noisy_path = os.path.join(patch_folder_path, "noisy.jpg")
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

class RENOIR(Dataset):
    def __init__(self, root_dir = r"C:\Users\Ahhyun\Desktop\Workplace\Dataset\RENOIR_Patched_Dataset", transform=None):
        self.root_dir = root_dir
        self.image_pairs = []        
        self.transform = transforms.Compose([        
        transforms.Resize((256, 256)),                                   
        transforms.ToTensor()         # totensor 
        ])

        # 모든 Patch 폴더 경로 수집
        for image_folder in os.listdir(root_dir):
            image_folder_path = os.path.join(root_dir, image_folder)
            if not os.path.isdir(image_folder_path):
                continue

            for patch_folder in os.listdir(image_folder_path):
                patch_folder_path = os.path.join(image_folder_path, patch_folder)
                if not os.path.isdir(patch_folder_path):
                    continue

                gt_path = os.path.join(patch_folder_path, "Reference.bmp")
                noisy_path = os.path.join(patch_folder_path, "Noisy.bmp")
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
            gt_img = self.transform(gt_img)
            noisy_img = self.transform(noisy_img)

        return noisy_img, gt_img  # 입력: noisy, 타깃: reference


class DnD(Dataset):
    def __init__(self, root_dir = r"C:\Users\Ahhyun\Desktop\Workplace\Dataset\PolyU_Patched", transform=None):
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

                gt_path = os.path.join(patch_folder_path, "groundtruth.jpg")
                noisy_path = os.path.join(patch_folder_path, "noisy.jpg")
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
    


class Custom(Dataset):
    def __init__(self, root_dir = r"C:\Users\Ahhyun\Desktop\Workplace\Dataset\Custom_Patched", transform=None):
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

                gt_path = os.path.join(patch_folder_path, "gt.bmp")
                noisy_path = os.path.join(patch_folder_path, "input.bmp")
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
    