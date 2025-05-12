import torch
import torchvision.transforms as transforms
from torch import nn, optim
from PIL import Image
import matplotlib.pyplot as plt

# Load Model

class CBDNet(nn.Module):
    def __init__(self):
        super(CBDNet, self).__init__()
        
        #Noise Estimation Subnetwork
        self.NES = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(32,32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(32,32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(32,32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(32,3,kernel_size=3,padding=1),
            nn.ReLU(),
            )
        
        #Non Blind Denosing subnetwork(U-Net)
        #Conv1
        self.NBDS_1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2,2)
            )
        #Conv2
        self.NBDS_2 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(128,128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(128,128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2,2)
            )
        #Conv3
        self.NBDS_3 = nn.Sequential(
            nn.Conv2d(128,256,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.ReLU()
            )
        
        #Upsample1
        self.upsample1 = nn.ConvTranspose2d(256,128,kernel_size=2,stride=2)
        
        self.NBDS_4 = nn.Sequential(
            nn.Conv2d(128,128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(128,128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(128,128,kernel_size=3,padding=1),
            nn.ReLU()
            )       
        
        #Upsample2
        self.upsample2 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        
        self.NBDS_5 = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.ReLU()
            )     
        self.OUT = nn.Conv2d(64,3,kernel_size=1)
        
    def forward(self, x):
        x = self.NES(x)
        x = self.NBDS_1(x)
        x = self.NBDS_2(x)
        x = self.NBDS_3(x)  
        x = self.upsample1(x)
        x = self.NBDS_4(x) 
        x = self.upsample2(x) 
        x = self.NBDS_5(x)    
        x = self.OUT(x) 
        return x
    
#Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CBDNet()
model_path = r"C:\Users\Ahhyun\Desktop\Workplace\Model\denoising_CBDNet_Epoch30.pth"  
model.load_state_dict(torch.load(model_path,map_location=device))
model.to(device)
model.eval()  


import torchsummary


torchsummary.summary(model,(3,256,256))
# Pre-Processing
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  
        transforms.ToTensor(),  
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 정규화
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).to("cuda")  # 배치 차원을 추가

image_path = r"C:\Users\Ahhyun\Downloads\NOISY_SRGB_010.png" # 테스트할 이미지 경로
input_tensor = preprocess_image(image_path)

original_image_path = r"C:\Users\Ahhyun\Downloads\GT_SRGB_010.png"
original_image = Image.open(original_image_path).convert("RGB")

# 3. 모델 예측
with torch.no_grad():  # 기울기 계산 비활성화
    output = model(input_tensor) 

#출력 이미지 후처리 함수
def postprocess_image(tensor):
    tensor = tensor.squeeze(0).cpu()  # 배치 차원 제거 및 CPU로 이동
    tensor = torch.clamp(tensor, -1, 1)  # 정규화 범위를 [-1, 1]로 클램프
    tensor = (tensor + 1) / 2  # -1~1 범위를 0~1로 변환
    return transforms.ToPILImage()(tensor)

# 이미지 후처리
input_image = postprocess_image(input_tensor)
output_image = postprocess_image(output)

# 5. 결과 시각화

plt.figure(figsize=(18,6))

# 원본 이미지
plt.subplot(1, 3, 1)
plt.imshow(original_image)
plt.title("Original Image")
plt.axis("off")

# Input Noisy Image
plt.subplot(1, 3, 2)
plt.imshow(input_image)
plt.title("Noisy Image")
plt.axis("off")

# Denoised Image
plt.subplot(1, 3, 3)
plt.imshow(output_image)
plt.title("Transformed Image")
plt.axis("off")

plt.show()