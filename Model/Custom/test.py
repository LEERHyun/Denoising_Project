import torch
import torchvision.transforms as transforms
from PIL import Image
from model import HybridNAFNet
import matplotlib.pyplot as plt

# 모델 정의
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridNAFNet()  # 모델 클래스 인스턴스 생성
checkpoint = torch.load(r"C:\Users\Ahhyun\Desktop\Workplace\Model\nafnet.pth", map_location=device, weights_only=True)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()


import torchsummary 
inp_shape = (3,256,256)
torchsummary.summary(model,inp_shape)

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor()
    ])
    
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # 배치 차원 추가
    return image.to(device)

image_path = r"C:\Users\Ahhyun\Desktop\Workplace\TestImage\NOISY_SRGB_010.png" # 테스트할 이미지 경로
input_tensor = preprocess_image(image_path)

original_image_path = r"C:\Users\Ahhyun\Desktop\Workplace\TestImage\GT_SRGB_010.png"
original_image = Image.open(original_image_path).convert("RGB")

with torch.no_grad():  # 기울기 계산 비활성화
    output = model(input_tensor) 

def postprocess_image(tensor):
    tensor = tensor.squeeze(0).cpu()
    transforms.Resize((256,256))(tensor)
    return     transforms.ToPILImage()(tensor)

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
