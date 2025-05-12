import torch
import torchvision.transforms as transforms
from PIL import Image
from model import HybridNAFNet, NAFNet
import matplotlib.pyplot as plt

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridNAFNet(middle_blk_num=12,enc_blk_nums=[4,2,6],dec_blk_nums=[2,2,4],refinement=4) 
#model = HybridNAFNet()
#model = NAFNet()
checkpoint = torch.load(r"C:\Users\Ahhyun\Desktop\Workplace\Model\hybrid_model_36.pth", map_location=device, weights_only=True)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()


import torchsummary 
inp_shape = (3,256,256)
torchsummary.summary(model,inp_shape)

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor()
    ])
    
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0) 
    return image.to(device)

image_path = r"path/to/image"
input_tensor = preprocess_image(image_path)

original_image_path = r"path/to/image"
original_image = Image.open(original_image_path).convert("RGB")

with torch.no_grad(): 
    output = model(input_tensor) 

def postprocess_image(tensor):
    tensor = tensor.squeeze(0).cpu()
    transforms.Resize((256,256))(tensor)
    return     transforms.ToPILImage()(tensor)

# Post Process
input_image = postprocess_image(input_tensor)
output_image = postprocess_image(output)

# Visualize

plt.figure(figsize=(18,6))


plt.subplot(1, 3, 1)
plt.imshow(original_image)
plt.title("Original Image")
plt.axis("off")


plt.subplot(1, 3, 2)
plt.imshow(input_image)
plt.title("Noisy Image")
plt.axis("off")


plt.subplot(1, 3, 3)
plt.imshow(output_image)
plt.title("Transformed Image")
plt.axis("off")

plt.show()
