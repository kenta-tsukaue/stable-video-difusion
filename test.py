from diffusers import DiffusionPipeline
import torch
from PIL import Image
from torchvision import transforms

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid")
pipe.enable_attention_slicing()
image_path="./test.jpg"

# PILを使って画像を読み込む
image = Image.open(image_path)

# 画像をテンソルに変換するための変換を定義
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 画像を224x224にリサイズ
    transforms.ToTensor(),  # PIL画像をTensorに変換
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 正規化
])

# 画像を変換
tensor_image = transform(image)

# バッチ次元を追加
tensor_image = tensor_image.unsqueeze(0)

print(tensor_image.size())
output = pipe(tensor_image).images[0]
print(output)