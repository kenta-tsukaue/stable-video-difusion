#=======[import libraries]=======
import os
import pickle
import sys
from get_model import getModel
from diffusers import EulerDiscreteScheduler
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

# プロジェクトのルートディレクトリへのパスを直接指定
#sys.path.append('C:/Users/Public/Documents/プログラミング/stable_video/')
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)
from  diffusers_lib.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import StableVideoDiffusionPipeline

#========[import models]========
unet = getModel("unet")
vae = getModel("vae")
scheduler = EulerDiscreteScheduler.from_pretrained("../weights/stable-video-diffusion-img2vid/scheduler", subfolder="scheduler")
image_encoder =  CLIPVisionModelWithProjection.from_pretrained("../weights/stable-video-diffusion-img2vid/image_encoder")
feature_extractor = CLIPImageProcessor.from_pretrained("../weights/stable-video-diffusion-img2vid/feature_extractor")
pipe = StableVideoDiffusionPipeline(
    vae,
    image_encoder,
    unet,
    scheduler,
    feature_extractor
)


#========[predict]========
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
output = pipe(tensor_image)

# 保存するファイル名
file_name = 'output.pkl'

# Pickleファイルとして保存
with open(file_name, 'wb') as f:
    pickle.dump(output, f)

print(f"File saved as {file_name}")
    