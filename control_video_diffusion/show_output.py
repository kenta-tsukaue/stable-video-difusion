import pickle
import os
import sys

import torch
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)
import diffusers_lib
from get_model import getModel

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 読み込むファイル名
file_name = './output_latents.pkl'
vae = getModel("vae")

vae.to(device).to(dtype=torch.float16)


# Pickleファイルから読み込み
with open(file_name, 'rb') as f:
    latents = pickle.load(f)


frames = vae.decode_latents(latents, 14, 14)


# 画像のリストを取得
images = frames.frames[0]

# GIFとして保存するファイル名
gif_filename = 'output.gif'

# GIFを作成
images[0].save(
    gif_filename,
    save_all=True,
    append_images=images[1:],
    duration=100,  # 各フレームの表示時間（ミリ秒）
    loop=0  # ループ回数（0は無限ループ）
)

print(f"GIF saved as {gif_filename}")