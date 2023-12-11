import pickle
import os
import sys
import inspect
import torch
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)
import diffusers_lib
from get_model import getModel

#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'


def decode_vae_latent(vae, latents, device):
    #print("video.size()", video.size())
    # 元のビデオの形状を保存
    batch_size, num_frames, channels, height, width = latents.size()

    video_list = []

    # バッチごとにループ
    for batch_idx in range(batch_size):
        #print(batch_idx)
        batch_latents = []

        # 各フレームを個別に処理
        for frame_idx in range(num_frames):
            print(batch_idx, frame_idx)
            # フレームを取り出す
            frame_latent = latents[batch_idx, frame_idx, :, :, :].unsqueeze(0).to(device=device)
            print(frame_latent.size())
            # VAEを使用してエンコード
            frame = vae.decode(frame_latent, 1).sample

            # 処理結果をリストに追加
            batch_latents.append(frame.cpu())

            # 不要なテンソルの削除
            del frame
            del frame_latent

            # GPUメモリキャッシュのクリア
            torch.cuda.empty_cache()
        
        video_list.append(torch.stack(batch_latents, dim=1))

    # 全バッチを結合

    video = torch.cat(video_list, dim=0).to(device=device)

    return video

# 読み込むファイル名
file_name = './output_latents.pkl'
vae = getModel("vae").to(device).to(dtype=torch.float32)


# Pickleファイルから読み込み
with open(file_name, 'rb') as f:
    latents = pickle.load(f).to(device).to(dtype=torch.float32)

print(latents.size())
frames = decode_vae_latent(vae, latents, device)


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


