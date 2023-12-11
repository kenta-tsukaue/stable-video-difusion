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

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'


"""def decode_latents(vae, latents, num_frames, decode_chunk_size=14):
        # [batch, frames, channels, height, width] -> [batch*frames, channels, height, width]
        latents = latents.flatten(0, 1)

        latents = 1 / vae.config.scaling_factor * latents

        accepts_num_frames = "num_frames" in set(inspect.signature(vae.forward).parameters.keys())

        # decode decode_chunk_size frames at a time to avoid OOM
        frames = []
        for i in range(0, latents.shape[0], decode_chunk_size):
            print(i)
            num_frames_in = latents[i : i + decode_chunk_size].shape[0]
            decode_kwargs = {}
            if accepts_num_frames:
                # we only pass num_frames_in if it's expected
                decode_kwargs["num_frames"] = num_frames_in

            frame = vae.decode(latents[i : i + decode_chunk_size], **decode_kwargs).sample
            frames.append(frame)
            # 不要なテンソルの削除
        
        frames = torch.cat(frames, dim=0)

        # [batch*frames, channels, height, width] -> [batch, channels, frames, height, width]
        frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3, 4)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        frames = frames.float()
        return frames"""

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
            #print(batch_idx, frame_idx)
            # フレームを取り出す
            frame_latent = latents[batch_idx, frame_idx, :, :, :].unsqueeze(0).to(device=device)
            
            # VAEを使用してエンコード
            frame = vae.decode(frame_latent).sample

            # 処理結果をリストに追加
            batch_latents.append(frame.cpu())

            # 不要なテンソルの削除
            del frame
            del frame_latent

            # GPUメモリキャッシュのクリア
            torch.cuda.empty_cache()

            # 各バッチのフレームを結合
            #print("torch.stack(batch_latents, dim=1)", torch.stack(batch_latents, dim=1).size())
        
        video_list.append(torch.stack(batch_latents, dim=1))

    # 全バッチを結合

    video = torch.cat(video_list, dim=0).to(device=device)

    return video

# 読み込むファイル名
file_name = './output_latents.pkl'
vae = getModel("vae").to(device).to(dtype=torch.float16)


# Pickleファイルから読み込み
with open(file_name, 'rb') as f:
    latents = pickle.load(f).to(device).to(dtype=torch.float16)

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


