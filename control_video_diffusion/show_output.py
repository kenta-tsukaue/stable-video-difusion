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


def decode_latents(vae, latents, num_frames, decode_chunk_size=14):
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
            del frame
            # GPUメモリキャッシュのクリア
            torch.cuda.empty_cache()
        
        frames = torch.cat(frames, dim=0)

        # [batch*frames, channels, height, width] -> [batch, channels, frames, height, width]
        frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3, 4)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        frames = frames.float()
        return frames


# 読み込むファイル名
file_name = './output_latents.pkl'
vae = getModel("vae")

vae.to(dtype=torch.float16).to(device)


# Pickleファイルから読み込み
with open(file_name, 'rb') as f:
    latents = pickle.load(f).to(device)


frames = decode_latents(vae, latents, 14, 14)


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


