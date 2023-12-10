import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_video
from torchvision.transforms import Compose, Resize, Normalize, ToTensor


class CustomVideoDataset(Dataset):
    def __init__(self, config, transform=None):
        self.directory = config.data_path
        self.transform = transform
        self.videos = self._load_video_paths()

    def _load_video_paths(self):
        # フォルダ内のすべての動画ファイルのパスを取得します。
        video_paths = []
        for category in os.listdir(self.directory):
            category_path = os.path.join(self.directory, category)
            for video_folder in os.listdir(category_path):
                #print(video_folder)
                video_folder_path = os.path.join(category_path, video_folder)
                if video_folder == "Annotation":
                    continue
                for video in os.listdir(video_folder_path):
                    if video.endswith('.avi'):
                        video_paths.append(os.path.join(video_folder_path, video))
        #print(video_paths[0])
        return video_paths

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_path = self.videos[idx]
        video, _, _ = read_video(video_path)  # videoはtorch.uint8型のTensor
        video = video.float() / 255.0  # スケーリング

        # 中央の56フレームを取得
        center_frame = video.shape[0] // 2
        start_frame = max(center_frame - 28, 0)
        end_frame = min(center_frame + 28, video.shape[0])

        video = video[start_frame:end_frame]

        # 56フレームから等間隔で14フレームを選択
        interval = len(video) // 14
        video = video[::interval][:14]

        # チャネルの順番を変更
        video = video.permute(0, 3, 1, 2)

        if self.transform:
            video = torch.stack([self.transform(frame) for frame in video])

        # 最初と最後のフレームを取得
        first_frame = video[0]
        last_frame = video[-1]
        # ラベル（動画のファイル名）
        label = os.path.splitext(os.path.basename(video_path))[0]

        return video, first_frame, last_frame, label