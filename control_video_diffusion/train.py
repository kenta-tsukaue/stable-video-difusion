#=======[import libraries]=======
import os
import pickle
import sys

from diffusers import EulerDiscreteScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from loss import get_loss
from utils.dataset import CustomVideoDataset
from utils.config import TrainingConfig
from get_model import getModel
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)
from  diffusers_lib.pipelines.stable_video_diffusion.pipeline_control_video_diffusion import ControlVideoDiffusionPipeline
from  diffusers_lib.models.controlnet_spatio_temporal_condition import ControlNetSpatioTemporalConditionModel



#========[train]========
def train_loop(
        config, 
        unet, 
        controlnet, 
        vae, 
        image_encoder, 
        feature_extractor, 
        noise_scheduler, 
        optimizer, 
        train_dataloader, 
        lr_scheduler, device
    ):

    if config.output_dir is not None:
        os.makedirs(config.output_dir, exist_ok=True)

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch}")

        for step, (video, first_frame, last_frame, label) in enumerate(train_dataloader):
            # get loss
            loss = get_loss(
                video,
                unet, controlnet, vae, image_encoder, feature_extractor, noise_scheduler, #models
                device,
                first_frame,#最初のフレーム
                last_frame,#最後のフレーム
            )

            """loss.backward()
            optimizer.step()
            lr_scheduler.step()

            progress_bar.update(1)
"""


        
        if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
            pipeline = ControlVideoDiffusionPipeline(
                vae,
                image_encoder,
                unet,
                controlnet,
                noise_scheduler,
                feature_extractor
            )
            evaluate(config, epoch, pipeline)

        if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
            # save model
            print("save model")




#========[evaluate]========
def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
    ).images

    # save image


#========[main]========
def main():
    #set device
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # import config
    config = TrainingConfig()


    # set transform
    transform = Compose([
        Resize((config.height, config.width)),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # get dataset
    dataset = CustomVideoDataset(config, transform=transform)


    # set dataloader
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

    # import models
    unet = getModel("unet")
    controlnet = ControlNetSpatioTemporalConditionModel.from_unet(unet) # 訓練対象
    vae = getModel("vae")
    noise_scheduler = EulerDiscreteScheduler.from_pretrained("../weights/stable-video-diffusion-img2vid/scheduler", subfolder="scheduler")
    image_encoder =  CLIPVisionModelWithProjection.from_pretrained("../weights/stable-video-diffusion-img2vid/image_encoder")
    feature_extractor = CLIPImageProcessor.from_pretrained("../weights/stable-video-diffusion-img2vid/feature_extractor")

    # to eval mode　
    unet.eval()
    vae.eval()
    image_encoder.eval()

    # set optimizer
    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )

    train_loop(config, unet, controlnet, vae, image_encoder, feature_extractor, noise_scheduler, optimizer, train_dataloader, lr_scheduler, device)


if __name__ == "__main__":
    main()