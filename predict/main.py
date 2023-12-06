#=======[import libraries]=======
import torch
from model import getModel
from diffusers import EulerDiscreteScheduler
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

#========[import models]========
unet = getModel("unet")
vae = getModel("vae")
scheduler = EulerDiscreteScheduler.from_pretrained("./weights/stable-video-diffusion-img2vid/scheduler", subfolder="scheduler")
image_encoder =  CLIPVisionModelWithProjection.from_pretrained("./weights/stable-video-diffusion-img2vid/image_encoder")
feature_extractor = CLIPImageProcessor.from_pretrained("./weights/stable-video-diffusion-img2vid/feature_extractor")

#========[predict]========
torch_device = "cpu"
vae.to(torch_device)
image_encoder.to(torch_device)
unet.to(torch_device)

image_path="./test.jpg"

# open img
image = Image.open(image_path)

# set to tensor
transform = transforms.Compose([
    transforms.ToTensor(), 
])

# convert pil to tensor
tensor_image = transform(image)

# add batch dim
tensor_image = tensor_image.unsqueeze(0)

# stable setting
num_frames = 25
height = 512  # default height of Stable Diffusion
width = 512  # default width of Stable Diffusion
guidance_scale = 7.5  # Scale for classifier-free guidance
generator = torch.manual_seed(0)  # Seed generator to create the initial latent noise
batch_size = len(tensor_image)


with torch.no_grad():
    # get features
    features = feature_extractor(tensor_image, return_tensors="pt", do_rescale=False)
    features = features.to(torch_device)  # デバイスに移動
    # CLIPImageProcessor to tensor
    pixel_values = features['pixel_values'].to(torch_device)

    # encode image
    image_embeddings = image_encoder(pixel_values)["image_embeds"]
    encoder_hidden_states = image_embeddings.unsqueeze(1)
    print(encoder_hidden_states.size())


# get latent
latents = torch.randn(
    (batch_size, num_frames, unet.config.out_channels, height // 8, width // 8),
    generator=generator,
    device=torch_device,
)
print(latents.size())
batch_size = tensor_image.size(0)  # バッチサイズ
added_time_ids = torch.arange(num_frames, device=torch_device).repeat(batch_size, 1)

for t in tqdm(scheduler.timesteps):
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)

    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

    # predict the noise residual
    with torch.no_grad():
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=encoder_hidden_states, added_time_ids=added_time_ids).sample

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # compute the previous noisy sample x_t -> x_t-1
    latents = scheduler.step(noise_pred, t, latents).prev_sample
    