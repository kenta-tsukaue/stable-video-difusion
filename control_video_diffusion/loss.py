import os
import sys
from typing import Callable, Dict, List, Optional, Union

import torch
import PIL.Image

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)


from diffusers_lib.image_processor import VaeImageProcessor
from diffusers_lib.models import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel #丁寧に型定義してもいいかもね
from diffusers_lib.schedulers import EulerDiscreteScheduler #丁寧に型定義してもいいかもね
from diffusers_lib.utils.torch_utils import randn_tensor, is_compiled_module
from diffusers_lib.pipelines.stable_video_diffusion.pipeline_control_video_diffusion import ControlVideoDiffusionPipeline
from diffusers_lib.pipelines.controlnet.multicontrolnet import MultiControlNetModel

"""
================================================
                lossを出す関数
================================================
"""
def get_loss(
    video,
    unet, controlnet, vae, image_encoder, feature_extractor, noise_scheduler, #models
    device,
    image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.FloatTensor],
    image_c: Union[PIL.Image.Image, List[PIL.Image.Image], torch.FloatTensor],
    height: int = 576,
    width: int = 1024,
    num_frames: Optional[int] = None,
    num_inference_steps: int = 25,
    min_guidance_scale: float = 1.0,
    max_guidance_scale: float = 3.0,
    fps: int = 7,
    num_images_per_prompt: Optional[int] = 1,
    motion_bucket_id: int = 127,
    noise_aug_strength: int = 0.02,
    decode_chunk_size: Optional[int] = None,
    num_videos_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    guess_mode: bool = False,
    control_guidance_start: Union[float, List[float]] = 0.0,
    control_guidance_end: Union[float, List[float]] = 1.0,
):
    
    print("train_method")

    controlnet = controlnet._orig_mod if is_compiled_module(controlnet) else controlnet


    # align format for control guidance
    if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
        control_guidance_start = len(control_guidance_end) * [control_guidance_start]
    elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
        control_guidance_end = len(control_guidance_start) * [control_guidance_end]
    elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
        mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
        control_guidance_start, control_guidance_end = (
            mult * [control_guidance_start],
            mult * [control_guidance_end],
        )
    # 0. Default height and width to unet
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    height = height or unet.config.sample_size * vae_scale_factor
    width = width or unet.config.sample_size * vae_scale_factor

    num_frames = num_frames if num_frames is not None else unet.config.num_frames
    decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames

    # 1. Set processors
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
    control_image_processor = VaeImageProcessor(
        vae_scale_factor=vae_scale_factor, do_normalize=False
    )

    # 2. Define call parameters
    if isinstance(image, PIL.Image.Image):
        batch_size = 1
    elif isinstance(image, list):
        batch_size = len(image)
    else:
        batch_size = image.shape[0]

    guess_mode = guess_mode
    do_classifier_free_guidance = max_guidance_scale > 1.0
    print(do_classifier_free_guidance)

    # 3-1. Encode input image
    image_embeddings = encode_image(image, device, num_videos_per_prompt, do_classifier_free_guidance, image_encoder, feature_extractor, image_processor)

    fps = fps - 1

    image_c = prepare_image(
            image=image_c,
            width=width,
            height=height,
            batch_size=batch_size * num_images_per_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            dtype=controlnet.dtype,
            control_image_processor=control_image_processor,
            do_classifier_free_guidance=do_classifier_free_guidance,
            guess_mode=guess_mode,
        )
    print("\nimage_c.size()",image_c.size())
    height, width = image_c.shape[-2:]

    # 4. Encode input video using VAE

    needs_upcasting = vae.dtype == torch.float16 and vae.config.force_upcast
    if needs_upcasting:
        vae.to(dtype=torch.float32)

    video_latents = encode_vae_video(vae, video, device)
    video_latents = image_latents.to(image_embeddings.dtype)


    print("\nvideo_latents.size()", video_latents.size())

    # cast back to fp16 if needed
    if needs_upcasting:
        vae.to(dtype=torch.float16)

    # Repeat the image latents for each frame so we can concatenate them with the noise
    # image_latents [batch, channels, height, width] ->[batch, num_frames, channels, height, width]
    image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)

    # 5. Get Added Time IDs
    added_time_ids = get_add_time_ids(
        unet,
        fps,
        motion_bucket_id,
        noise_aug_strength,
        image_embeddings.dtype,
        batch_size,
        num_videos_per_prompt,
        do_classifier_free_guidance,
    )
    added_time_ids = added_time_ids.to(device)

    # 6. Prepare timesteps
    noise_scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = noise_scheduler.timesteps

    # 7. Prepare latent variables
    num_channels_latents = unet.config.in_channels
    latents = prepare_latents(
        batch_size * num_videos_per_prompt,
        num_frames,
        num_channels_latents,
        height,
        width,
        image_embeddings.dtype,
        device,
        generator,
        vae_scale_factor,
        latents,
    )
    # 8. Prepare guidance scale
    guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
    guidance_scale = guidance_scale.to(device, latents.dtype)
    guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
    guidance_scale = _append_dims(guidance_scale, latents.ndim)

    return 0


def _append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]

def encode_image(image, device, num_videos_per_prompt, do_classifier_free_guidance, image_encoder, feature_extractor, image_processor):
    dtype = next(image_encoder.parameters()).dtype
    print("\nimage_size()", image.size())
    """if not isinstance(image, torch.Tensor):
        image = image_processor.pil_to_numpy(image)
        image = image_processor.numpy_to_pt(image)

        # We normalize the image before resizing to match with the original implementation.
        # Then we unnormalize it after resizing.
        image = image * 2.0 - 1.0
        image = _resize_with_antialiasing(image, (224, 224))
        image = (image + 1.0) / 2.0

        # Normalize the image with for CLIP input
        image = feature_extractor(
            images=image,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        ).pixel_values"""
    image = _resize_with_antialiasing(image, (224, 224))
    image = (image + 1.0) / 2.0
    image = feature_extractor(
            images=image,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        ).pixel_values
    
    print("\n image_size()", image.size())
    image = image.to(device=device, dtype=dtype)
    print("\nimage[0].size()", image[0].size())
    image_embeddings = image_encoder(image).image_embeds
    print("\n image_embeddings", image_embeddings.size())
    image_embeddings = image_embeddings.unsqueeze(1)
    print("\n image_embeddings", image_embeddings.size())
    # duplicate image embeddings for each generation per prompt, using mps friendly method
    bs_embed, seq_len, _ = image_embeddings.shape
    image_embeddings = image_embeddings.repeat(1, num_videos_per_prompt, 1)
    image_embeddings = image_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

    if do_classifier_free_guidance:
        negative_image_embeddings = torch.zeros_like(image_embeddings)

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        image_embeddings = torch.cat([negative_image_embeddings, image_embeddings])

    return image_embeddings


def prepare_image(
    image,
    width,
    height,
    batch_size,
    num_images_per_prompt,
    device,
    dtype,
    control_image_processor,
    do_classifier_free_guidance=False,
    guess_mode=False,   
):
    image = control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
    image_batch_size = image.shape[0]

    if image_batch_size == 1:
        repeat_by = batch_size
    else:
        # image batch size is the same as prompt batch size
        repeat_by = num_images_per_prompt

    image = image.repeat_interleave(repeat_by, dim=0)

    image = image.to(device=device, dtype=dtype)

    if do_classifier_free_guidance and not guess_mode:
        image = torch.cat([image] * 2)

    return image

"""def encode_vae_video(vae, video: torch.Tensor, device):
    torch.cuda.empty_cache()
    print("video.size()",video.size())
    # 元のビデオの形状を保存
    batch_size, num_frames, channels, height, width = video.size()

    # ビデオをフレームごとのバッチに変形（バッチサイズ＊フレーム数、チャネル、縦、横）
    video_flattened = video.view(batch_size * num_frames, channels, height, width)

    # デバイスに送る
    video_flattened = video_flattened.to(device=device)
    print("video_flattened.size()",video_flattened.size())
    # VAEを使用してエンコード
    video_latents = vae.encode(video_flattened).latent_dist.mode()
    print("video_latents.size()", video_latents.size())

    # テンソルを元のビデオ形状に戻す（バッチサイズ、フレーム数、チャネル、縦、横）
    video_latents = video_latents.view(batch_size, num_frames, channels, height, width)

    return video_latents"""
def encode_vae_video(vae, video: torch.Tensor, device):
    torch.cuda.empty_cache()
    print("video.size()", video.size())

    # 元のビデオの形状を保存
    batch_size, num_frames, channels, height, width = video.size()

    # 出力用のリストを初期化
    video_latents_list = []

    # 各フレームを個別に処理
    for frame_idx in range(num_frames):
        print(frame_idx)
        # フレームを取り出す
        frame = video[:, frame_idx, :, :, :].to(device=device)
        
        print("frame.size()", frame.size())

        # VAEを使用してエンコード
        frame_latent = vae.encode(frame).latent_dist.mode()
        print("frame_latent.size()", frame_latent.size())
        
        # 処理結果をリストに追加
        video_latents_list.append(frame_latent)

        # 中間テンソルの削除
        del frame
        del frame_latent

        # GPUメモリキャッシュのクリア
        torch.cuda.empty_cache()

    # 処理結果を結合
    video_latents = torch.stack(video_latents_list, dim=1)
    print(" video_latents.size()",  video_latents.size())


    # 結果を元のビデオ形状に戻す
    video_latents = video_latents.view(batch_size, num_frames, channels, height, width)
    print(" video_latents.size()",  video_latents.size())


    return video_latents



def get_add_time_ids(
    unet,
    fps,
    motion_bucket_id,
    noise_aug_strength,
    dtype,
    batch_size,
    num_videos_per_prompt,
    do_classifier_free_guidance,
):
    add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

    passed_add_embed_dim = unet.config.addition_time_embed_dim * len(add_time_ids)
    expected_add_embed_dim = unet.add_embedding.linear_1.in_features

    if expected_add_embed_dim != passed_add_embed_dim:
        raise ValueError(
            f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
        )

    add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
    add_time_ids = add_time_ids.repeat(batch_size * num_videos_per_prompt, 1)

    if do_classifier_free_guidance:
        add_time_ids = torch.cat([add_time_ids, add_time_ids])

    return add_time_ids

def prepare_latents(
        self,
        batch_size,
        num_frames,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        vae_scale_factor,
        latents=None,
    ):
        shape = (
            batch_size,
            num_frames,
            num_channels_latents // 2,
            height // vae_scale_factor,
            width // vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents
# resizing utils
# TODO: clean up later
def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])

    # First, we have to determine sigma
    # Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )

    # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
    # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
    # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    # Make sure it is odd
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]

    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)

    output = torch.nn.functional.interpolate(input, size=size, mode=interpolation, align_corners=align_corners)
    return output


def _compute_padding(kernel_size):
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding


def _filter2d(input, kernel):
    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    padding_shape: list[int] = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = torch.nn.functional.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    out = output.view(b, c, h, w)
    return out


def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])

    batch_size = sigma.shape[0]

    x = (torch.arange(window_size, device=sigma.device, dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)


def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)

    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    out = _filter2d(out_x, kernel_y[..., None])

    return out