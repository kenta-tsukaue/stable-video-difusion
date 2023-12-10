from dataclasses import dataclass

@dataclass
class TrainingConfig:
    height = 576  # the generated image resolution
    width = 1024
    train_batch_size = 2
    eval_batch_size = 2  # how many images to sample during evaluation
    data_path = "../dataset/action_youtube_naudio" #train.pyからの位置
    num_epochs = 10
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "ddpm-butterflies-128"  # the model name locally and on the HF Hub
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0

