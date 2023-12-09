import json
import sys
import os

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)
from  diffusers_lib.pipelines.stable_video_diffusion.pipeline_control_video_diffusion import ControlVideoDiffusionPipeline

class ControlVideoDiffusionPipelineForTrain(ControlVideoDiffusionPipeline):
    @classmethod
    def train(cls):
        print("train_method")