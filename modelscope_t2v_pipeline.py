import os
from typing import Any, Dict

import imageio
import torch
from einops import rearrange
from modelscope.metainfo import Pipelines
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.config import Config
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

from modelscope_t2v import TextToVideoSynthesis

logger = get_logger()


@PIPELINES.register_module(Tasks.text_to_video_synthesis, module_name=Pipelines.text_to_video_synthesis)
class TextToVideoSynthesisPipeline(Pipeline):
    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a kws pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        cfg = Config.from_file(os.path.join(model, "configuration.json"))
        if "retrained_path" not in kwargs.keys():  # True
            kwargs["retrained_path"] = None
        if "finetune_autoenc" not in kwargs.keys():  # True
            kwargs["finetune_autoenc"] = None
        model_cfg = cfg.model
        args = {
            "model_dir": model,
            "model_args": model_cfg["model_args"],
            "model_cfg": model_cfg["model_cfg"],
            "device": kwargs["device"],
            "retrained_path": kwargs["retrained_path"],
            "finetune_autoenc": kwargs["finetune_autoenc"],
        }
        self.model = TextToVideoSynthesis(**args)

    def preprocess(self, input: Input, **preprocess_params) -> Dict[str, Any]:
        self.model.clip_encoder.to(self.model.device)
        text_emb = self.model.clip_encoder(input)
        text_emb_zero = self.model.clip_encoder("")
        bs = len(input)
        text_emb_zero = text_emb_zero.repeat(bs, 1, 1)
        if self.model.config.model.model_args.tiny_gpu == 1:
            self.model.clip_encoder.to("cpu")
        return {"text_emb": text_emb, "text_emb_zero": text_emb_zero}

    def preprocess_vid(self, vid):
        vid_tensor = torch.from_numpy(vid / 255.0).type(torch.float32)
        vid_tensor = vid_tensor.unsqueeze(dim=0)
        vid_tensor = vid_tensor.permute(0, 4, 1, 2, 3)  # ncfhw
        # normalization
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        mean = torch.tensor(mean, device=vid_tensor.device).reshape(1, -1, 1, 1, 1)  # ncfhw
        std = torch.tensor(std, device=vid_tensor.device).reshape(1, -1, 1, 1, 1)  # ncfhw
        vid_tensor = (vid_tensor - mean) / std
        return vid_tensor

    def forward(self, input: Dict[str, Any], **forward_params) -> Dict[str, Any]:
        video = self.model(input)
        return video

    def forward_with_vid_resample(self, input, vid, image_pos, add_vid_cond, use_ddpm_inversion, use_ddim_inversion,
                                  resample_iter, percentage, delta_p, r_step, delta_s, ddim_step=50, guide_scale=9.0):
        video = self.model.forward_with_vid_resample(input, vid, image_pos, add_vid_cond, use_ddpm_inversion, use_ddim_inversion, 
                                                     resample_iter, percentage, delta_p, r_step, delta_s, ddim_step, guide_scale)
        return video

    def postprocess(self, inputs, file_path, fps=8):
        video = tensor2vid(inputs)
        if file_path.endswith(".gif"):
            imageio.mimwrite(file_path, video, duration=1000 / fps)
        if file_path.endswith(".mp4"):
            imageio.mimwrite(file_path, video, fps=fps)
        return video


def tensor2vid(video, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    mean = torch.tensor(mean, device=video.device).reshape(1, -1, 1, 1, 1)  # ncfhw
    std = torch.tensor(std, device=video.device).reshape(1, -1, 1, 1, 1)  # ncfhw
    video = torch.clamp(video * std + mean, min=0, max=1)
    images = rearrange(video, "i c f h w -> f h (i w) c")
    images = images.unbind(dim=0)
    images = [(image.numpy() * 255).astype("uint8") for image in images]  # f h w c
    return images
