import os
from copy import deepcopy

import imageio
import numpy as np
import torch
import time
from PIL import Image

from modelscope_t2v_pipeline import TextToVideoSynthesisPipeline, tensor2vid
from util import center_crop, set_seed

print(torch.cuda.is_available())
print("Num GPUs available: ", torch.cuda.device_count())


set_seed(42)

# Choose your GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Text input examples
inputs = [
    "A panda is dancing in the Times Square.",
    "A person is surfing on a wave.",
    "People are participating in a horse race.",
    "A person is riding a horse.",
    "A baby is crawling.",
    "A person is spinning a hula hoop.",
    "A person is doing pull-ups.",
    "A person is punching a boxing bag."
]

# Image input examples
img_paths = [
    "./examples/panda_dancing_sd.png",
    "./examples/v_Surfing_g07_c03-000.png",
    "./examples/v_HorseRace_g11_c01-000.png",
    "./examples/v_HorseRiding_g11_c01-000.png",
    "./examples/v_BabyCrawling_g03_c04-000.png",
    "./examples/v_HulaHoop_g19_c02-000.png",
    "./examples/v_PullUps_g06_c02-000.png",
    "./examples/v_BoxingPunchingBag_g05_c05-000.png"
]

assert len(inputs) == len(img_paths)

# ----------------------------------- PARAMETER SETTINGS ------------------------------------
# After running initialization.py, set the config path to your ModelScope path
config = {"model": "./weights", "device": "gpu"}

# Set your output path
output_dir = "./example-video/results-replace-patch-flex-pos"
output_img_dir = "./example-image/results-replace-patch-flex-pos"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_img_dir, exist_ok=True)

# Default parameters
IMG_H = 256  # condition image size
IMG_W = 256  # condition image size
NUM_FRAMES = 16  # output video size
NUM_COND_FRAMES = 1
NUM_NEW_FRAMES = 15  # the number of new frames
NUM_SAMPLES = 1  # the number of generated videos

ddim_step = 20
resample_iter = 0  # Set parameters for temporal resampling and DDIM
add_vid_cond = True
use_ddpm_inversion = True  # if True, use the inversion of condition video frames as initialization; if False, initialize with random noise.
use_ddim_inversion = False  # if True, use DDIM inversion
image_pos = 0
percentage = 0.3
delta_p = 0.005
r_step = 10
delta_s = 0.3

mark = f"-P{percentage:.2f}-delta{delta_p:.3f}-step{r_step:02d}-delta{delta_s:.2f}"
postfix = "-pos%02d-rsmpl%02d-s%02d-mean%04d" % (image_pos, resample_iter, ddim_step, np.random.randint(low=0, high=10000)) + mark

print(input, postfix)
print("#new_frame:", NUM_NEW_FRAMES)
print("video_cond:", add_vid_cond, "ddpm_inv:", use_ddpm_inversion, "#resample:", resample_iter)
# -------------------------------------------------------------------------------------------

print("Loading weights...")
t2v_pipeline = TextToVideoSynthesisPipeline(**config)

start_time = time.time()

for input, img_path in zip(inputs, img_paths):
    # read image
    first_img_npy = imageio.v2.imread(img_path)
    # crop image
    first_img_npy = center_crop(first_img_npy)
    # resize image
    first_img_npy = np.asarray(Image.fromarray(first_img_npy).resize((IMG_H, IMG_W)))

    # repeat image
    first_img_npy_list = [first_img_npy for i in range(NUM_COND_FRAMES)]
    cond_vid_npy = np.stack(first_img_npy_list, axis=0)  # (1, 256, 256, 3), 0~255
    processed_input = t2v_pipeline.preprocess([input])  # {"text_emb": (1, 77, 1024), "text_emb_zero": (1, 77, 1024)}

    for sample_idx in range(NUM_SAMPLES):
        vid_tensor = t2v_pipeline.preprocess_vid(deepcopy(cond_vid_npy))  # (1, 3, 15, 256, 256), -1~1
        new_output_tensor = vid_tensor.clone().detach().cpu()
        
        newpostfix = postfix + "-%02d" % sample_idx
        output_filename = input.replace(" ", "_")[:-1] + "%s-%02d" % (newpostfix, NUM_NEW_FRAMES) + ".mp4"
        video_name = os.path.basename(output_filename)[:-4]
        save_img_dir = os.path.join(output_img_dir, video_name)
        os.makedirs(save_img_dir, exist_ok=True)
        img_name = video_name + "%03d.jpg" % 0
        img_path = os.path.join(save_img_dir, img_name)
        imageio.v2.imsave(img_path, first_img_npy)

        output = t2v_pipeline.forward_with_vid_resample(  # (1, 4, 16, 32, 32)
            processed_input,
            vid=vid_tensor,
            image_pos=[image_pos],
            add_vid_cond=add_vid_cond,
            use_ddpm_inversion=use_ddpm_inversion,
            use_ddim_inversion=use_ddim_inversion,
            resample_iter=resample_iter,
            percentage=percentage,
            delta_p=delta_p,
            r_step=r_step,
            delta_s=delta_s,
            ddim_step=ddim_step,
            guide_scale=9.0
        )

        output_frames = list()
        for i in range(output.shape[2]):
            with torch.no_grad():
                new_frame = t2v_pipeline.model.autoencoder.decode(output[:, :, i].cuda())  # (1, 3, 1, 256, 256)
            new_frame = new_frame.data.cpu().unsqueeze(dim=2)  # (1, 3, 1, 256, 256)
            img_npy = tensor2vid(new_frame.clone().detach())[0]
            img_name = video_name + "%03d.jpg" % (i + 1)
            img_path = os.path.join(save_img_dir, img_name)
            imageio.v2.imsave(img_path, img_npy)
            output_frames.append(new_frame)

        assert len(output_frames) == NUM_FRAMES
        output_frame_stack = torch.cat(output_frames, dim=2)
        output_video = t2v_pipeline.postprocess(output_frame_stack, os.path.join(output_dir, output_filename))
        print("saving to", save_img_dir)
        print("saving video to", os.path.join(output_dir, output_filename))

print('Process time:', time.time() - start_time)
