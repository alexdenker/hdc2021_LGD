
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import pytorch_lightning as pl
from math import ceil
from tqdm import tqdm 

import torch 
import torch.nn as nn
import matplotlib.pyplot as plt

from deblurrer.utils.blurred_dataset import BlurredDataModule, MultipleBlurredDataModule, ConcatBlurredDataModule
from deblurrer.model.GD_deblurrer_version2 import IterativeReconstructor
from deblurrer.model.bokeh_blur_rfft_version2 import BokehBlur

from PIL import Image
from pathlib import Path

import numpy as np 

for step in range(14, 20):
#step = 2
    print("Test Bokeh for step: ", step)
    dataset = ConcatBlurredDataModule(batch_size=1, blurring_step=step)#BlurredDataModule(batch_size=8, blurring_step=step)
    dataset.prepare_data()
    dataset.setup()



    for i, batch in tqdm(zip(range(1), dataset.test_dataloader()),
                         total=1):
        with torch.no_grad():
            for r in np.linspace(30, 200, 200):
                forward_model = BokehBlur(r=r, kappa=0.002)
                gt, obs, _ = batch

                reco = forward_model.wiener_filter(obs)

                img_save_path = os.path.join('step_{}_wiener_img'.format(step))
                Path(img_save_path).mkdir(parents=True, exist_ok=True)

                im = Image.fromarray(reco.numpy()[0][0]*255.).convert("L")
                im.save(os.path.join(img_save_path, "wiener_filter_r={}.PNG".format(r)))

       