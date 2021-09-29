
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

# for downsampling_factor 8 (steps = 3)
radius_dict = {
    0 : 1.0, 
    1 : 1.2, 
    2 : 1.3,
    3 : 1.4, 
    4 : 2.2,
    5 : 3.75,
    6 : 4.5,
    7 : 5.25, 
    8 : 6.75,
    9 : 8.2,
    10 : 8.8,
    11 : 9.4,
    12 : 10.3,
    13 : 10.8,
    14 : 11.5,
    15 : 12.1,
    16 : 13.5,
    17 : 16., 
    18 : 17.8, 
    19 : 19.4
}

# wiener filter kappa
kappa_dict = {
    0 : 0.1, 
    1 : 0.1, 
    2 : 0.05,
    3 : 0.05, 
    4 : 0.025,
    5 : 0.025,
    6 : 0.01,
    7 : 0.007, 
    8 : 0.01,
    9 : 0.01,
    10 : 0.01,
    11 : 0.01,
    12 : 0.002,
    13 : 0.002,
    14 : 0.002,
    15 : 0.001,
    16 : 0.001,
    17 : 0.0005, 
    18 : 0.0005, 
    19 : 0.0001
}

class Downsampling(nn.Module):
    def __init__(self, steps=3):
        super(Downsampling, self).__init__()

        self.steps = steps 

        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)


    def forward(self, x):

        for i in range(self.steps):
            x = self.avg_pool(x)

        return x 

for step in range(15, 20):
#step = 2
    print("Test Bokeh for step: ", step)
    dataset = ConcatBlurredDataModule(batch_size=1, blurring_step=step)#BlurredDataModule(batch_size=8, blurring_step=step)
    dataset.prepare_data()
    dataset.setup()

    downsampling_steps = 2

    downsampler = Downsampling(steps=downsampling_steps)

    img_shape = (ceil(1460 / 2**downsampling_steps), ceil(2360 / 2**downsampling_steps))

    forward_model = BokehBlur(r=radius_dict[step]*2, shape=img_shape, kappa=0.002)

    for i, batch in tqdm(zip(range(1), dataset.test_dataloader()),
                         total=1):
        with torch.no_grad():
            gt, obs, _ = batch
            gt, obs = gt.to("cuda"), obs.to("cuda")
            
            gt = downsampler(gt)
            obs = downsampler(obs)

            y_pred = forward_model.forward(gt)

            reco = forward_model.wiener_filter(obs)

            x_0 = torch.zeros_like(gt)
            #lamb = 0.05
            #for i in range(10):
            x_0 = forward_model.filtered_grad(x_0, obs)

            fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5, sharex=True, sharey=True)

            ax1.imshow(gt.cpu().numpy()[0][0], cmap="gray")
            ax1.set_title("Groundtruth")

            ax2.imshow(obs.cpu().numpy()[0][0], cmap="gray")
            ax2.set_title("Measurements")

            ax3.imshow(y_pred.cpu().numpy()[0][0], cmap="gray")
            ax3.set_title("Simulation")

            ax4.imshow(reco.cpu().numpy()[0][0], cmap="gray")
            ax4.set_title("Reconstruction")

            ax5.imshow(x_0.cpu().numpy()[0][0], cmap="gray")
            ax5.set_title("Wiener(Ax - y)")

            plt.show()

            