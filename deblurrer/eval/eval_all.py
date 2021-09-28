

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
from pathlib import Path
from tqdm import tqdm

import torch 

import torchvision 
import torchvision.transforms as transforms

import matplotlib.pyplot as plt 

from PIL import Image
import numpy as np 
from skimage.transform import resize

from deblurrer.utils.blurred_dataset import BlurredDataModule, MultipleBlurredDataModule, ConcatBlurredDataModule
from deblurrer.model.GD_deblurrer_version2 import IterativeReconstructor
from deblurrer.utils import data_util


from dival.util.plot import plot_images



for step in range(20):
    print("Deblurring for step ", step)


    dataset = BlurredDataModule(batch_size=1, blurring_step=step)
    dataset.prepare_data()
    dataset.setup()

    base_path = "/localdata/AlexanderDenker/deblurring_experiments"
    experiment_name = 'step_' + str(step)  
    version = 'version_0'

    identifier = "last"


    path_parts = [base_path, 'run_29', experiment_name, 'default',
                version, 'checkpoints']
    chkp_path = os.path.join(*path_parts)
    
    if not os.path.exists(chkp_path):
        print("File not found: ", chkp_path)
        continue

    def search_for_file(chkp_path, identifier):
        for ckpt_file in os.listdir(chkp_path):
            if identifier in ckpt_file:
                return ckpt_file

    print(search_for_file(chkp_path, identifier))
    chkp_name = search_for_file(chkp_path, identifier)
    if chkp_name is None:
        print("No checkpint found...")
        continue
    chkp_path = os.path.join(chkp_path, chkp_name)
    print("Load from ",chkp_path)
    #reconstructor = IterativeReconstructor(radius=42, n_memory=2, n_iter=13, channels=[4,4, 8, 8, 16], skip_channels=[4,4,8,8,16])
    reconstructor = IterativeReconstructor.load_from_checkpoint(chkp_path)
    reconstructor.net.eval()
    #reconstructor.to("cuda")

    report_name = "report_step_" + str(step) + "_model_" + identifier
    report_path = path_parts[:-1]
    report_path.append(report_name)
    report_path = os.path.join(*report_path)
    Path(report_path).mkdir(parents=True, exist_ok=True)
    
    for i, batch in tqdm(zip(range(3), dataset.test_dataloader()),
                         total=3):
        with torch.no_grad():
            gt, obs, _ = batch
            
            #obs = obs.to('cuda')
            upsample = torch.nn.Upsample(size=gt.shape[2:], mode='bilinear') # 'nearest'

            # create reconstruction from observation
            obs_down = reconstructor.downsampling(obs)
            reco = reconstructor.forward(obs_down)
            reco = upsample(reco)
            reco = reco.cpu().numpy()
            reco = np.clip(reco, 0, 1)

        img_save_path = os.path.join(report_path,'img')
        Path(img_save_path).mkdir(parents=True, exist_ok=True)


        im = Image.fromarray(reco[0][0]*255.).convert("L")
        im.save(os.path.join(img_save_path, "test_sample_{}.PNG".format(i)))
       
        gt = gt.cpu().numpy()[0][0]

        _, ax = plot_images([reco[0,0,:,:].T, gt.T, obs.cpu().numpy()[0,0,:,:].T],
                            fig_size=(10, 4), vrange='equal', cbar='auto')

        ax[0].set_title('Reconstruction')
        ax[1].set_title('Ground truth')
        ax[2].set_title('Blurred Image')
        ax[0].axis('off')
        ax[1].axis('off')
        ax[2].axis('off')

        ax[0].figure.suptitle('test sample {:d}'.format(i))
        

        img_save_path = os.path.join(img_save_path, 'test sample {:d}'.format(i)+'.png')
        plt.savefig(img_save_path, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', format=None, transparent=False,
                bbox_inches=None, pad_inches=0.1, metadata=None)
