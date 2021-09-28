
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
from pathlib import Path

import torch
import yaml

import matplotlib.pyplot as plt 
from dival.measure import PSNR, SSIM
from tqdm import tqdm
import numpy as np 

import pytesseract
from fuzzywuzzy import fuzz

from skimage.transform import resize
from PIL import Image



from deblurrer.utils.blurred_dataset import BlurredDataModule, MultipleBlurredDataModule, ConcatBlurredDataModule
from deblurrer.model.GD_deblurrer_downsampling import IterativeReconstructor
from deblurrer.utils import data_util


def normalize(img):
    """
    Linear histogram normalization
    """
    arr = np.array(img, dtype=float)

    arr = (arr - arr.min()) * (255 / arr[:, :50].min())
    arr[arr > 255] = 255
    arr[arr < 0] = 0

    return Image.fromarray(arr.astype('uint8'), 'L')


def evaluateImage(img, trueText):

    # resize image to improve OCR
    w, h = img.shape
    img = resize(img, (int(w / 2), int(h / 2)))

    img = Image.fromarray(np.uint8(img*255))
    img = normalize(img)

    # run OCR
    options = r'--oem 1 --psm 6 -c load_system_dawg=false -c load_freq_dawg=false  -c textord_old_xheight=0  -c textord_min_xheight=100 -c ' \
              r'preserve_interword_spaces=0'
    OCRtext = pytesseract.image_to_string(img, config=options)

    # removes form feed character  \f
    OCRtext = OCRtext.replace('\n\f', '').replace('\n\n', '\n')

    # split lines
    OCRtext = OCRtext.split('\n')

    # remove empty lines
    OCRtext = [x.strip() for x in OCRtext if x.strip()]

    # check if OCR extracted 3 lines of text
    #print('True text (middle line): %s' % trueText[1])

    if len(OCRtext) != 3:
        print('ERROR: OCR text does not have 3 lines of text!')
        #print(OCRtext)
        return 0.0
    else:
        score = fuzz.ratio(trueText, OCRtext[1])
        #print('OCR  text (middle line): %s' % OCRtext[1])
        #print('Score: %d' % score)

        return float(score)


step = 5

dataset = ConcatBlurredDataModule(batch_size=1, blurring_step=step)
dataset.prepare_data()
dataset.setup()

num_test_images = len(dataset.test_dataloader())

identifier = "last"

base_path = "weights/default"
experiment_name = 'step_' + str(step)  
version = 'version_1'
#chkp_name ='last'# 'epoch=8-step=125' #
#path_parts = [base_path, experiment_name, 'default',
#            version, 'checkpoints', chkp_name + '.ckpt']

path_parts = [base_path, experiment_name, 'default',
            version, 'checkpoints']

chkp_path = os.path.join(*path_parts)

def search_for_file(chkp_path, identifier):
    for ckpt_file in os.listdir(chkp_path):
        if identifier in ckpt_file:
            return ckpt_file

print(search_for_file(chkp_path, identifier))
chkp_name = search_for_file(chkp_path, identifier)

chkp_path = os.path.join(chkp_path, chkp_name)

#reconstructor = IterativeReconstructor(radius=42, n_memory=2, n_iter=13, channels=[4,4, 8, 8, 16], skip_channels=[4,4,8,8,16])
reconstructor = IterativeReconstructor.load_from_checkpoint(chkp_path)
reconstructor.to("cuda")
reconstructor.net.eval()

print(reconstructor.blur.r)
print(reconstructor.net.n_iter)

psnrs = []
ssims = []
ocr_acc = []
with torch.no_grad():
    for i, batch in tqdm(zip(range(num_test_images),dataset.test_dataloader()), 
                         total=num_test_images):
        #batch = batch[0]
        gt, obs, text = batch
        obs = obs.to('cuda')
        upsample = torch.nn.Upsample(size=gt.shape[2:], mode='nearest')

        # create reconstruction from observation
        obs = reconstructor.downsampling(obs)
        reco = reconstructor.net.forward(obs)
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1,3)
        ax1.imshow(reco.cpu()[0][0], cmap="gray")
        ax1.set_title("model output")
        ax2.imshow(gt.cpu()[0][0], cmap="gray")
        ax2.set_title("gt")
        ax3.imshow(obs.cpu()[0][0], cmap="gray")
        ax3.set_title("obs (model input)")
        plt.show()
        """
        reco = upsample(reco)
        reco = reco.cpu().numpy()
        reco = np.clip(reco, 0, 1)

        # calculate quality metrics
        psnrs.append(PSNR(reco[0][0], gt.numpy()[0][0]))
        ssims.append(SSIM(reco[0][0], gt.numpy()[0][0]))
        ocr_acc.append(evaluateImage(reco[0][0], text))

mean_psnr = np.mean(psnrs)
std_psnr = np.std(psnrs)
mean_ssim = np.mean(ssims)
std_ssim = np.std(ssims)

print('---')
print('Results:')
print('mean psnr: {:f}'.format(mean_psnr))
print('std psnr: {:f}'.format(std_psnr))
print('mean ssim: {:f}'.format(mean_ssim))
print('std ssim: {:f}'.format(std_ssim))
print('mean ocr acc: ', np.mean(ocr_acc))

