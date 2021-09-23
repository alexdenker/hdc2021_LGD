
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from pathlib import Path

import torch
import yaml
from dival.util.plot import plot_images

import matplotlib.pyplot as plt 
from dival.measure import PSNR, SSIM
from tqdm import tqdm
import numpy as np 

import pytesseract
from fuzzywuzzy import fuzz

from skimage.transform import resize
from PIL import Image



from deblurrer.utils.blurred_dataset import BlurredDataModule, MultipleBlurredDataModule
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
        score = fuzz.ratio(trueText[1], OCRtext[1])
        #print('OCR  text (middle line): %s' % OCRtext[1])
        #print('Score: %d' % score)

        return float(score)


for step in range(5, 20):
    print("Eval OCR for step ", step)
    print("--------------------------------\n")
    save_report = True 
    plot_examples = False 

    X_times, Y_times, text_times = data_util.load_data_with_text('Times', step)
    X_verdana, Y_verdana, text_verdana = data_util.load_data_with_text('Verdana', step)

    X = np.concatenate([X_times, X_verdana], axis=0)
    Y = np.concatenate([Y_times, Y_verdana], axis=0)
    text = text_times + text_verdana

    X = np.expand_dims(X, axis=1).astype(np.float32)
    Y = np.expand_dims(Y, axis=1).astype(np.float32)

    X = torch.from_numpy(X)/65535.
    Y = torch.from_numpy(Y)/65535.

    num_test_images = len(text)

    base_path = "/localdata/AlexanderDenker/deblurring_experiments/no_pm_no_sigmoid"
    experiment_name = 'step_' + str(step)  
    version = 'version_0'
    chkp_name = 'learned_gradient_descent'
    path_parts = [base_path, experiment_name, 'default',
                version, 'checkpoints', chkp_name + '.ckpt']
    
    chkp_path = os.path.join(*path_parts)
    if not os.path.exists(chkp_path):
        print("File not found: ", chkp_path)
        continue
    #reconstructor = IterativeReconstructor(radius=42, n_memory=2, n_iter=13, channels=[4,4, 8, 8, 16], skip_channels=[4,4,8,8,16])
    reconstructor = IterativeReconstructor.load_from_checkpoint(chkp_path)
    reconstructor.to("cuda")


    if save_report:
        report_name = version + '_' + chkp_name + '_images=' + str(num_test_images) + "_ocr"
        report_path = path_parts[:-2]
        report_path.append(report_name)
        report_path = os.path.join(*report_path)
        Path(report_path).mkdir(parents=True, exist_ok=True)


    psnrs = []
    ssims = []
    ocr_acc = []
    with torch.no_grad():
        for i in tqdm(range(num_test_images), total=num_test_images):
            gt, obs = X[i, :, :], Y[i, :, :]
            gt, obs = gt.unsqueeze(0), obs.unsqueeze(0)
            print(gt.shape, obs.shape)
            obs = obs.to('cuda')
            upsample = torch.nn.Upsample(size=gt.shape[2:], mode='nearest')

            # create reconstruction from observation
            obs = reconstructor.downsampling(obs)
            reco = reconstructor.forward(obs)
            reco = upsample(reco)
            reco = reco.cpu().numpy()
            reco = np.clip(reco, 0, 1)
            # calculate quality metrics
            psnrs.append(PSNR(reco[0][0], gt.numpy()[0][0]))
            ssims.append(SSIM(reco[0][0], gt.numpy()[0][0]))
            ocr_acc.append(evaluateImage(reco[0][0], text[i]))

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


    if save_report:
        report_dict = {'stettings': {'num_test_images': num_test_images},
                    'results': {'mean_psnr': float(np.mean(psnrs)) , 
                                'std_psnr': float(np.std(psnrs)),
                                'mean_ssim': float(np.mean(ssims)) ,
                                'std_ssim': float(np.std(ssims)), 
                                'mean_ocr_acc': float(np.mean(ocr_acc)) }}
        report_file_path =  os.path.join(report_path, 'report.yaml')
        with open(report_file_path, 'w') as file:
            documents = yaml.dump(report_dict, file)

    dataset = BlurredDataModule(batch_size=8, blurring_step=step)
    dataset.prepare_data()
    dataset.setup()


    if plot_examples:
        for i, batch in tqdm(zip(range(3), dataset.test_dataloader()),
                            total=3):

            with torch.no_grad():
                gt, obs = batch
                obs = obs.to('cuda')
                upsample = torch.nn.Upsample(size=gt.shape[2:], mode='nearest')

                # create reconstruction from observation
                obs = reconstructor.downsampling(obs)
                reco = reconstructor.forward(obs)
                reco = upsample(reco)
                reco = reco.cpu().numpy()
        
            gt = gt.cpu().numpy()[0][0]

            _, ax = plot_images([reco[0,0,:,:].T, gt.T],
                                fig_size=(10, 4), vrange='equal', cbar='auto')
            ax[0].set_xlabel('PSNR: {:.2f}, SSIM: {:.2f}'.format(psnrs[i],
                                                                ssims[i]))
            ax[0].set_title('Reconstruction')
            ax[1].set_title('Ground truth')
            ax[0].figure.suptitle('test sample {:d}'.format(i))
            
            if save_report:
                img_save_path = os.path.join(report_path,'img')

                Path(img_save_path).mkdir(parents=True, exist_ok=True)
                img_save_path = os.path.join(img_save_path, 'test sample {:d}'.format(i)+'.png')
                plt.savefig(img_save_path, dpi=None, facecolor='w', edgecolor='w',
                        orientation='portrait', format=None, transparent=False,
                        bbox_inches=None, pad_inches=0.1, metadata=None)

    reconstructor.to("cpu") 