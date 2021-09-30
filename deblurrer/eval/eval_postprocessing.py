

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
from deblurrer.model.GD_deblurrer import IterativeReconstructor
from deblurrer.model.upsampling_net import UpsamplingNet

from deblurrer.utils import data_util


from dival.util.plot import plot_images



from dival.util.plot import plot_images
import yaml

import pytesseract
from fuzzywuzzy import fuzz

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

for step in range(20):
    #step = 3

    base_path = "/localdata/AlexanderDenker/deblurring_experiments"
    experiment_name = 'step_' + str(step)  

    version_dict =  {
        0 : 0, 
        1 : 0, 
        2 : 0,
        3 : 0, 
        4 : 0,
        5 : 1,
        6 : 3,
        7 : 1, 
        8 : 4,
        9 : 2,
        10 : 1,
        11 : 1,
        12 : 1,
        13 : 0,
        14 : 1,
        15 : 1,
        16 : 0,
        17 : 1, 
        18 : 1, 
        19 : 1
    }



    version = 'version_' + str(version_dict[step])

    identifier = "val_ocr"


    path_parts = [base_path, 'run_31', experiment_name, 'default',
                version, 'checkpoints']
    chkp_path = os.path.join(*path_parts)

    if not os.path.exists(chkp_path):
        print("File not found: ", chkp_path)

    def search_for_file(chkp_path, identifier):
        for ckpt_file in os.listdir(chkp_path):
            if identifier in ckpt_file:
                return ckpt_file

    print(search_for_file(chkp_path, identifier))
    chkp_name = search_for_file(chkp_path, identifier)
    if chkp_name is None:
        print("No checkpint found...")

    chkp_path = os.path.join(chkp_path, chkp_name)
    print("Load from ",chkp_path)

    dataset = BlurredDataModule(batch_size=1, blurring_step=step)
    dataset.prepare_data()
    dataset.setup()

    num_test_images = len(dataset.test_dataloader())


    #reconstructor = IterativeReconstructor(radius=42, n_memory=2, n_iter=13, channels=[4,4, 8, 8, 16], skip_channels=[4,4,8,8,16])
    reconstructor = IterativeReconstructor.load_from_checkpoint(chkp_path)
    reconstructor.net.eval()
    reconstructor.to("cuda")

    upsampling_model = UpsamplingNet(in_ch=1, hidden_ch=32, out_ch=1)
    upsampling_model.load_state_dict(torch.load('upsampling_model_step_{}.pt'.format(step))) # TODO
    upsampling_model.to("cuda")


    report_name = "postprocessing_report_step_" + str(step) + "_model_" + identifier
    report_path = path_parts[:-1]
    report_path.append(report_name)
    report_path = os.path.join(*report_path)
    Path(report_path).mkdir(parents=True, exist_ok=True)


    ocr_acc = []
    for i, batch in tqdm(zip(range(num_test_images), dataset.test_dataloader()),
                            total=num_test_images):
        with torch.no_grad():
            gt, obs, text = batch
            
            obs = obs.to('cuda')
            #upsample = torch.nn.Upsample(size=gt.shape[2:], mode='bilinear') # '' nearest

            # create reconstruction from observation
            obs_down = reconstructor.downsampling(obs)
            reco = reconstructor.forward(obs_down)

            #reco_iter = reco.clone()
            #for _ in range(10):
            #    reco_iter = reco_iter - 0.001*reconstructor.blur.filtered_grad(reco_iter, obs_down)

            #fig, (ax1, ax2) = plt.subplots(1,2)
            #ax1.imshow(reco.cpu().numpy()[0][0], cmap="gray")
            #ax2.imshow(reco_iter.cpu().numpy()[0][0], cmap="gray")
            #plt.show()
            reco = upsampling_model(reco)
            reco = reco.cpu().numpy()
            reco = np.clip(reco, 0, 1)

            ocr_acc.append(evaluateImage(reco[0][0], text))

        
        if i < 4:
            img_save_path = os.path.join(report_path,'img')
            Path(img_save_path).mkdir(parents=True, exist_ok=True)


            im = Image.fromarray(reco[0][0]*255.).convert("L")
            im.save(os.path.join(img_save_path, "test_sample_{}.PNG".format(i)))
        
            gt = gt.cpu().numpy()[0][0]

            _, ax = plot_images([reco[0,0,:,:].T, gt.T, obs.cpu().numpy()[0,0,:,:].T],
                                fig_size=(10, 4), vrange='individual', cbar='auto')

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

        
    print('---')
    print('Results:')
    print('mean ocr acc: ', np.mean(ocr_acc))


    report_dict = {'stettings': {'num_test_images': num_test_images},
                'results': {'mean_ocr_acc': float(np.mean(ocr_acc)) }}
    report_file_path =  os.path.join(report_path, 'report.yaml')
    with open(report_file_path, 'w') as file:
        documents = yaml.dump(report_dict, file)