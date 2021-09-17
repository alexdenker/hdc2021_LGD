
import argparse
from PIL import Image

import torch 
import matplotlib.pyplot as plt 
import os 
import numpy as np 
from model.GD_deblurrer_downsampling import IterativeReconstructor

parser = argparse.ArgumentParser(description='Apply Deblurrer to every image in a directory.')

parser.add_argument('input_files')

parser.add_argument('output_files')

parser.add_argument('step')



def main(input_files, output_files, step):
    # load model 
    base_path = "weights"
    experiment_name = 'step_' + str(step)  
    version = 'version_1'
    chkp_name = 'learned_gradient_descent'
    path_parts = [base_path, experiment_name, 'default',
                version, 'checkpoints', chkp_name + '.ckpt']
    chkp_path = os.path.join(*path_parts)
    #reconstructor = IterativeReconstructor(radius=42, n_memory=2, n_iter=13, channels=[4,4, 8, 8, 16], skip_channels=[4,4,8,8,16])
    reconstructor = IterativeReconstructor.load_from_checkpoint(chkp_path)
    reconstructor.to("cuda")

    upsample = torch.nn.Upsample(size=(1460, 2360), mode='nearest')

    for f in os.listdir(input_files):
        if f.endswith("tif"):
            y = np.array(Image.open(os.path.join(input_files, f))) # not blurry
            print(y.shape)
            y = torch.from_numpy(y/65535.).float()
            y = y.unsqueeze(0).unsqueeze(0)
            y = y.to("cuda")
            with torch.no_grad():
                y = reconstructor.downsampling(y)
                x_hat = reconstructor.forward(y)
                x_hat = upsample(x_hat)
                x_hat = x_hat.cpu().numpy()

            im = Image.fromarray(x_hat[0][0]*255.).convert("L")
            print(im)
            im.save(os.path.join(output_files,f.split(".")[0] + ".PNG"))


    return 0




if __name__ == "__main__":

    args = parser.parse_args()


    main(args.input_files, args.output_files, args.step)


"""

step = 19
base_path = "/localdata/AlexanderDenker/deblurring_experiments"
experiment_name = 'step_' + str(step)  
version = 'version_0'
chkp_name = 'epoch=17-step=359'
path_parts = [base_path, 'gd_deblurring', experiment_name, 'default',
              version, 'checkpoints', chkp_name + '.ckpt']
chkp_path = os.path.join(*path_parts)
#reconstructor = IterativeReconstructor(radius=42, n_memory=2, n_iter=13, channels=[4,4, 8, 8, 16], skip_channels=[4,4,8,8,16])
reconstructor = IterativeReconstructor.load_from_checkpoint(chkp_path)
reconstructor.to("cuda")

#print("radius", reconstructor.hparams.radius)


with torch.no_grad():
    blur = BokehBlur(r=radius_dict[step], shape=reconstructor.hparams.img_shape)
    #images = 1 - images
    image_deblur = blur(images)
    
    image_deblur = image_deblur.to("cuda")
    image_reco = reconstructor.forward(image_deblur)
    image_deblur = image_deblur.cpu()
    image_reco = image_reco.cpu()

    fig, axes = plt.subplots(batch_size, 3)
    for i in range(image_reco.shape[0]):

        axes[i,0].imshow(images[i,0,:,:], cmap="gray")
        axes[i,0].set_title("gt x")
        axes[i,1].imshow(image_deblur[i,0,:,:], cmap="gray")
        axes[i,1].set_title("blurred y")
        axes[i,2].imshow(image_reco[i,0,:,:], cmap="gray")
        axes[i,2].set_title("reconstruction")
    plt.savefig("sanity_check_blur_" + str(step) + ".png")
    plt.show()

"""