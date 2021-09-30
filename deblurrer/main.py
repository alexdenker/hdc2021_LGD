
import argparse
from PIL import Image

import torch 
import matplotlib.pyplot as plt 
import os 
import numpy as np 
from deblurrer.model.GD_deblurrer import IterativeReconstructor
from deblurrer.model.upsampling_net import UpsamplingNet

parser = argparse.ArgumentParser(description='Apply Deblurrer to every image in a directory.')

parser.add_argument('input_files')

parser.add_argument('output_files')

parser.add_argument('step')



def main(input_files, output_files, step):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model 
    base_path = os.path.join(os.path.dirname(__file__), 'hdc2021_weights')
    experiment_name = 'step_' + str(step)  
    path_parts = [base_path, experiment_name, 'checkpoints',  'learned_gradient_descent.ckpt']
    chkp_path = os.path.join(*path_parts)

    reconstructor = IterativeReconstructor.load_from_checkpoint(chkp_path)
    reconstructor.to(device)

    upsampling_model = UpsamplingNet(in_ch=1, hidden_ch=32, out_ch=1)
    chkp_upsample_path = path_parts[:-1] + ['upsampling_model_step_{}.pt'.format(step)]
    upsampling_model.load_state_dict(torch.load(os.path.join(*chkp_upsample_path), map_location=device))
    upsampling_model.to(device)


    for f in os.listdir(input_files):
        if f.endswith("tif"):
            y = np.array(Image.open(os.path.join(input_files, f))).astype(np.float32) # not blurry
            y = torch.from_numpy(y).float()
            y = (y - torch.min(y))/(torch.max(y) - torch.min(y))
            y = y.unsqueeze(0).unsqueeze(0)
            y = y.to(device)
            with torch.no_grad():
                y = reconstructor.downsampling(y)
                x_hat = reconstructor.forward(y)
                x_hat = upsampling_model(x_hat)
                x_hat = x_hat.cpu().numpy()

            im = Image.fromarray(x_hat[0][0]*255.).convert("L")

            os.makedirs(output_files, exist_ok=True)
            im.save(os.path.join(output_files,f.split(".")[0] + ".PNG"))


    return 0


if __name__ == "__main__":

    args = parser.parse_args()


    main(args.input_files, args.output_files, args.step)
