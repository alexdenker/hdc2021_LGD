import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from deblurrer.utils.blurred_dataset import BlurredDataModule, MultipleBlurredDataModule, ConcatBlurredDataModule
from deblurrer.model.upsampling_net import UpsamplingNet
from deblurrer.model.GD_deblurrer import IterativeReconstructor



from tqdm import tqdm

import numpy as np 

import matplotlib.pyplot as plt

step = 19

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


class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        #### Gradient Loss 
        self.conv2d_sobelx = torch.nn.Conv2d(1,1, 3, bias=False)
        sobel_x = torch.tensor([[-1., 0., 1.],[-2., 0., 2.],[-1., 0., 1.]],dtype=torch.float32).reshape(1,1,3,3)
        self.conv2d_sobelx.weight = torch.nn.parameter.Parameter(sobel_x, requires_grad=False)

        self.conv2d_sobely = torch.nn.Conv2d(1,1, 3, bias=False)
        sobel_y = torch.tensor([[-1., -2., -1.],[0., 0., 0.],[1., 2., 1.]],dtype=torch.float32).reshape(1,1,3,3)
        self.conv2d_sobely.weight = torch.nn.parameter.Parameter(sobel_y, requires_grad=False)

    def gradient_magnitude(self, xhat, x):
        diff = (x - xhat)**2

        return torch.sqrt(self.conv2d_sobelx(diff)**2 + self.conv2d_sobely(diff)**2).mean()





dataset = ConcatBlurredDataModule(batch_size=4, blurring_step=step)
dataset.prepare_data()
dataset.setup()

num_images = len(dataset.train_dataloader())

#reconstructor = IterativeReconstructor(radius=42, n_memory=2, n_iter=13, channels=[4,4, 8, 8, 16], skip_channels=[4,4,8,8,16])
reconstructor = IterativeReconstructor.load_from_checkpoint(chkp_path)
reconstructor.net.eval()
reconstructor.to("cuda")

for param in reconstructor.net.parameters():
    param.requires_grad = False


upsample = torch.nn.Upsample(size=(1460, 2360), mode='bilinear', align_corners=True) # 'nearest'

upsampling_model = UpsamplingNet(in_ch=1, hidden_ch=32, out_ch=1)
upsampling_model.load_state_dict(torch.load('upsampling_model.pt')) # TODO
upsampling_model.to("cuda")


gradient_loss = GradientLoss()
gradient_loss.to("cuda")

optimizer = optim.Adam(upsampling_model.parameters(), lr=0.001)

max_epochs = 50
for epoch in range(max_epochs):
    mean_loss = []
    for i, batch in tqdm(zip(range(num_images), dataset.train_dataloader()),
                        total=num_images):
        optimizer.zero_grad()
        x, y, _ = batch[0]
        x, y = x.to("cuda"), y.to("cuda")
        y = reconstructor.downsampling(y)
        x_hat = reconstructor.net(y) 

        x_upsample = upsampling_model(x_hat)

        loss = 0.5*F.mse_loss(x_upsample, x) + 5*torch.mean((x_upsample - x)**2/(x**2 + 0.1)) + gradient_loss.gradient_magnitude(x_upsample, x)


        x, _ = batch[-1]
        x = x[0:1, ...]
        x = x.to('cuda')
        x = reconstructor.resize_image(x)
        y = reconstructor.blur(x)

        x_hat = reconstructor.net(y) 
        x_upsample = upsampling_model(x_hat)
        x = upsample(x)

        loss = loss + 0.5* F.mse_loss(x_upsample, x)

        loss.backward()
        optimizer.step()

        mean_loss.append(loss.item())

    for i, batch in tqdm(zip(range(1), dataset.test_dataloader()),
                        total=1):
        with torch.no_grad():
            x, y, _ = batch
            x, y = x.to("cuda"), y.to("cuda")
            y = reconstructor.downsampling(y)

            x_hat = reconstructor.net(y) 
            x_upsample = upsampling_model(x_hat)

            fig, (ax1, ax2) = plt.subplots(1,2)
            ax1.imshow(x_upsample.cpu().numpy()[0][0], cmap="gray")
            ax1.set_title("reconstruction")

            ax2.imshow(x.cpu().numpy()[0][0], cmap="gray")
            ax2.set_title("gt")
            #plt.savefig("val_img_at_epoch_{}.png".format(epoch))
            plt.close()
            #plt.show()
    print("Epoch {}. Loss: {}".format(epoch, np.mean(mean_loss)))
    torch.save(upsampling_model.state_dict(), 'upsampling_model_step_{}.pt'.format(step))

torch.save(upsampling_model.state_dict(), 'upsampling_model_step_{}.pt'.format(step))