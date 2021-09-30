

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import torch 

import torchvision 
import torchvision.transforms as transforms

import matplotlib.pyplot as plt 

from deblurrer.model.GD_deblurrer import IterativeReconstructor
from deblurrer.model.upsampling_net import UpsamplingNet


transform = transforms.Compose(
    [transforms.Grayscale(), 
    transforms.ToTensor(), 
    transforms.Resize(size=(181, 294)),
    transforms.RandomInvert(p=0.5)])

batch_size = 4

trainset = torchvision.datasets.STL10(root="/localdata/STL10", split='train', download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)


images, labels = next(iter(trainloader))


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


    #reconstructor = IterativeReconstructor(radius=42, n_memory=2, n_iter=13, channels=[4,4, 8, 8, 16], skip_channels=[4,4,8,8,16])
    reconstructor = IterativeReconstructor.load_from_checkpoint(chkp_path)
    reconstructor.net.eval()
    reconstructor.to("cuda")

    upsampling_model = UpsamplingNet(in_ch=1, hidden_ch=32, out_ch=1)
    upsampling_model.load_state_dict(torch.load('upsampling_model_step_{}.pt'.format(step))) # TODO
    upsampling_model.to("cuda")


    with torch.no_grad():
        images = images.to("cuda")
        image_deblur = reconstructor.blur(images)
        
        image_reco = reconstructor.forward(image_deblur)
        image_reco = upsampling_model(image_reco)
        image_deblur = image_deblur.cpu()
        image_reco = image_reco.cpu()

        fig, axes = plt.subplots(batch_size, 3)
        for i in range(image_reco.shape[0]):

            axes[i,0].imshow(images.cpu().numpy()[i,0,:,:], cmap="gray")
            axes[i,0].set_title("gt x")
            axes[i,0].axis("off")
            axes[i,1].imshow(image_deblur[i,0,:,:], cmap="gray")
            axes[i,1].set_title("blurred y")
            axes[i,1].axis("off")
            axes[i,2].imshow(image_reco[i,0,:,:], cmap="gray")
            axes[i,2].set_title("reconstruction")
            axes[i,2].axis("off")

        plt.savefig("sanity_check_blur_" + str(step) + ".png")
        #plt.show()
        plt.close()

