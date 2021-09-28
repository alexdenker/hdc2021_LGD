"""
learned iterative deblurrer

"""


import pytorch_lightning as pl

import torch 
import torch.nn as nn 
import torch.nn.functional as F

import numpy as np
import torchvision


from deblurrer.model.UNet import UNet
#from dival.reconstructors.networks.unet import UNet
from deblurrer.model.bokeh_blur_rfft import BokehBlur
from deblurrer.model.AnisotropicDiffusion import PeronaMalik
from deblurrer.utils.ocr import evaluateImage

import torchvision.transforms.functional as TF
import random


class IterativeReconstructor(pl.LightningModule):
    def __init__(self, lr=1e-4, n_iter=8, n_memory=5, 
                 batch_norm=True,channels=[2,4, 8, 16, 16], skip_channels=[2,4,8,16,16], radius=5.5, img_shape=(1460, 2360), kappa=0.03, 
                 regularization='pm', use_sigmoid=True, jittering_std=0.0, loss="l1", op_init=None, kappa_wiener=0.0, op_filtered=None):
        super().__init__()
        # img_shape: 181, 294
        self.save_hyperparameters()

        self.lr = self.hparams.lr
        self.n_iter = self.hparams.n_iter
        self.n_memory = self.hparams.n_memory
        self.batch_norm = self.hparams.batch_norm
        self.channels = self.hparams.channels
        self.skip_channels = self.hparams.skip_channels
        self.radius = self.hparams.radius
        self.img_shape = self.hparams.img_shape
        self.kappa = self.hparams.kappa
        self.regularization = self.hparams.regularization
        self.use_sigmoid = self.hparams.use_sigmoid
        self.jittering_std = self.hparams.jittering_std
        self.loss = self.hparams.loss
        self.op_init = self.hparams.op_init 
        self.kappa_wiener = self.hparams.kappa_wiener
        self.op_filtered = self.hparams.op_filtered

        self.downsampling = Downsampling(steps=3)

        self.blur = BokehBlur(r=self.radius, shape=self.img_shape)
        if self.regularization == 'pm':
            op_reg = PeronaMalik(kappa=self.kappa)
        else:
            op_reg = None


        if self.op_init == 'wiener':
            op_init = lambda x : self.blur.wiener_filter(x, kappa=self.kappa_wiener)
        else:
            op_init = None

        if self.op_filtered is not None: 
            op_filtered = lambda x : self.blur.wiener_filter(x, kappa=self.kappa_wiener)
        else:
            op_filtered = None            

        self.net = IterativeDeblurringNet(n_iter=self.n_iter, op=self.blur, op_reg=op_reg, op_init=op_init, op_filtered=op_filtered, use_sigmoid = self.use_sigmoid,
            n_memory=self.n_memory,channels=self.channels, skip_channels=self.skip_channels,
            batch_norm=self.batch_norm)


    def my_data_augmentation(self, image, blurred_image):
        if random.random() > 0.75:
            image = TF.hflip(image)
            blurred_image = TF.hflip(blurred_image)
        if random.random() > 0.75: 
            image = TF.vflip(image)
            blurred_image = TF.vflip(blurred_image)        
        if random.random() > 0.9: 
            image = TF.invert(image)
            blurred_image = TF.invert(blurred_image)
        return image, blurred_image


    def forward(self, y):

        return self.net(y)

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y, _ = batch[0]
        x = self.downsampling(x)
        y = self.downsampling(y)

        if self.jittering_std > 0:
            y = y + torch.randn(y.shape,device=self.device)*self.jittering_std

        x, y = self.my_data_augmentation(x, y)

        x_hat = self.net(y) 
        """
        if batch_idx == 0:
            import matplotlib.pyplot as plt 
            import datetime
            fig, (ax1, ax2, ax3) = plt.subplots(1,3)

            ax1.imshow(x.detach().cpu()[0,0,:,:])
            ax1.set_title("x")
            ax2.imshow(y.detach().cpu()[0,0,:,:])
            ax2.set_title("y")
            ax3.imshow(x_hat.detach().cpu()[0,0,:,:])
            ax3.set_title("x hat")


            plt.savefig("train_img_{}.png".format(datetime.datetime.now().strftime('%s'))) 
        """

        if self.loss == 'l1':
            l1_loss = torch.nn.L1Loss()
            loss = l1_loss(x_hat, x) #F.mse_loss(x_hat, x) 
        elif self.loss == 'both':
            l1_loss = torch.nn.L1Loss()
            loss = 0.5*l1_loss(x_hat, x) + 0.5*F.mse_loss(x_hat, x)
        else: 
            loss = F.mse_loss(x_hat, x)
        

        if random.random() > 0.6:
            #for i in range(1, len(batch)):
            #    x, _ = batch[i]
            x, _ = batch[-1]
            x = x[0:2, ...]

            y = self.blur(x)

            if self.jittering_std > 0:
                y = y + torch.randn(y.shape,device=self.device)*self.jittering_std
                x = x + torch.randn(x.shape,device=self.device)*self.jittering_std

            x_hat = self.net(y) 
        
            loss = loss +  0.1 * F.mse_loss(x_hat, x)
        
        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        
        if batch_idx == 0:
            self.last_batch = batch

        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y, text = batch
        x = self.downsampling(x)
        y = self.downsampling(y)
        x_hat = self.net(y) 
        
        if self.loss == 'l1':
            l1_loss = torch.nn.L1Loss()
            loss = l1_loss(x_hat, x) #F.mse_loss(x_hat, x) 
        elif self.loss == 'both':
            l1_loss = torch.nn.L1Loss()
            loss = 0.5*l1_loss(x_hat, x) + 0.5*F.mse_loss(x_hat, x)
        else: 
            loss = F.mse_loss(x_hat, x)

        # preprocess image 
        upsample = torch.nn.Upsample(size=(1460, 2360), mode='nearest')
        x_hat = upsample(x_hat)
        x_hat = x_hat.cpu().numpy()
        x_hat = np.clip(x_hat, 0, 1)
        
        """
        if batch_idx == 0:
            import matplotlib.pyplot as plt 
            import datetime
            fig, (ax1, ax2, ax3) = plt.subplots(1,3)

            ax1.imshow(x.detach().cpu()[0,0,:,:])
            ax1.set_title("x")
            ax2.imshow(y.detach().cpu()[0,0,:,:])
            ax2.set_title("y")
            ax3.imshow(x_hat[0,0,:,:])
            ax3.set_title("x hat")

            plt.savefig("val_img_{}.png".format(datetime.datetime.now().strftime('%s'))) 
        """
        ocr_acc = []
        for i in range(len(text)):
            ocr_acc.append(evaluateImage(x_hat[i, 0, :, :], text[i]))

        # Logging to TensorBoard by default
        self.log('val_loss', loss)
        self.log('val_ocr_acc', np.mean(ocr_acc))
        return loss 

    
    def training_epoch_end(self, result):
        # no logging of histogram. Checkpoint gets too big
        #for name,params in self.named_parameters():
        #    self.logger.experiment.add_histogram(name, params, self.current_epoch)

        x, y, _ = self.last_batch[0]

        x = self.downsampling(x)
        y = self.downsampling(y)

        img_grid = torchvision.utils.make_grid(x, normalize=True,
                                               scale_each=True)

        self.logger.experiment.add_image(
            "ground truth", img_grid, global_step=self.current_epoch)
        
        blurred_grid = torchvision.utils.make_grid(y, normalize=True,
                                               scale_each=True)
        self.logger.experiment.add_image(
            "blurred image", blurred_grid, global_step=self.current_epoch)

        with torch.no_grad():
            x_hat = self.forward(y)

            reco_grid = torchvision.utils.make_grid(x_hat, normalize=True,
                                                    scale_each=True)
            self.logger.experiment.add_image(
                "deblurred", reco_grid, global_step=self.current_epoch)
            for idx in range(1, len(self.last_batch)):
                x, _ = self.last_batch[idx]
                with torch.no_grad():
                    y = self.blur(x)
                    x_hat = self.net(y) 

                    reco_grid = torchvision.utils.make_grid(x_hat, normalize=True,
                                                            scale_each=True)
                    self.logger.experiment.add_image(
                        "deblurred set " + str(idx) , reco_grid, global_step=self.current_epoch)

                    gt_grid = torchvision.utils.make_grid(x, normalize=True,
                                                            scale_each=True)
                    self.logger.experiment.add_image(
                        "ground set " + str(idx), gt_grid, global_step=self.current_epoch)

                    blurred_grid = torchvision.utils.make_grid(y, normalize=True,
                                                            scale_each=True)
                    self.logger.experiment.add_image(
                        "blurred set " + str(idx), blurred_grid, global_step=self.current_epoch)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.9)
        # initial lr = 0.001
        # after epoch 10: lr = 0.0009
        # after epoch 20: lr = 0.00081
        # ... 

        schedulers = {
         'scheduler': scheduler,
         'monitor': 'val_loss', 
         'interval': 'epoch',
         'frequency': 1 }
        return [optimizer], [schedulers]


class Downsampling(nn.Module):
    def __init__(self, steps=3):
        super(Downsampling, self).__init__()

        self.steps = steps 

        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=2)


    def forward(self, x):

        for i in range(self.steps):
            x = self.avg_pool(x)

        return x 


class IterativeDeblurringBlock(nn.Module):
    """
    Block for iterative deblurring network. Because of the huge size of our images (1460, 2360)
    this module first consits of several up- and downsampling steps, i.e. it is implemented as a Unet
    
    """
    def __init__(self, n_in=3, n_out=1, n_memory=5, channels=[4, 8, 16, 16], skip_channels=[4,8,16,16], batch_norm=True):
        super(IterativeDeblurringBlock, self).__init__()

        self.block = UNet(in_ch=n_in + n_memory, out_ch=n_out + n_memory, channels=channels, skip_channels=skip_channels, use_sigmoid=False, use_norm=batch_norm)

    def forward(self, x):
        upd = self.block(x)
        return upd

class IterativeDeblurringNet(nn.Module):
    def __init__(self, n_iter, op, op_reg, op_filtered = None, op_init=None, n_memory=5, 
                 use_sigmoid=True, channels=[4,4, 8, 16, 16], skip_channels=[4,4,8,16,16],
                 batch_norm=True):
        super(IterativeDeblurringNet, self).__init__()
        self.n_iter = n_iter
        self.op = op
        self.op_reg = op_reg
        self.op_init = op_init
        self.op_filtered = op_filtered
        self.n_memory = n_memory
        self.use_sigmoid = use_sigmoid
        
        if self.op_reg is not None:
            if self.op_filtered is not None:
                self.iterative_blocks = nn.ModuleList()
                for it in range(n_iter):
                    self.iterative_blocks.append(IterativeDeblurringBlock(
                        n_in=5, n_out=1, n_memory=self.n_memory, batch_norm=batch_norm,channels=channels, skip_channels=skip_channels))
            else: 
                self.iterative_blocks = nn.ModuleList()
                for it in range(n_iter):
                    self.iterative_blocks.append(IterativeDeblurringBlock(
                        n_in=4, n_out=1, n_memory=self.n_memory, batch_norm=batch_norm,channels=channels, skip_channels=skip_channels))
        else: 
            if self.op_filtered is not None: 
                self.iterative_blocks = nn.ModuleList()
                for it in range(n_iter):
                    self.iterative_blocks.append(IterativeDeblurringBlock(
                        n_in=4, n_out=1, n_memory=self.n_memory, batch_norm=batch_norm,channels=channels, skip_channels=skip_channels))
            else:
                self.iterative_blocks = nn.ModuleList()
                for it in range(n_iter):
                    self.iterative_blocks.append(IterativeDeblurringBlock(
                        n_in=3, n_out=1, n_memory=self.n_memory, batch_norm=batch_norm,channels=channels, skip_channels=skip_channels))

    def forward(self, y, it=-1):
        # current iterate
        x_cur = torch.zeros(y.shape[0], 1+self.n_memory, *self.op.shape,
                                 device=y.device)
        if self.op_init is not None:
            x_cur[:] = self.op_init(y)  # broadcast across dim=1
        
        # memory
        #s = torch.zeros(y.shape[0], self.n_memory, *self.op.shape,
        #                         device=y.device)
        n_iter = self.n_iter if it == -1 else min(self.n_iter, it)
        for i in range(n_iter):
            grad = self.op.grad(x_cur[:, 0:1, ...], y)

            if self.op_reg is not None: 
                pm = self.op_reg(x_cur[:, 0:1, ...])
                x_update = torch.cat([x_cur, grad, pm, y], dim=1)
            else:
                x_update = torch.cat([x_cur, grad, y], dim=1)

            if self.op_filtered is not None: 
                x_filtered = self.op_filtered(x_cur[:, 0:1, ...])
                x_update = torch.cat([x_update, x_filtered], dim=1)

            x_update = self.iterative_blocks[i](x_update)
            x_cur = x_cur + x_update
            #x_cur = x_cur + x_update[:, 0:1, ...]
            #s = x_update[:, 1:, ...]

        x = x_cur[:, 0:1, ...]
        if self.use_sigmoid:
            x = torch.sigmoid(x)
        return x