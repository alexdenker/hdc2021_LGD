"""
learned iterative deblurrer

"""


import pytorch_lightning as pl

import torch 
import torch.nn as nn 
import torch.nn.functional as F

import numpy as np
import torchvision


from dival.reconstructors.networks.unet import UNet
from deblurrer.model.bokeh_blur_rfft_train import BokehBlur
from deblurrer.model.AnisotropicDiffusion import PeronaMalik
from deblurrer.utils.ocr import evaluateImage


class IterativeReconstructor(pl.LightningModule):
    def __init__(self, lr=1e-4, n_iter=8, n_memory=5, 
                 batch_norm=True,channels=[2,4, 8, 16, 16], skip_channels=[2,4,8,16,16], radius=5.5, img_shape=(1460, 2360), kappa=0.03, 
                 regularization='pm', use_sigmoid=True, jittering_std=0.0, loss="l1", op_init=None, kappa_wiener=0.0):
        super().__init__()
        # img_shape: 181, 294
        self.lr = lr

        save_hparams = {
            'lr': lr,
            'n_iter': n_iter,
            'n_memory': n_memory,
            'batch_norm': batch_norm,
            'radius': radius, 
            'img_shape': img_shape, 
            'kappa' : kappa,
            'channels': channels, 
            'skip_channels': skip_channels,
            'regularization': regularization,
            'use_sigmoid': use_sigmoid, 
            'jittering_std': jittering_std,
            'which_loss' : loss,
            'op_init': op_init, 
            'kappa_wiener' : kappa_wiener
        }
        self.save_hyperparameters(save_hparams)

        self.jittering_std = jittering_std
        self.which_loss = loss

        self.downsampling = Downsampling(steps=3)

        self.blur = BokehBlur(r=radius, shape=img_shape)
        if regularization == 'pm':
            op_reg = PeronaMalik(kappa=kappa)
        else:
            op_reg = None


        if op_init == 'wiener':
            op_init = lambda x : self.blur.wiener_filter(x, kappa=kappa_wiener)
        else:
            op_init = None

        self.net = IterativeDeblurringNet(n_iter=n_iter, op=self.blur, op_reg=op_reg, op_init=op_init, use_sigmoid = use_sigmoid,
            n_memory=n_memory,channels=channels, skip_channels=skip_channels,
            batch_norm=batch_norm)


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

        x_hat = self.net(y) 


        if self.which_loss == 'l1':
            l1_loss = torch.nn.L1Loss()
            loss = l1_loss(x_hat, x) #F.mse_loss(x_hat, x) 
        elif self.which_loss == 'both':
            l1_loss = torch.nn.L1Loss()
            loss = 0.5*l1_loss(x_hat, x) + 0.5*F.mse_loss(x_hat, x)
        else: 
            loss = F.mse_loss(x_hat, x)
           
        for i in range(1, len(batch)):
            x, _ = batch[i]
            x = x[0:2, ...]

            y = self.blur(x)

            if self.jittering_std > 0:
                y = y + torch.randn(y.shape,device=self.device)*self.jittering_std
                x = x + torch.randn(x.shape,device=self.device)*self.jittering_std

            x_hat = self.net(y) 
        
            loss = loss +  0.1 * F.mse_loss(x_hat, x)

        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        
        if batch_idx == 3:
            self.last_batch = batch

        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y, text = batch
        x = self.downsampling(x)
        y = self.downsampling(y)
        x_hat = self.net(y) 
        
        if self.which_loss == 'l1':
            l1_loss = torch.nn.L1Loss()
            loss = l1_loss(x_hat, x) #F.mse_loss(x_hat, x) 
        elif self.which_loss == 'both':
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
            fig, (ax1, ax2) = plt.subplots(1,2)
            ax1.imshow(x_hat[0,0,:,:], cmap="gray")
            score = evaluateImage(x_hat[0, 0, :, :], text[0])
            ax1.set_title(text[0] + " score: " + str(score))

            x = upsample(x)
            x = x.detach().cpu().numpy()
            x = np.clip(x, 0, 1)

            ax2.imshow(x[0,0,:,:], cmap="gray")
            ax2.set_title("GT")

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

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
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
    def __init__(self, n_iter, op, op_reg, op_init=None, n_memory=5, 
                 use_sigmoid=True, channels=[4,4, 8, 16, 16], skip_channels=[4,4,8,16,16],
                 batch_norm=True):
        super(IterativeDeblurringNet, self).__init__()
        self.n_iter = n_iter
        self.op = op
        self.op_reg = op_reg
        self.op_init = op_init
        self.n_memory = n_memory
        self.use_sigmoid = use_sigmoid
        
        if self.op_reg is not None:
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

            x_update = self.iterative_blocks[i](x_update)
            x_cur = x_cur + x_update
            #x_cur = x_cur + x_update[:, 0:1, ...]
            #s = x_update[:, 1:, ...]

        x = x_cur[:, 0:1, ...]
        if self.use_sigmoid:
            x = torch.sigmoid(x)
        return x