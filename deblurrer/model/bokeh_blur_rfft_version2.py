import torch 
import torch.nn as nn 
from torch.nn import functional as F 
import numpy as np 
from scipy.special import j1

class BokehBlur(nn.Module):
    def __init__(self, r=500., shape=(1460, 2360), kappa=0.002):
        super(BokehBlur, self).__init__()

        self.r = torch.nn.parameter.Parameter(torch.tensor(r).float(), requires_grad=False)
        self.shape = shape

        kx, ky = torch.meshgrid(torch.arange(-self.shape[0]//2, self.shape[0]//2),
                              torch.arange(-self.shape[1]//2, self.shape[1]//2))
        kx = kx.float()
        ky = ky.float()
        self.KR = torch.sqrt((kx**2)/1.0 + (ky**2)/1.5) # 1.5

        self.scale = torch.tensor(200.).float()
        self.kappa = kappa

        self.filter = self.get_filter()
        self.filter_wiener = self.get_wiener_filter()
        #D_numeric = torch.zeros(self.shape)
        #D_numeric[KR < self.r] = 1.0/(np.pi*r**2)
        #D_numeric = torch.fft.fftshift(D_numeric) # get the circle in the right place!!!

        #D_filter = torch.fft.rfft2(D_numeric)
        #self.D_filter = D_filter
        #self.D_filter_adjoint = torch.conj(D_filter)#torch.flip(torch.conj(D_filter), [1])
    
    def get_filter(self):
        D_numeric = torch.zeros(self.shape)
        D_numeric[self.KR < self.r] = 1.0/(np.pi*self.r**2)
        #D_numeric[self.KR < 3/4*self.r] = D_numeric[self.KR < 3/4*self.r] + 1.0/(np.pi*self.r**2)

        #D_numeric =  1.0/(np.pi*self.r**2)*torch.sigmoid(self.scale*(self.r - self.KR))

        D_numeric = torch.fft.fftshift(D_numeric) # get the circle in the right place!!!
        D_filter = torch.fft.rfft2(D_numeric)
        return D_filter
    
    def get_wiener_filter(self):
        kappa = torch.tensor(self.kappa)
        
        D_numeric = torch.zeros(self.shape)
        D_numeric[self.KR < self.r] = 1.0/(np.pi*self.r**2)            
        #D_numeric =  1.0/(np.pi*self.r**2)*torch.sigmoid(self.scale*(self.r - self.KR))
        D_numeric = torch.fft.fftshift(D_numeric) # get the circle in the right place!!!
        D_filter = torch.fft.rfft2(D_numeric)

        wiener_filter = (torch.abs(D_filter)**2/(torch.abs(D_filter)**2 + kappa))/D_filter

        return wiener_filter


    def forward(self, x):
        #D_numeric =  1.0/(np.pi*self.r**2)*F.relu(self.r - self.KR.to(x.device))
        #D_numeric =  1.0/(np.pi*self.r**2)*torch.sigmoid(self.scale.to(x.device)*(self.r - self.KR.to(x.device)))

        #D_numeric = torch.fft.fftshift(D_numeric) # get the circle in the right place!!!
        #D_filter = torch.fft.rfft2(D_numeric)

        x_fft = torch.fft.rfft2(x)
        x_processed = torch.multiply(self.filter.to(x.device),x_fft)

        x_ifft = torch.fft.irfft2(x_processed)
        return x_ifft


    def adjoint(self, x):
        #D_numeric =  1.0/(np.pi*self.r**2)*torch.sigmoid(self.scale.to(x.device)*(self.r - self.KR.to(x.device)))

        #D_numeric = torch.fft.fftshift(D_numeric) # get the circle in the right place!!!
        #D_filter = torch.fft.rfft2(D_numeric)
        D_filter_adjoint = torch.conj(self.filter.to(x.device))

        x_fft = torch.fft.rfft2(x)
        x_processed = torch.multiply(D_filter_adjoint.to(x.device),x_fft)

        x_ifft = torch.fft.irfft2(x_processed)
        return x_ifft

    def grad(self, x, y):
        
        x_fft = torch.fft.rfft2(x)
        y_fft = torch.fft.rfft2(y)

        #D_numeric =  1.0/(np.pi*self.r**2)*torch.sigmoid(self.scale.to(x.device)*(self.r - self.KR.to(x.device)))

        #D_numeric = torch.fft.fftshift(D_numeric) # get the circle in the right place!!!
        #D_filter = torch.fft.rfft2(D_numeric)
        #res = torch.multiply(self.D_filter.to(x.device), x_fft) - y_fft

        #f_grad = torch.multiply(self.D_filter_adjoint.to(x.device), res)

        f_grad = torch.multiply(self.filter.to(x.device)**2, x_fft) - torch.multiply(self.filter.to(x.device), y_fft)
        out = torch.fft.irfft2(f_grad)
    
        return out

    def wiener_filter(self, x):
        #kappa = torch.tensor(kappa)
                
        x_fft = torch.fft.rfft2(x)
        
        #D_numeric =  1.0/(np.pi*self.r**2)*torch.sigmoid(self.scale.to(x.device)*(self.r - self.KR.to(x.device)))
        #D_numeric = torch.fft.fftshift(D_numeric) # get the circle in the right place!!!
        #D_filter = torch.fft.rfft2(D_numeric)

        #wiener_filter = (torch.abs(D_filter.to(x.device))**2/(torch.abs(D_filter.to(x.device))**2 + kappa))/D_filter.to(x.device)

        x_re = torch.fft.irfft2(torch.multiply(self.filter_wiener.to(x.device), x_fft))
        return x_re

    def filtered_grad(self, x, y):

        x_fft = torch.fft.rfft2(x)
        y_fft = torch.fft.rfft2(y)

        res = torch.multiply(self.filter.to(x.device), x_fft) - y_fft 

        f_grad = torch.fft.irfft2(torch.multiply(self.filter_wiener.to(x.device), res))

        return f_grad

if __name__ == "__main__":

    forward_model = BokehBlur(r=65.)

    from hdc2021_challenge.utils.data_util import load_data
    import matplotlib.pyplot as plt 
    import torch.nn as nn
    from math import ceil

    downsampling_factor = 1

    step = 10
    x, y = load_data('Verdana',step)
    x = x[0,:,:]/65535.
    y = y[0,:,:]/65535.
    
    x_torch = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0)
    y_torch = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(0)

    print(x_torch.shape, y_torch.shape)
    
    with torch.no_grad():
        y_pred = forward_model(x_torch)

    print(y_pred.shape)

    class Downsampling(nn.Module):
            def __init__(self, steps=3):
                super(Downsampling, self).__init__()

                self.steps = steps 

                self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)


            def forward(self, x):

                for i in range(self.steps):
                    x = self.avg_pool(x)


                return x 

    downsampler = Downsampling(downsampling_factor)

    h_down = ceil(1460 / 2*downsampling_factor)
    w_down = ceil(2360 / 2*downsampling_factor)
    print("shape:", h_down, w_down)
    # downsample forward model 
    print("filter: ", forward_model.filter.shape)
    forward_model.filter = nn.functional.interpolate(forward_model.filter.unsqueeze(0).unsqueeze(0), (h_down, w_down), mode = 'bilinear', align_corners =False)

    x_down = downsampler(x_torch)
    y_down = downsampler(y_torch)

    with torch.no_grad():
        y_pred_down = forward_model(x_down)

    fig, axes = plt.subplots(2,3)
    
    axes[0,0].imshow(x_torch[0,0,:,:], cmap="gray")
    axes[0,1].imshow(y_torch[0,0,:,:], cmap="gray")
    axes[0,2].imshow(y_pred[0,0,:,:], cmap="gray")

    axes[1,].imshow(x_down[0,0,:,:], cmap="gray")
    axes[1,1].imshow(y_down[0,0,:,:], cmap="gray")
    axes[1,2].imshow(y_pred_down[0,0,:,:], cmap="gray")

    plt.show()