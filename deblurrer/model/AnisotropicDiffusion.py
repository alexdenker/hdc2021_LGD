import torch
import torch.nn as nn


class PeronaMalik(nn.Module):
    def __init__(self, kappa, option = 2):
        """
        kappa: float
            Conduction coefficient. ``kappa`` controls conduction
            as a function of the gradient. If ``kappa`` is low small intensity
            gradients are able to block conduction and hence diffusion across
            steep edges. A large value reduces the influence of intensity gradients
            on conduction.
            20 - 100 seems to work quite well for ground truth images 
            the edges of the blurred image are nearly non-existent...

        option: optional, int
            Type of anisotrpic function g(|nabla I|)
            1: 1./(1. + (nabla_x/kappa)**2)   (favours wide regions over smaller ones)
            2: torch.exp(-(nabla_x/kappa)**2.)  (favours high contrast edges over low contrast ones)
        
        """
        super(PeronaMalik, self).__init__()

        self.kappa = kappa 
        self.option = option

        self.grad_x1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, padding_mode='reflect', bias=False)
        grad_x1_filter = 1./torch.sqrt(torch.tensor(2.))*torch.tensor([[0.0, 0.0, 0.0]
                                                    , [0.0, -1.0, 1.0]
                                                    , [0.0, 0.0, 0.0]]).unsqueeze(0).unsqueeze(0).float()    
        self.grad_x1.weight = torch.nn.parameter.Parameter(grad_x1_filter, requires_grad=False)


        self.grad_x2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, padding_mode='reflect', bias=False)
        grad_x2_filter = 1./torch.sqrt(torch.tensor(2.))*torch.tensor([[0.0, 0.0, 0.0]
                                                    , [-1.0, 1.0, 0.0]
                                                    , [0.0, 0.0, 0.0]]).unsqueeze(0).unsqueeze(0).float()    
        self.grad_x2.weight = torch.nn.parameter.Parameter(grad_x2_filter, requires_grad=False)

        self.grad_y1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, padding_mode='reflect', bias=False)
        grad_y1_filter = 1./torch.sqrt(torch.tensor(2.))*torch.tensor([[0.0, 1.0, 0.0]
                                                    , [0.0, -1.0, 0.0]
                                                    , [0.0, 0.0, 0.0]]).unsqueeze(0).unsqueeze(0).float()    
        self.grad_y1.weight = torch.nn.parameter.Parameter(grad_y1_filter, requires_grad=False)


        self.grad_y2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, padding_mode='reflect', bias=False)
        grad_y2_filter = 1./torch.sqrt(torch.tensor(2.))*torch.tensor([[0.0, 0.0, 0.0]
                                                    , [0.0, 1.0, 0.0]
                                                    , [0.0, -1.0, 0.0]]).unsqueeze(0).unsqueeze(0).float()    
        self.grad_y2.weight = torch.nn.parameter.Parameter(grad_y2_filter, requires_grad=False)



    def forward(self, x):
        fy = self.grad_y1(x)
        fx = self.grad_x1(x)

        gradf2 = fy**2 + fx**2

        if self.option == 1: 
            cI = 1./(1. + gradf2/self.kappa**2)
        else: 
            cI = torch.exp(-gradf2/self.kappa**2.)

        gradfx = torch.mul(fx, cI)
        gradfy = torch.mul(fy, cI)

        term = self.grad_x2(gradfx) + self.grad_y2(gradfy)

        return term



if __name__ == "__main__":
    import matplotlib.pyplot as plt 

    from hdc2021_challenge.utils.data_util import load_calibration_images, load_data

    step= 9
    x, y = load_data('Verdana',step)
    x = x[0,:,:]/65535.
    y = y[0,:,:]/65535.

    x_torch = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float()
    y_torch = torch.from_numpy(y).unsqueeze(0).unsqueeze(0).float()


    perona_malik = PeronaMalik(kappa=0.03)

    x_pm = perona_malik(x_torch)
    y_pm = perona_malik(y_torch)

    fig, axes = plt.subplots(2,2, sharex=True, sharey=True)

    axes[0, 0].imshow(x_torch[0,0,:,:], cmap="gray")
    axes[0, 0].set_title("x")
    axes[0, 1].imshow(x_pm[0,0,:,:])
    axes[0, 1].set_title("pm(x)")

    axes[1, 0].imshow(y_torch[0,0,:,:], cmap="gray")
    axes[1, 0].set_title("y")
    axes[1, 1].imshow(y_pm[0,0,:,:])
    axes[1, 1].set_title("pm(y)")


    plt.show()


    #fdiscrepancy = real(ifftshift(ifft2(hf.*fft2(g))) - ifft2(hf2.*fft2(fPM)))