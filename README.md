# hdc2021_LGD
Our submission for the HDC2021. 

## Install 

Install the package using:

```
pip install -e .
```

The neural networks weights can be downloaded using this link: https://seafile.zfn.uni-bremen.de/d/c19be19f1f134c0d8ad2/. Afterwards move all weights to *deblurrer/*, i.e. folder structure *deblurrer/hdc2021_weights/step_i* for each step i=0,...,19.

## Usage 

Prediction on images in a folder can be done using:

```
python deblurrer/main.py path-to-input-files path-to-output-files step
```

## Method

We want to reconstruct the original image <a href="https://www.codecogs.com/eqnedit.php?latex=f&space;\in&space;X" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f&space;\in&space;X" title="f \in X" /></a> from blurred and noisy data <a href="https://www.codecogs.com/eqnedit.php?latex=g_\eta&space;\in&space;X" target="_blank"><img src="https://latex.codecogs.com/gif.latex?g_\eta&space;\in&space;X" title="g_\eta \in X" /></a>. The forward operator is given by <a href="https://www.codecogs.com/eqnedit.php?latex=\mathcal{A}:&space;X&space;\rightarrow&space;X" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathcal{A}:&space;X&space;\rightarrow&space;X" title="\mathcal{A}: X \rightarrow X" /></a>. This corresponds to an inverse problem: 

<a href="https://www.codecogs.com/eqnedit.php?latex=g_\eta&space;=&space;\mathcal{A}&space;f&space;&plus;&space;\eta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?g_\eta&space;=&space;\mathcal{A}&space;f&space;&plus;&space;\eta" title="g_\eta = \mathcal{A} f + \eta" /></a>

Our goal is to define an reconstructor <a href="https://www.codecogs.com/eqnedit.php?latex=\mathcal{R}_\Theta&space;:&space;X&space;\rightarrow&space;X" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathcal{R}_\Theta&space;:&space;X&space;\rightarrow&space;X" title="\mathcal{R}_\Theta : X \rightarrow X" /></a> which produces an unblurred image from the blurred <a href="https://www.codecogs.com/eqnedit.php?latex=g_\eta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?g_\eta" title="g_\eta" /></a>:

<a href="https://www.codecogs.com/eqnedit.php?latex=f&space;\approx&space;\mathcal{R}_\Theta&space;(g_\eta)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f&space;\approx&space;\mathcal{R}_\Theta&space;(g_\eta)" title="f \approx \mathcal{R}_\Theta (g_\eta)" /></a>


Our approach is inspired by the recent trend of combining physical forward models  and learned iterative schemes [(Adler et. al.)](https://arxiv.org/abs/1704.04058). There is also experimental evidence that this approach can work well even with an approximated forward model [(Hauptmann et. al.)](https://arxiv.org/abs/1807.03191).
This approach has two steps:
1. First, we define an approximate forward model for the blurring process.
2. This approximate model is integrated into a learned gradient descent scheme.  

This approach is repeated for every step. The same training setup and neural network architecture was used for all blurring steps. The learned gradient scheme was retrained on each blurring step. 

### Approximate Forward Model 

For a fixed blurring level the out-of-focus blur can be modeled by as a linear, position invariant convolution with a circular point-spread-function: 

<a href="https://www.codecogs.com/eqnedit.php?latex=g_\eta&space;=&space;\mathcal{A}&space;f&space;&plus;&space;\eta&space;=&space;k&space;*&space;f&space;&plus;&space;\eta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?g_\eta&space;=&space;\mathcal{A}&space;f&space;&plus;&space;\eta&space;=&space;k&space;*&space;f&space;&plus;&space;\eta" title="g_\eta = \mathcal{A} f + \eta = k * f + \eta" /></a>
with 

<a href="https://www.codecogs.com/eqnedit.php?latex=k(x)&space;=&space;\left\{\begin{array}{lr}&space;\frac{1}{\pi&space;r^2},&space;&&space;\text{for&space;}&space;\|&space;x&space;\|^2&space;\le&space;r^2\\&space;0,&space;&&space;\text{else&space;}&space;\end{array}\right\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?k(x)&space;=&space;\left\{\begin{array}{lr}&space;\frac{1}{\pi&space;r^2},&space;&&space;\text{for&space;}&space;\|&space;x&space;\|^2&space;\le&space;r^2\\&space;0,&space;&&space;\text{else&space;}&space;\end{array}\right\}" title="k(x) = \left\{\begin{array}{lr} \frac{1}{\pi r^2}, & \text{for } \| x \|^2 \le r^2\\ 0, & \text{else } \end{array}\right\}" /></a>

This model works well for small blurring levels. For higher blurring levels the average error between the approximate model and the real measurements gets bigger. 

### Learned Gradient Descent

Variational methods try to recover the original image as the minimizer of an objective function 

<a href="https://www.codecogs.com/eqnedit.php?latex=\hat{f}&space;=&space;\mathcal{R}_\Theta(g\eta)&space;\in&space;\arg&space;\min_f&space;d(\mathcal{A}f,&space;g_\eta)&space;&plus;&space;\lambda&space;R(f)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{f}&space;=&space;\mathcal{R}_\Theta(g\eta)&space;\in&space;\arg&space;\min_f&space;d(\mathcal{A}f,&space;g_\eta)&space;&plus;&space;\lambda&space;R(f)" title="\hat{f} = \mathcal{R}_\Theta(g\eta) \in \arg \min_f d(\mathcal{A}f, g_\eta) + \lambda R(f)" /></a>

where <a href="https://www.codecogs.com/eqnedit.php?latex=d:X\times&space;X&space;\rightarrow&space;\mathbb{R}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?d:X\times&space;X&space;\rightarrow&space;\mathbb{R}" title="d:X\times X \rightarrow \mathbb{R}" /></a> is a data fidelity term and <a href="https://www.codecogs.com/eqnedit.php?latex=R:X&space;\rightarrow&space;\mathbb{R}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?R:X&space;\rightarrow&space;\mathbb{R}" title="R:X \rightarrow \mathbb{R}" /></a> is a regularizer. In many examples, the regularizer is convex but not differentiable (e.g. TV or l1). In these cases proximal gradient descent can be used to obtain the minimizer: 

<a href="https://www.codecogs.com/eqnedit.php?latex=f_{k&plus;1}&space;=&space;\text{prox}_{\lambda&space;R}(f_k&space;-&space;\gamma&space;\nabla&space;d(\mathcal{A}f_k,&space;g_\eta))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f_{k&plus;1}&space;=&space;\text{prox}_{\lambda&space;R}(f_k&space;-&space;\gamma&space;\nabla&space;d(\mathcal{A}f_k,&space;g_\eta))" title="f_{k+1} = \text{prox}_{\lambda R}(f_k - \gamma \nabla d(\mathcal{A}f_k, g_\eta))" /></a>

with a step size <a href="https://www.codecogs.com/eqnedit.php?latex=\gamma&space;>&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\gamma&space;>&space;0" title="\gamma > 0" /></a>. The idea of learned iterative methods is to unroll this iteration for a fixed number of steps and replace the proximal mapping with a convolutional neural network: 

<a href="https://www.codecogs.com/eqnedit.php?latex=f_{k&plus;1}&space;=&space;\Lambda_{\theta_k}(f_k,&space;\nabla&space;d(\mathcal{A}f_k,&space;g_\eta))&space;\qquad&space;\text{&space;for&space;}&space;k=0,\dots,&space;K-1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f_{k&plus;1}&space;=&space;\Lambda_{\theta_k}(f_k,&space;\nabla&space;d(\mathcal{A}f_k,&space;g_\eta))&space;\qquad&space;\text{&space;for&space;}&space;k=0,\dots,&space;K-1" title="f_{k+1} = \Lambda_{\theta_k}(f_k, \nabla d(\mathcal{A}f_k, g_\eta)) \qquad \text{ for } k=0,\dots, K-1" /></a>

We use the K-th iterate as the final reconstruction <a href="https://www.codecogs.com/eqnedit.php?latex=\mathcal{R}_\Theta(g_\eta)&space;=&space;f_K" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathcal{R}_\Theta(g_\eta)&space;=&space;f_K" title="\mathcal{R}_\Theta(g_\eta) = f_K" /></a>. The mappings <a href="https://www.codecogs.com/eqnedit.php?latex=\Lambda_{\theta_k}:&space;X&space;\times&space;X&space;\rightarrow&space;X" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Lambda_{\theta_k}:&space;X&space;\times&space;X&space;\rightarrow&space;X" title="\Lambda_{\theta_k}: X \times X \rightarrow X" /></a> are implemted as convolutional UNets. The parameter <a href="https://www.codecogs.com/eqnedit.php?latex=\Theta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Theta" title="\Theta" /></a> includes all parameters of all subnetworks, i.e. <a href="https://www.codecogs.com/eqnedit.php?latex=\Theta&space;=&space;(\theta_0,&space;\dots)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Theta&space;=&space;(\theta_0,&space;\dots)" title="\Theta = (\theta_0, \dots)" /></a>. We split the provided data into a training, validation and test part. We train our learned iterative method by minimizing the mean squared error between the reconstruction and the groundtruth model on the training set:

<a href="https://www.codecogs.com/eqnedit.php?latex=\hat{\Theta}&space;\in&space;\arg&space;\min_\Theta&space;\frac{1}{n}&space;\sum_{i=1}^n&space;\|&space;\mathcal{R}_\Theta(g_\eta)&space;-&space;f&space;\|_2^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{\Theta}&space;\in&space;\arg&space;\min_\Theta&space;\frac{1}{n}&space;\sum_{i=1}^n&space;\|&space;\mathcal{R}_\Theta(g_\eta)&space;-&space;f&space;\|_2^2" title="\hat{\Theta} \in \arg \min_\Theta \frac{1}{n} \sum_{i=1}^n \| \mathcal{R}_\Theta(g_\eta) - f \|_2^2" /></a>

Here, the mean square error served us as a proxy for the real goal: high accuracy for character recognition.

### Sanity Check

We tried our own sanity check on images from the [STL10 dataset](https://cs.stanford.edu/~acoates/stl10/). STL10 is an image reconginition dataset consisting of natural images of 10 different classes. We used our approximate forward model to simulate a blurred version of STL10. We evaluated our learned iterative model on this blurred natural images. 

![Sanity Check on initial model](images_readme/sanity_check_blur_stl10_13.png "Sanity Check")

*Figure: Sanity Check on intial model fails.*

It was clear, that our initial model would not pass the sanity check.

Due to the sanity check, we have a kind of constrained optimization problem. Maximize the OCR accuracy under the constraint that we have a slight beblurring effect on natural images. In order to tackle this problem, we used a combined training of the provided challenge data with STL10 images. For every training step we checked if the PSNR between the reconstruction <a href="https://www.codecogs.com/eqnedit.php?latex=\mathcal{R}(g_{STL10})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathcal{R}(g_{STL10})" title="\mathcal{R}(g_{STL10})" /></a> and the unblurred image <a href="https://www.codecogs.com/eqnedit.php?latex=f_{STL10}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f_{STL10}" title="f_{STL10}" /></a> was lower than the PSNR btween the blurred image <a href="https://www.codecogs.com/eqnedit.php?latex=g_{STL10}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?g_{STL10}" title="g_{STL10}" /></a> and the unblurred image <a href="https://www.codecogs.com/eqnedit.php?latex=f_{STL10}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f_{STL10}" title="f_{STL10}" /></a>. If our reconstruction was more than 2dB lower in terms of PSNR than the blurred image we added one STL10 image to the current training batch. 

### Down- and Upsampling

The large image size *1460x2360* the usage of large deep learning approaches is limited. There are ways around it, i.e. use invertible neural networks or checkpointing. We tackle this problem by simply downsample the original images to *181x294* (about 8 times downsampling). This was motivated by the fact, that the target area on the e-ink display has a resolution of *200x300*. Downsampling was performed using an average filter. All learned iterative networks were trained on this downsampled images. The final upsampling is performed by a small convolutional neural network, consiting of bilinear upsampling layers and a few convolutional layers. 


## Examples

![Example for step 5 (test data)](images_readme/example_step_5.png "step 5")

*Figure: Example for step 5 (test data).*

![Example for step 5 (test data)](images_readme/example_step_6.png "step 6")

*Figure: Example for step 6 (test data).*

![Example for step 5 (test data)](images_readme/example_step_8.png "step 8")

*Figure: Example for step 8 (test data).*



## Requirements 

* numpy = 1.20.3
* pytorch = 1.9.0 
* pytorch-lightning = 1.3.8
* torchvision = 0.10.0

## Authors

Team University of Bremen, Center of Industrial Mathematics (ZeTeM): 
- Alexander Denker, Maximilian Schmidt, Johannes Leuschner, Sören Dittmer, Judith Nickel, Clemens Arndt, Gael Rigaud, Richard Schmähl
