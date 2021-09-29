# hdc2021_LGD (In Progress)
Our submission for the HDC2021. 

## Install 

Install the package using:

```
pip install -e .
```
Make sure to have git-lfs installed to pull the weight files for the model.

## Usage 

Prediction on images in a folder can be done using:

```
python deblurrer/main.py path-to-input-files path-to-output-files step
```

## Method

We want to reconstruct the original image $f \in X$ from blurred and noisy data $g_\eta \in X$. The forward operator is given by $\mathcal{A}: X \rightarrow X$. This corresponds to an inverse problem: 
$$ g_\eta = \mathcal{A} f + \eta $$

Our goal is to define an reconstructor $\mathcal{R}_\Theta : X \rightarrow X$ which produces an unblurred image from the blurred $g_\eta$:

$$ f \approx \mathcal{R}_\Theta (g_\eta)$$


Our approach is inspired by the recent trend of combining physical forward models  and learned iterative schemes [(Adler et. al.)](https://arxiv.org/abs/1704.04058). There is also experimental evidence that this approach can work well even with an approximated forward model [(Hauptmann et. al.)](https://arxiv.org/abs/1807.03191).
This approach has two steps:
1. First, we define an approximate forward model for the blurring process.
2. This approximate model is integrated into a learned gradient descent scheme.  

### Approximate Forward Model 

Out-of-focus blur can be modeled by as a linear, position invariant convolution with a circular point-spread-function: 

$$ g_\eta = \mathcal{A} f + \eta = k * f + \eta $$

with 

$$ k(x) =  \left\{\begin{array}{lr}
        \frac{1}{\pi r^2}, & \text{for } \| x \|^2 \le r^2\\
        0, & \text{else }
        \end{array}\right\} $$

This model works well for small blurring levels. For higher blurring levels the average error between the approximate model and the real measurements gets bigger. 

### Learned Gradient Descent

Variational methods try to recover the original image as the minimizer of an objective function 

$$ \hat{f} = \mathcal{R}_\Theta(g\eta) \in \argmin_f d(\mathcal{A}f, g_\eta) + \lambda R(f) $$

where $d:X\times X \rightarrow \mathbb{R}$ is a data fidelity term and $R:X \rightarrow \mathbb{R}$ is a regularizer. In many examples, the regularizer is convex but not differentiable (e.g. TV or l1). In these cases proximal gradient descent can be used to obtain the minimizer: 

$$ f_{k+1} = \text{prox}_{\lambda R}(f_k - \gamma \nabla d(\mathcal{A}f_k, g_\eta)) $$

with a step size $\gamma > 0$. The idea of learned iterative methods is to unroll this iteration for a fixed number of steps and replace the proximal mapping with a convolutional neural network: 

$$ f_{k+1} = \Lambda_{\theta_k}(f_k, \nabla d(\mathcal{A}f_k, g_\eta)) \qquad \text{ for } k=1,\dots, K $$


## Examples

## Requirements 

* numpy = 1.20.3
* pytorch = 1.9.0 
* pytorch-lightning = 1.3.8
* torchvision = 0.10.0

## Authors

Team University of Bremen, Center of Industrial Mathematics (ZeTeM): 
- Alexander Denker, Maximillian Schmidt, Johannes Leuschner, Sören Dittmer, Judith Nickel, Clemens Arndt, Gael Rigaud, Richard Schmähl