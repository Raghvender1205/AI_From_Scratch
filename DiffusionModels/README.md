# Introduction to Diffusion Models
`Diffusion models` are generative models as they generate data similar to the training data. They work by `destroying` training data through successive `addition` of `Gaussian Noise` and then <b>learning to recover</b> by reversing the `noise` process. After the training, we can pass randomly sampled noise and generate data through the learned `denoising` process.

<img src="https://www.assemblyai.com/blog/content/images/2022/05/image-10.png"/>

More specifically, a Diffusion Model is a `latent variable` model which maps to the latent space using a fixed `Markov chain`. This chain gradually adds noise to the data in order to obtain the `approximate posterior` $q(\textbf{x}_{1:T}|\textbf{x}_0)$, where $\textbf{x}_1, ... , \textbf{x}_T$ are latent variables with same dimensionality as `x0`.

<img src="https://www.assemblyai.com/blog/content/images/size/w1000/2022/05/image.png"/>

Ultimately, the image is transformed to pure `Gaussian Noise`. The goal of the diffusion model is to learn the `denoising` process i.e `training` $p_\theta(x_{t-1}|x_t)$ by traversing backwards along the markov chain.

<img src="https://www.assemblyai.com/blog/content/images/size/w1000/2022/05/image-1.png"/>

## Working of Diffusion Models
A Diffusion model consists of a `forward process` in which a generated image is progressively `noised` and a `reverse process` or `denoising process` in which noise is transformed back into a sample from the `target` distribution

The sampling chain transistions in the `forward process` can be set to conditional `Gaussians` when the noise level is `low`. Combining this with the `Markov` assumption leads to a simple `parameterization` of the `forward process`.

<img src="https://www.assemblyai.com/blog/content/images/size/w1000/2022/06/1_v2.png"/>

where $\beta_1, ..., \beta_T$ is a variance schedule which ensures that $x_T$ is nearly an `isotropic Gaussian` for sufficiently large T.

<img src="https://www.assemblyai.com/blog/content/images/size/w1000/2022/05/image.png"/>

During training, the model learns to `reverse` the diffusion process in order to generate new data. Starting with pure `Gaussian Noise` 
$p(\textbf{x}_{T}) := \mathcal{N}(\textbf{x}_T, \textbf{0}, \textbf{I})$
model learns the joint distribution 
$p_\theta(\textbf{x}_{0:T})$ as:

<img src="https://www.assemblyai.com/blog/content/images/size/w1000/2022/06/2_v2.png"/>

where time dependent parameters of the `Gaussian` transitions are learned. The `Markov` formulation asserts that a given `reverse` diffusion transition function depends only on the previous timestep.

<img src="https://www.assemblyai.com/blog/content/images/size/w1000/2022/05/3.png"/>
<h4>Reverse Process</h4>
<img src="https://www.assemblyai.com/blog/content/images/size/w1000/2022/05/image-1.png">

## Training
A diffusion model is trained by finding `reverse Markov transitions` that maximize the likelihood of the data. Training equivalently consists of minimizing the variational upper bound on the `negative log likelihood`.
<img src="https://www.assemblyai.com/blog/content/images/size/w1000/2022/05/4-1.png">

We seek to rewrite the $L_{vlb}$ in terms of `Kullback-Leibler (KL) Divergences`. The KL Divergence is an asymmetric statiscal distance of how much on probability distribution `P` differs from a reference distribution `Q`. We are interested in formulating $L_{vlb}$ in terms of KL divergences because the distributions in the `Markov Chain` are `Gaussians` and the <b>KL divergence b/w Gaussians has a closed form</b>

## KL Divergence
The mathematical form of `KL Divergence` for continous distributions is
<img src="https://www.assemblyai.com/blog/content/images/2022/05/image-12.png"/>

### Casting $L_{vlb}$ in terms of KL Divergence
It is possible to rewrite $L_{vlb}$ almost completely in terms of KL divergences.
<img src="https://www.assemblyai.com/blog/content/images/size/w1000/2022/05/5-1.png"/>

where
<img src="https://www.assemblyai.com/blog/content/images/size/w1000/2022/05/6-2.png">

Conditioning the `forward process` posterior on `x0` in $L_{t-1}$ results in a tracable form that leads to all <b>KL divergences being comparisons b/w Gaussians</b>. This means that the divergences can be exactly calculated using `closed-form` expressions rather than `Monte-Carlo` estimates.

### Forward Process and $L_T$
For forward process, we must define the `variance schedule`. We set them to be `time-dependent constants`, ignoring the fact that they can be learned.

Regardless of the values choosen, the variance schedule is fixed results in $L_{T}$ becoming a constant w.r.t to our set of `learnable parameters` allowing us to ignore it.

### Reverse Process and $L_{1:T-1}$
As we recall the `reverse Markov` transitions as a Gaussian

<img src="https://www.assemblyai.com/blog/content/images/size/w1000/2022/05/7.png"/>

Now, we must define the functional forms of $\pmb{\mu}_\theta$ or $\pmb{\Sigma}_\theta$
We simply set,

<img src="https://www.assemblyai.com/blog/content/images/size/w1000/2022/05/8.png">

In this, we assume that the multivariate `Gaussian` is a product of <b>independent gaussians </b> with identical variance, which can change with time. We set these `variances to be equivalent to our forward process variance schedule`.

Now we have $\pmb{\Sigma}_\theta$ as

<img src="https://www.assemblyai.com/blog/content/images/size/w1000/2022/05/9.png"/>

which allows us to transform 
<img src="https://www.assemblyai.com/blog/content/images/size/w1000/2022/05/6.2.png"/>
into
<img src="https://www.assemblyai.com/blog/content/images/size/w1000/2022/05/10-1.png"/>

where the first term in the difference is a linear combination of $x_t$ and $x_0$ that depends on the variance schedule $\beta_t$. The significance is that most straightforward parameterization of $\mu_\theta$ to predict the `noise` component at any given timestamps yields better results. Let

<img src="https://www.assemblyai.com/blog/content/images/size/w1000/2022/05/image-16.png"/>

where 
<img src="https://www.assemblyai.com/blog/content/images/size/w1000/2022/05/image-17.png">

<b>This leads to a loss function</b> which provides more stable training and better results.

<img src="https://www.assemblyai.com/blog/content/images/size/w1000/2022/05/image-18.png"/>

## Network Architecture
The Diffusion models are implemented with `UNet` like architectures. This is an architecture of the `UNet Model`
<img src="https://www.assemblyai.com/blog/content/images/size/w1000/2022/05/image-19.png"/>

### Reverse Process Decoder and $L_0$
The path along the `reverse` process consists of many transformations under `continous conditional Gaussian distributions`. At the end of the process, we are trying to generate an image containing `INT` pixels. So, we must devise a way to obtain `discrete (log) likelihoods` for each possible pixel value across all pixels.

This can be done by setting the last transition in `reverse` diffusion chain to an `independent discrete decoder`. To determine the likelihood of a given image $x_0$ given $x_1$, we impose independence b/w the data dimensions.

<img src="https://www.assemblyai.com/blog/content/images/size/w1000/2022/05/12.png"/>

where `D` is the dimensionality of the data and the superscript `i` indicates the extraction of `one` coordinate. The goal now is to determine how `likely` each integer value is for a given pixel given the distribution across possible values for the corresponding pixel in the slightly noised image at time $t = 1$

<img src="https://www.assemblyai.com/blog/content/images/size/w1000/2022/05/13-1.png">

where pixel distributions at $t=1$ are derived from `multivariate Gaussian` whose `covariance matrix` allows us to split the distribution into a product of univariate Gaussians.

<img src="https://www.assemblyai.com/blog/content/images/size/w1000/2022/05/11-2.png"/>

We assume that images consist of integers in $0, 1, ..., 255$ (RGB images) which have been scaled linearly $[-1, 1]$. The probability of `pixel` value $x$, given the univariate Gaussian distribution of the corressponding pixel in $x_1$, is <b>area under that univariate Gaussian distribution within the bucket centered at x</b>

Given at time $t = 0$ pixel value of each pixel, the value of $p_\theta(x_0 | x_1)$ is simply their product. So, the equation becomes
<img src="https://www.assemblyai.com/blog/content/images/size/w1000/2022/05/14-1.png"/>

where
<img src="https://www.assemblyai.com/blog/content/images/size/w1000/2022/05/15.png"/>
and 
<img src="https://www.assemblyai.com/blog/content/images/size/w1000/2022/05/15.png"/>

Now, given this equation for $p_\theta(x_0 | x_1)$, we can calculate the final term of $L_{vlb}$ which is not formulated as a KL Divergence

<img src="https://www.assemblyai.com/blog/content/images/size/w1000/2022/05/6.1-1.png"/>

So, the `training` and `sampling` algorithms for Diffusion Model can be derived as

<img src="https://www.assemblyai.com/blog/content/images/size/w1000/2022/05/image-20.png"/>