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

During training, the model learns to `reverse` the diffusion process in order to generate new data. Starting with pure `Gaussian Noise` $p(\textbf{x}_{T}) := \mathcal{N}(\textbf{x}_T, \textbf{0}, \textbf{I})$, model learns the joint distribution $p_\theta(\textbf{x}_{0:T})$ as:

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

Regardless of the values choosen, 