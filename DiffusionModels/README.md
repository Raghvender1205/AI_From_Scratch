# Diffusion Models
`Diffusion Models` are a class of generative models that learn to create data (`images`, `audio` etc) by reversing a `noising` process.
They consist of 2 processes
1. Forward process (diffusion) -> In this, noise is gradually added to the real data
2. Reverse Process (Generation) -> In this, the model learns to denoise and generate realistic samples from noise.

Let's denote
- $x_0$: Original Data sample
- $x_t$: Noisy version of the data at time step $t$
- $T$: Total number of steps 

## 1. Forward Diffusion Process (Adding Noise)
This is a `Markov process` that slowly add Gaussian noise

$$q(x_t|x_{t-1}) = N(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_tI)$$

Here

* $\beta_t \in (0, 1)$ is a small noise variance at time $t$
* The total process from $x_0$ to $x_T$ (pure noise) is:

$$q(x_{1:T}|x_0) = \prod_{t=1}^T q(x_t|x_{t-1})$$

By composing, we can directly sample $x_t$ from $x_0$

$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\mathbf{\epsilon}, \quad \mathbf{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$$

where `ᾱₜ = ∏ₛ₌₁ᵗ (1 - βₛ)`


## 2. Reverse Process (Denoising)
The model learns the reverse `conditional` probability:

$$p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$$

We model this as `Gaussian` as well

$p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1};\mu_\theta(\mathbf{x}_t, t), \sum_{\theta}(\mathbf{x}_t, t))$

Instead of predicting $\mu_{\theta}$ and $\sum_{\theta}$ directly, [DDPM](https://arxiv.org/abs/2006.11239) reformulates the problem to predict the noise $\epsilon$.

$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{\mathbf{x}_0, \mathbf{\epsilon}, t} \left[ \|\mathbf{\epsilon} - \mathbf{\epsilon}_\theta(\mathbf{x}_t, t)\|^2 \right]
$$

This becomes `MSE` loss, where the network tries to predict the exact noise added at step t.

## 3. Generation (Sampling)
Once trained, we start from `pure noise` $\mathbf{x}_T ∼ \mathcal{N}(0, I)$ and denoise it step by step:

$$\mathbf{x_{t-1}} = \mu_{\theta}(\mathbf{x_t}, t) + \sigma_tz , z ∼  \mathcal{N}(0, I)$$

At each step, the model predicts the clean version from the noisy version until $x_0$ is obtained.

## Training Objective (Variational Low Bound)
The Variational Low bound or `Evidence Lower Bound (ELBO)` is a concept in variational inference, particularly used in `Variational AutoEncoders (VAE)` and Diffusion Models.

It provides a tracable way to `approximate` the intracable true log-likelihood of the data.

Now, the model optimizes the `ELBO` or Variational Lower bound on the `negative log-likelihood`

$$
\log p(\mathbf{x}_0) \ge \mathbb{E}_q \left[ \log \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \right]
$$

This decomposes into terms of measuring how well the model matches the reverse process to the true distribution.

Diffusion models are trained by minimizing a loss function that corresponds to the negative `ELBO` which is 

$$log p_{\theta}(\mathbf{x}_0) \ge ELBO(\theta)$$

$$
\text{ELBO} = \mathbb{E}_q \left[ \sum_{t=1}^T \text{KL}(q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) || p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)) - \log p_\theta(\mathbf{x}_T) \right]
$$

So, we are minimizing this:
$$
-ELBO(\theta)
$$

### TODO
* Denoising Diffusion (DDPM)
* Stable Diffusion
* Rectified flow