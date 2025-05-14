This contains code and blogs regarding Diffusion Models

# Diffusion Models
`Diffusion Models` are a class of generative models that learn to create data (`images`, `audio` etc) by reversing a `noising` process.
They consist of 2 processes
1. Forward process (diffusion) -> In this, noise is gradually added to the real data
2. Reverse Process (Generation) -> In this, the model learns to denoise and generate realistic samples from noise.

Let's denote
- $\mathbf{x}_0$: Original Data sample
- $\mathbf{x}_t$: Noisy version of the data at time step $t$
- $T$: Total number of steps 

## 1. Forward Diffusion Process (Adding Noise)
This is a `Markov process` that slowly add Gaussian noise

$q(\mathbf{x}_t|\mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t\mathbf{I})$

Here

* $\beta_t \in (0, 1)$ is a small noise variance at time $t$
* The total process from $\mathbf{x}_0$ to $\mathbf{x}_T$ (pure noise) is:

$q(\mathbf{x}_{1:T}|\mathbf{x}_0) = \prod_{t=1}^T q(\mathbf{x}_t|\mathbf{x}_{t-1})$

By composing, we can directly sample $\mathbf{x}_t$ from $\mathbf{x}_0$

$\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\mathbf{\epsilon}, \quad \mathbf{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$

where $\bar{\alpha}_t = \prod_{s=1}^t (1-\beta_s)$

## 2. Reverse Process (Denoising)
The model learns the reverse `conditional` probability:

$p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$



### TODO
* Denoising Diffusion (DDPM)
* Stable Diffusion
* Rectified flow