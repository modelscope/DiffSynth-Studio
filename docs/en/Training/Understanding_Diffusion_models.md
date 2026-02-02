# Basic Principles of Diffusion Models

This document introduces the basic principles of Diffusion models to help you understand how the training framework is constructed. To make these complex mathematical theories easier for readers to understand, we have reconstructed the theoretical framework of Diffusion models, abandoning complex stochastic differential equations and presenting them in a more concise and understandable form.

## Introduction

Diffusion models generate clear images or video content through iterative denoising. We start by explaining the generation process of a data sample $x_0$. Intuitively, in a complete round of denoising, we start from random Gaussian noise $x_T$ and iteratively obtain $x_{T-1}$, $x_{T-2}$, $x_{T-3}$, $\cdots$, gradually reducing the noise content at each step until we finally obtain the noise-free data sample $x_0$.

![Image](https://github.com/user-attachments/assets/6471ae4c-a635-4924-8b36-b0bd4d42043d)

This process is intuitive, but to understand the details, we need to answer several questions:

* How is the noise content at each step defined?
* How is the iterative denoising computation performed?
* How to train such Diffusion models?
* What is the architecture of modern Diffusion models?
* How does this project encapsulate and implement model training?

## How is the noise content at each step defined?

In the theoretical system of Diffusion models, the noise content is determined by a series of parameters $\sigma_T$, $\sigma_{T-1}$, $\sigma_{T-2}$, $\cdots$, $\sigma_0$. Where:

* $\sigma_T=1$, corresponding to $x_T$ as pure Gaussian noise
* $\sigma_T>\sigma_{T-1}>\sigma_{T-2}>\cdots>x_0$, the noise content gradually decreases during iteration
* $\sigma_0=0$, corresponding to $x_0$ as a data sample without any noise

As for the intermediate values $\sigma_{T-1}$, $\sigma_{T-2}$, $\cdots$, $\sigma_1$, they are not fixed and only need to satisfy the decreasing condition.

At an intermediate step, we can directly synthesize noisy data samples $x_t=(1-\sigma_t)x_0+\sigma_t x_T$.

![Image](https://github.com/user-attachments/assets/e25a2f71-123c-4e18-8b34-3a066af15667)

## How is the iterative denoising computation performed?

Before understanding the iterative denoising computation, we need to clarify what the input and output of the denoising model are. We abstract the model as a symbol $\hat \epsilon$, whose input typically consists of three parts:

* Time step $t$, the model needs to understand which stage of the denoising process it is currently in
* Noisy data sample $x_t$, the model needs to understand what data to denoise
* Guidance condition $c$, the model needs to understand what kind of data sample to generate through denoising

Among these, the guidance condition $c$ is a newly introduced parameter that is input by the user. It can be text describing the image content or a sketch outlining the image structure.

The model's output $\hat \epsilon(x_t,c,t)$ approximately equals $x_T-x_0$, which is the direction of the entire diffusion process (the reverse process of denoising).

Next, we analyze the computation occurring in one iteration. At time step $t$, after the model computes an approximation of $x_T-x_0$, we calculate the next $x_{t-1}$:

$$
\begin{aligned}
x_{t-1}&=x_t + (\sigma_{t-1} - \sigma_t) \cdot \hat \epsilon(x_t,c,t)\\
&\approx x_t + (\sigma_{t-1} - \sigma_t) \cdot (x_T-x_0)\\
&=(1-\sigma_t)x_0+\sigma_t x_T + (\sigma_{t-1} - \sigma_t) \cdot (x_T-x_0)\\
&=(1-\sigma_{t-1})x_0+\sigma_{t-1}x_T
\end{aligned}
$$

Perfect! It perfectly matches the noise content definition at time step $t-1$.

> (This part might be a bit difficult to understand. Don't worry; it's recommended to skip this part on first reading without affecting the rest of the document.)
>
> After completing this somewhat complex formula derivation, let's consider a question: why should the model's output approximately equal $x_T-x_0$? Can it be set to other values?
>
> Actually, Diffusion models rely on two definitions to form a complete theory. From the above formulas, we can extract these two definitions and derive the iterative formula:
>
> * Data definition: $x_t=(1-\sigma_t)x_0+\sigma_t x_T$
> * Model definition: $\hat \epsilon(x_t,c,t)=x_T-x_0$
> * Derived iterative formula: $x_{t-1}=x_t + (\sigma_{t-1} - \sigma_t) \cdot \hat \epsilon(x_t,c,t)$
>
> These three mathematical formulas are complete. For example, in the previous derivation, substituting the data definition and model definition into the iterative formula yields $x_{t-1}$ that matches the data definition.
>
> These are two definitions built on Flow Matching theory, but Diffusion models can also be implemented with other definitions. For example, early models based on DDPM (Denoising Diffusion Probabilistic Models) have their two definitions and derived iterative formulas as:
>
> * Data definition: $x_t=\sqrt{\alpha_t}x_0+\sqrt{1-\alpha_t}x_T$
> * Model definition: $\hat \epsilon(x_t,c,t)=x_T$
> * Derived iterative formula: $x_{t-1}=\sqrt{\alpha_{t-1}}\left(\frac{x_t-\sqrt{1-\alpha_t}\hat \epsilon(x_t,c,t)}{\sqrt{\sigma_t}}\right)+\sqrt{1-\alpha_{t-1}}\hat \epsilon(x_t,c,t)$
>
> More generally, we describe the derivation process of the iterative formula using matrices. For any data definition and model definition:
>
> * Data definition: $x_t=C_T(x_0,x_T)^T$
> * Model definition: $\hat \epsilon(x_t,c,t)=C_T^{[\epsilon]}(x_0,x_T)^T$
> * Derived iterative formula: $x_{t-1}=C_{t-1}(C_t,C_t^{[\epsilon]})^{-T}(x_t,\hat \epsilon(x_t,c,t))^T$
>
> Where $C_t$ and $C_t^{[\epsilon]}$ are $1\times 2$ coefficient matrices. It's not difficult to see that when constructing the two definitions, the matrix $(C_t,C_t^{[\epsilon]})^T$ must be invertible.
>
> Although Flow Matching and DDPM have been widely verified by numerous pre-trained models, this doesn't mean they are optimal solutions. We encourage developers to design new Diffusion model theories for better training results.

## How to train such Diffusion models?

After understanding the iterative denoising process, we next consider how to train such Diffusion models.

The training process differs from the generation process. If we retain multi-step iterations during training, the gradient would need to backpropagate through multiple steps, bringing catastrophic time and space complexity. To improve computational efficiency, we randomly select a time step $t$ for training.

The following is pseudocode for the training process:

> Obtain data sample $x_0$ and guidance condition $c$ from the dataset
>
> Randomly sample time step $t\in(0,T]$
>
> Randomly sample Gaussian noise $x_T\in \mathcal N(O,I)$
>
> $x_t=(1-\sigma_t)x_0+\sigma_t x_T$
>
> $\hat \epsilon(x_t,c,t)$
>
> Loss function $\mathcal L=||\hat \epsilon(x_t,c,t)-(x_T-x_0)||_2^2$
>
> Backpropagate gradients and update model parameters

## What is the architecture of modern Diffusion models?

From theory to practice, more details need to be filled in. Modern Diffusion model architectures have matured, with mainstream architectures following the "three-stage" architecture proposed by Latent Diffusion, including data encoder-decoder, guidance condition encoder, and denoising model.

![Image](https://github.com/user-attachments/assets/43855430-6427-4aca-83a0-f684e01438b1)

### Data Encoder-Decoder

In the previous text, we consistently referred to $x_0$ as a "data sample" rather than an image or video because modern Diffusion models typically don't process images or videos directly. Instead, they use an Encoder-Decoder architecture model, usually a VAE (Variational Auto-Encoders) model, to encode images or videos into Embedding tensors, obtaining $x_0$.

After data is encoded by the encoder and then decoded by the decoder, the reconstructed content is approximately consistent with the original, with minor errors. So why process on the encoded Embedding tensor instead of directly on images or videos? The main reasons are twofold:

* Encoding compresses the data simultaneously, reducing computational load during processing.
* Encoded data distribution is more similar to Gaussian distribution, making it easier for denoising models to model the data.

During generation, the encoder part doesn't participate in computation. After iteration completes, the decoder part decodes $x_0$ to obtain clear images or videos. During training, the decoder part doesn't participate in computation; only the encoder is used to compute $x_0$.

### Guidance Condition Encoder

User-input guidance conditions $c$ can be complex and diverse, requiring specialized encoder models to process them into Embedding tensors. According to the type of guidance condition, we classify guidance condition encoders into the following categories:

* Text type, such as CLIP, Qwen-VL
* Image type, such as ControlNet, IP-Adapter
* Video type, such as VAE

> The model $\hat \epsilon$ mentioned in the previous text refers to the entirety of all guidance condition encoders and the denoising model. We list guidance condition encoders separately because these models are typically frozen during Diffusion training, and their output values are independent of time step $t$, allowing guidance condition encoder computations to be performed offline.

### Denoising Model

The denoising model is the true essence of Diffusion models, with diverse model structures such as UNet and DiT. Model developers can freely innovate on these structures.

## How does this project encapsulate and implement model training?

Please read the next document: [Standard Supervised Training](/docs/en/Training/Supervised_Fine_Tuning.md)