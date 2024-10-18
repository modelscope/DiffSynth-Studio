# Schedulers

Schedulers control the entire denoising (or sampling) process of the model. When loading the Pipeline, DiffSynth automatically selects the most suitable schedulers for the current Pipeline, requiring no additional configuration.

The supported schedulers are:

- **EnhancedDDIMScheduler**: Extends the denoising process introduced in the Denoising Diffusion Probabilistic Models (DDPM) with non-Markovian guidance.

- **FlowMatchScheduler**: Implements the flow matching sampling method introduced in Stable Diffusion 3.

- **ContinuousODEScheduler**: A scheduler based on Ordinary Differential Equations (ODE).