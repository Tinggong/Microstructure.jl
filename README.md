<img src="docs/src/assets/logo_main.png#gh-light-mode-only" width=400>
<img src="docs/src/assets/logo-dark.png#gh-dark-mode-only" width=400>

[![Build Status](https://github.com/Tinggong/Microstructure.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Tinggong/Microstructure.jl/actions/workflows/CI.yml?query=branch%3Amain)

Microstructure.jl is a Julia package for fast and probabilistic microstructure imaging. It features flexible and extendable compartment modeling with diffusion MRI and combined diffusion-relaxometry MRI and provides generic estimators including Markov Chain Monte Carlo (MCMC) sampling methods and Monte Carlo dropout with neural networks. 

Microstructure.jl is under active development, testing and optimization and updates will be shared throughout this process. You are welcome to try it out and provide feedback on any issues encountered. Microstructure.jl has a [documentation website](https://tinggong.github.io/Microstructure.jl/stable/) introducing functional API and features of the package. 

**Updates!** We have a [preprint](https://arxiv.org/abs/2407.06379) if you are interested in knowing more! Companion tutorials of the preprint are avaliable [here](https://tinggong.github.io/Microstructure.jl/dev/guide/0_intro/). 

Gong, T., & Yendiki, A. (2024). Microstructure. jl: a Julia Package for Probabilistic Microstructure Model Fitting with Diffusion MRI. arXiv preprint arXiv:2407.06379. 

### Installation 
To install Microstructure.jl, open Julia and enter the package mode by typing `]`, then add the package:

```julia
julia> ]
pkg> add Microstructure
```

For upgrade to a new version, simply replace the `add` with `up`:
```julia
pkg> up Microstructure
```

### Usage 
This project is licensed under the MIT License, which covers the copyright of the source code.

⚠️ Note: Certain components implement patented methods and may only be used for non-commercial purposes.  
For details, please see the [NOTICE](./NOTICE) file.

### Relationship to Other Packages
Microstructure.jl focuses on tissue microstructure estimation. If you are also interested in fiber orientation and tractography, please check out [Fibers.jl](https://github.com/lincbrain/Fibers.jl). Additional, Microstructure.jl uses I/O functions from Fibers.jl for reading and writing NIfTI image files.

### Acknowledgements
Development of this package is supported by the NIH National Institute of Biomedical Imaging and Bioengineering (grant K99EB037396) and National Institute of Neurological Disorders and Stroke (grants UM1-NS132358, R01-NS119911, R01-NS127353).
