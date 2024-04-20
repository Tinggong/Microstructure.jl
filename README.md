<img src="docs/src/assets/logo.png" width=400>

[![Build Status](https://github.com/Tinggong/Microstructure.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Tinggong/Microstructure.jl/actions/workflows/CI.yml?query=branch%3Amain)

Microstructure.jl is a Julia toolbox designed for fast and probabilistic microstructure imaging. It offers flexible and extendable compartment modeling with diffusion MRI and combined diffusion-relaxometry MRI. The toolbox includes generic estimators such as Markov Chain Monte Carlo (MCMC) sampling methods and Monte Carlo dropout with neural networks. Microstructure.jl is currently undergoing testing and optimization. If you have any questions or encounter any issues while using it, please feel free to email me at tgong1@mgh.harvard.edu.

### Installation 
To install Microstructure.jl, open Julia and enter the package mode by typing `]`, then add the package:

```julia
julia> ]
(@v1.8) pkg> add Microstructrue
```

Microstructure.jl is under active development and is frequently updated. To ensure you have the latest version, use the following command in the package mode:

```julia
(@v1.8) pkg> up Microstructrue
```

### Usage
Documentation demonstrating various features of the package can be found on the [documentation website](https://tinggong.github.io/Microstructure.jl/dev/). Please note that the documentation website is currently under construction and will be frequently updated.

### Relationship to Other Packages
Microstructure.jl focuses on tissue microstructure estimation. If you are also interested in fiber orientation and tractography, consider exploring FreeSurfer.jl. Additional, Microstructure.jl uses I/O functions from FreeSurfer.jl for reading and writing NIfTI image files.
