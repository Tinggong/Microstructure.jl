<img src="docs/src/assets/logo_main.png" width=400>

[![Build Status](https://github.com/Tinggong/Microstructure.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Tinggong/Microstructure.jl/actions/workflows/CI.yml?query=branch%3Amain)

Microstructure.jl is a Julia package for fast and probabilistic microstructure imaging. It features flexible and extendable compartment modeling with diffusion MRI and combined diffusion-relaxometry MRI. It provides generic estimators including Markov Chain Monte Carlo (MCMC) sampling methods and Monte Carlo dropout with neural networks. Microstructure.jl is under active development, testing and optimization and updates will be shared throughout this process. You are welcome to try it out and provide feedback on any issues encountered. Microstructure.jl has a developing [documentation website](https://tinggong.github.io/Microstructure.jl/dev/) introducing functional API and features of the package. More tutorials and recommendations will be coming soon. 

### Installation 
To install Microstructure.jl, open Julia and enter the package mode by typing `]`, then add the package:

```julia
julia> ]
(@v1.8) pkg> add Microstructure
```

You can check if your installation is the latest version by typing `status` in the package mode and upgrade to the latest version using `up` in the package mode:

```julia
(@v1.8) pkg> up Microstructure
```

If a newer version isn't being installed using `up`, you can remove current installation and add the latest version by (replace `0.1.4` with latest version number):

```julia
(@v1.8) pkg> rm Microstructure
(@v1.8) pkg> add Microstructure@0.1.4
```

### Relationship to Other Packages
Microstructure.jl focuses on tissue microstructure estimation. If you are also interested in fiber orientation and tractography, please check out [Fibers.jl](https://github.com/lincbrain/Fibers.jl). Additional, Microstructure.jl uses I/O functions from Fibers.jl for reading and writing NIfTI image files.

### Acknowledgements
Development of this package is supported by the NIH National Institute of Neurologic Disorders and Stroke (grants UM1-NS132358, R01-NS119911, R01-NS127353).