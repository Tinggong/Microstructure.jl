# Microstructure.jl

[![Build Status](https://github.com/Tinggong/Microstructure.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Tinggong/Microstructure.jl/actions/workflows/CI.yml?query=branch%3Amain)

Microstructure.jl is a Julia toolbox aiming at fast and probabilistic microstructure imaging. It features flexible and extendable compartment modelling with diffusion MRI and combined diffusion-relaxometry MRI. Microstructure.jl is currently under testing and optimization. If you have any questions using it, please email me at tgong1@mgh.harvard.edu. 

## Installation 
To install Microstructure.jl and keep up to date, you can use the repository link. Type ] in Julia to enter package mode and add the link:

```julia
julia> ]
(@v1.8) pkg> add https://github.com/Tinggong/Microstructure.jl.git
```

Once package registry is approved, you will be able to install it by using the package name:

```julia
(@v1.8) pkg> add Microstructrue
```

## Usage 
Documentation demonstrating different features of the package can be found on the [documentation website](https://tinggong.github.io/Microstructure.jl/dev/)
The documentation website is under construction and will be frequently updated.

## Relationship to Other Packages
Microstructure.jl focuses on tissue microstructure estimation. If you are also interested in fiber orientation and tractography, please check out FreeSurfer.jl. Microstructure.jl also uses I/O functions from FreeSurfer.jl for reading and writing NIfTI image files. 
