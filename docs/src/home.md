# Microstructure.jl

Microstructure.jl is a Julia toolbox aiming at fast and probabilistic microstructure imaging. It supports flexible and extendable compartment modelling with diffusion MRI and combined diffusion-relaxometry MRI. 

## Installation 
To install Microstructure.jl, use the Julia package manager:

```julia
julia> using Pkg
julia> add Microstructrue
```

or use github link to keep up to date:

```julia
julia> using Pkg
julia> add https://github.com/Tinggong/Microstructure.jl.git
```

## Feature Summary 
- combined diffusion-relaxometry compartment modelling
- flexible in creating models and adjusting assumptions
- generic mcmc estimator
- parallel computing 
- quality checking  

## Relationship to Other Packages
Microstructure.jl focuses on tissue microstructure estimation. If you are also interested in fiber orientation and tractography, please check out FreeSurfer.jl. Microstructure.jl also uses I/O functions from FreeSurfer.jl for reading and writing mri image files. 

## Citation
