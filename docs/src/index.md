# Microstructure.jl

Microstructure.jl is a Julia toolbox aiming at fast and probabilistic microstructure imaging. It supports flexible and extendable compartment modelling with diffusion MRI and combined diffusion-relaxometry MRI. 

## Installation 
To install Microstructure.jl, type ] in Julia to enter package mode and add the package

```julia
julia> ]
(@v1.8) pkg> add Microstructrue
```

or use github link to keep up to date:

```julia
(@v1.8) pkg> add https://github.com/Tinggong/Microstructure.jl.git
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
