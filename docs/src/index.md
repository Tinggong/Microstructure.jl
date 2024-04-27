# Microstructure.jl

Microstructure.jl is a Julia toolbox aiming at fast and probabilistic microstructure imaging. It supports flexible and extendable compartment modelling with diffusion MRI and combined diffusion-relaxometry MRI. 

### Installation 
To install Microstructure.jl, type ] in Julia to enter package mode and add the package:

```julia
julia> ]
(@v1.8) pkg> add Microstructure
```

Microstructure.jl is under active development and is frequently updated. To ensure you have the latest version, use the following command in the package mode:

```julia
(@v1.8) pkg> up Microstructure
```

### Feature Summary 
- Combined diffusion-relaxometry compartment modelling
- Flexible in creating models and adjusting assumptions
- Generic MCMC and neural network estimators
- Parallel computing 

### Relationship to Other Packages
Microstructure.jl focuses on tissue microstructure estimation. If you are also interested in fiber orientation and tractography, please check out Fibers.jl. Microstructure.jl also uses I/O functions from Fibers.jl for reading and writing mri image files. 
