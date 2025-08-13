# What is Microstructure.jl for?

Microstructure.jl is a Julia toolbox aiming at fast and probabilistic microstructure imaging with diffusion and combined diffusion-relaxometry MRI. Microstructure imaging enables the estimation of biologically meaningful cellular parameters from MRI data. This is achieved by simplifying the tissue within a voxel into multiple compartments representing different cellular structures, e.g. axons and cell bodies etc. Each tissue compartment has unique features that affect the MR signals under a measurement protocol, thus allowing their estimation through solving an inverse problem.

Since MRI measurements are typically noisy and exhibit varying sensitivities to tissue features, point estimation methods, which provide a single estimate of each tissue parameter, are often insufficient for understanding the meaningfulness of the estimated tissue parameters. To address this issue, many previous studies have investigated more extensive methods such as Markov Chain Monte Carlo (MCMC) sampling, to sample the posterior distributions of tissue parameters and thereby obtain the probability of the estimates. However, such methods have seen limited applications in neuroimaging studies due to the significantly longer computation time required for analyzing whole-brain datasets. 

Microstructure.jl aims to reduce the computation time required for probabilistic microstructure imaging by leveraging Julia's high performance design. It does not directly address limitations in microstructure modelling itself but aims to serve as a flexible tool to better investigate modelling assumptions and model performance. General recommendations for model fitting will be shared after testing and optimization.  

If you are interested, please try it out! The getting started page includes the minimal steps for beginning with your MRI dataset. Visit manual and upcoming tutorials for more feature demonstrations, recommendations and references!

### Feature Summary 
- Combined diffusion-relaxometry compartment modelling
- Flexible in creating models and adjusting assumptions
- Generic MCMC sampling with parallel computing 
- Generic and fast neural network estimators with uncertainty quantification
- Compatible with the probabilistic programming language [Turing.jl](https://turinglang.org/dev/)

### Installation 
To install Microstructure.jl, type ] in Julia to enter package mode and add the package:

```julia
julia> ]
pkg> add Microstructure
```

Microstructure.jl is under active development and is frequently updated. To ensure you have the latest version, use the following command in the package mode:

```julia
pkg> up Microstructure
```

### Relationship to Other Packages
Microstructure.jl focuses on tissue microstructure estimation. If you are also interested in fiber orientation and tractography, please check out [Fibers.jl](https://github.com/lincbrain/Fibers.jl). Microstructure.jl also uses I/O functions from Fibers.jl for reading and writing mri image files. 

### Acknowledgements
Development of this package is supported by the NIH National Institute of Neurological Disorders and Stroke (grants UM1-NS132358, R01-NS119911, R01-NS127353).
