There were several julia version upgrades during the development of this package. 

The MCMC sampler was developed and tested using Julia 1.8. But the development and testing of NN estimator was done on at least julia 1.10, due to the requirement of Flux 0.14, one of the dependencies of the NN estimator. (Flux 0.14 allows you to use GPU without having to add device dependent denpendencies, e.g. CUDA.jl or Metal.jl)

While we did not do much changes to MCMC sampler since then, we found out the performance of MCMC sampling based on simulated data differ slightly based on different julia versions. [example figure]


To keep consistency with our previous reports, we revert back to Julia version 1.8 and Microstructure.jl version 1.5 for MCMC sampler. 







