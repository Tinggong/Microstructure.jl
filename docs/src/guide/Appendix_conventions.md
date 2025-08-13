# [Introduction to Microstructure.jl] (@id guide-conventions)

Conventions used for parameter estimation

1. Approximating b0 signal as the mean from b=0 measurements 

    (b0 is not estimated but treated as a known parameter during optimization)

    measurements from b=0 (especially mean b=0 image) has higher SNR than other images, and is often used as normalizing factor for modelling the diffusion signal decay. While b=0 signal can be estimated, this package generally assumes mean b=0 represents well the b=0 signals thus is considered known parameters. 


1. Single-TE data and model

We can treat b=0 as a measurement and a parameter to estimate in the model, which is usually the case in linear estimation methods. Since b=0 signal has no unit and the intensity range can vary substantially, creating difficulties for selecting starting point or constraining parameters to plausible range, measurements normalized by b=0, thus start from 1, are often used in other advanced optimization methods. In this case, the number of parameters and the number of measurements is reduced by 1. 

S(b) = S(b=0) .* (fin * Ain(b) + (1-fin)Aec(b))

In the MCMC, Microstructure.jl still expect the input from a full protocol including the b=0 and measurements of 1, but it considers only the loglikehood of non-b0 measurements. 


2. Multi-TE data and model

In MTE data, 
