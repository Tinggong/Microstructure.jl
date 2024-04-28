# Multi threads

Multi-threads processing is recommended when using MCMC estimation. The neural network estimators are relatively fast and take only minutes training on CPU. 

### Start julia in terminal with multi-threads 
```terminal
~ % julia --threads auto
```
You can also set enviornment variable by adding `export JULIA_NUM_THREADS=auto` in your bash profile, which will use multi-threads automatically when you start julia.

### Multi-threads for MCMC estimation

```@docs
threading
```

Function threading calls pre_allocate and mcmc! for multi-threads processing. When provided sampler is a Tuple containing two Samplers, it uses a two-stage MCMC to get final estimates.

```@docs
pre_allocate
```