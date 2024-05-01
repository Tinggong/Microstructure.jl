# Estimators 

This page introduces two type of estimators implemented in Microstructure.jl for estimating parameters and uncertainties: the Markov Chain Monte Carlo (MCMC) sampling method and Monte Carlo dropout using neural networks. While MCMC estimator will take longer computation time, it is recommend for more accrute parameter estimation. The performance of neural network estimator will be closely linked to the parameter distributions in the training samples. Currently, function to generate uniform parameter distributions is provided, which may not be the optimized solutions for every model. However, if you are interested in studying how training samples affects estimation accuracy, you are welcome to try it out and you can generate samples use other distributions. 

## MCMC

### Define a sampler for your model

```@docs
Sampler
```

### Define a noise model

```@docs
Noisemodel
```

### Run MCMC on your model and data

```@docs
mcmc!
```

Function mcmc! runs on single thread and suitable for testing sampler parameters and inspecting chains for small dataset. After optimizing sampler parameters, if you are processing datasets with many voxels, use the threading function for multi-threads processing. Refer to multi-threads page for more details.

## Neural Networks

### Specify a network model for your task

```@docs
NetworkArg
```

### Specify training parameters

```@docs
TrainingArg
```

### Prepare network and data for training

```@docs
prepare_training
```

"prepare_training" calls two functions to generate task specific MLP and training samples:

```@docs
create_mlp
```

```@docs
generate_samples
```

### 4. Training on generated training samples

```@docs
train_loop!
```

### 5. test on you data

```@docs
test
```


