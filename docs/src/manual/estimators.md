# Estimators 

This page introduces two types of estimators in Microstructure.jl for estimating parameters and quantifying uncertainties: the Markov Chain Monte Carlo (MCMC) sampling method and Monte Carlo dropout using neural networks. These two types of estimators are flexibly parametrized, allowing you to specify sampling options for MCMC and training options for neural networks. 

## MCMC

MCMC methods aim to generate independent samples from the posterior distributions of tissue parameters given certain MRI measurements. You will need to tune the sampler parameters for a specific biophysical model.

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

This module currently includes simple multi-layer perceptrons and training data generation function, which allows supervised training of the MLPs on synthesised data with given training parameter distributions. 

### 1. Specify a network model for your task

```@docs
NetworkArg
```

### 2. Specify training parameters

```@docs
TrainingArg
```

### 3. Train a network with specified network and training parameters

```@docs
training
```

### 4. Apply trained model to your test data

```@docs
test
```

### Other useful functions

Generate task specific MLP and training samples:

```@docs
create_mlp
```

```@docs
generate_samples
```

Training on given model and training samples

```@docs
train_loop!
```