# Estimators 

This page introduces two types of estimators implemented in Microstructure.jl for estimating parameters and uncertainties: the Markov Chain Monte Carlo (MCMC) sampling method and Monte Carlo dropout using neural networks. While MCMC estimator will take longer computation time, it is recommended for more accurate parameter estimation comparing with the neural network estimator currently implemented. The performance of neural network estimator will be closely linked to the parameter distributions in the training samples. Currently, function to generate uniform parameter distributions is provided, which may not be the optimized solutions for every model. However, if you are interested in studying how training samples affect estimation accuracy, you are welcome to try it out and you can also generate samples use other distributions. 

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

This module currently includes simple multi-layer perceptrons and training data generation function, which allows supervised training of the MLPs on synthesised data with uniform parameter distributions. 

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

### Training on generated training samples

```@docs
train_loop!
```

### Test on you data

```@docs
test
```


