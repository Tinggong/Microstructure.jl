# Estimators 

This page introduces two type of estimators implemented in Microstructure.jl for estimating parameters and uncertainties, including the Markov Chain Monte Carlo (MCMC) sampling method and Monte Carlo dropout using neural networks.

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


