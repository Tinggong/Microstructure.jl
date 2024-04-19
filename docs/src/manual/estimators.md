# Estimators 

This page introduces two type of estimators implemented in Microstructure.jl for estimating parameters and uncertainties, including the Markov Chain Monte Carlo (MCMC) sampling method and Monte Carlo dropout using neural networks.

## MCMC

### 1. Define a sampler for your model

```@docs
Sampler
```

### 2. Define a noise model

```@docs
Noisemodel
```

### 3. Run MCMC on your model and data

```@docs
mcmc!
```

## Neural Networks

### 1. Specify a network model for your task

```@docs
NetworkArg
```

### 2. Specify training parameters

```@docs
TrainingArg
```

### 3. Prepare network and data for training

```@docs
prepare_training
```

### 4. Training on generated training samples

```@docs
train_loop!
```

### 5. test on you data

```@docs
test
```


