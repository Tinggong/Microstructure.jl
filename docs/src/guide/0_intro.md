# [Introduction to Microstructure.jl] (@id guide-intro)

This guide accompanies the demonstrations introduced in our [preprint] (https://arxiv.org/abs/2407.06379) and provides an overview of how to use Microstructure.jl. We walk through the package from its building blocks to applying estabilished literature models on publicly available datasets. Hope this guide helps you understand the package and get started with using it in your own work. 

If you find the package useful for your research, please consider citing our preprint:

Gong, T., & Yendiki, A. (2024). Microstructure. jl: a Julia Package for Probabilistic Microstructure Model Fitting with Diffusion MRI. arXiv preprint arXiv:2407.06379.

## Content 

1. **Signal Simulation for Protocol Design**  
   
   Generating signals from compartment models using specified imaging protocols and tissue properties is an essential building block of microstructure modeling. This functionality is key for understanding how the data should be collected to improve sensitivity to the tissue properties of interest. We demonstrate this for the task of designing an imaging protocol for estimating axon diameter index with the *Cylinder*  compartment. 

   See [Sensitivity Range of Axon Diameter Index](1_sensitivity_range.md) for details.

2. **Two-Stage MCMC for Model Fitting**  
   
   Evaluating the quality of fit and the posterior distribution of parameter estimates is key to refining model fitting assumptions. See how we develop a [Two-Stage MCMC](2_two_stage_MCMC.md) approach for improving the estimating the axon diameter index in the ex vivo tissue.

3. **Fitting Synthetic Data for Performance Evaluation**  
   
   Synthetic datasets with known ground-truth parameters are essential for evaluating the accuracy and precision of microstructure model estimates for different data acquisition protocols. See the [Fitting Evaluation](3_fitting_eval.md) page for demonstration of this process using the two-state MCMC approach for estimating the axon diameter index.

4. **Applying Literature Models to Public Data**  

   We apply established microstructure models on publicly available datasets. This tutorial demonstrates how to use neural network estimators for WM and GM models and adapt the prior distributions in training datasets. This currently includes:  
   - The [SMT model in white matter](4_smt.md) using HCP data  
   - The [SANDI model in gray matter](5_sandi.md) using Connectome 1 open microstructure data

## Package overview
```@raw html
<embed src="./assets/package_demo/Figure1_overview.pdf" width="800px" height="600px">
```
Overview of the main data types (green boxes) and functions (blue boxes) in each module of Microstructure.jl. The direction of the orange arrow between a green and blue box indicates if a certain type is an input or output to the respective function. 
