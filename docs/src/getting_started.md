# What is microstructure imaging?

Microstructure imaging allows us to estimate biologically meaningful cellular parameters from MRI data. This is achieved through simplifying the tissue in a voxel as multiple compartments representing different cellular structures, e.g. axons and cell bodies etc. Each tissue compartment has unique features in determine the MR signals under a measurement protocol, and therefore can be estimated by solving an inverse problem.

## Minimal steps
Here includes the minimal steps for you to get started with your MRI dataset. Visit tutorial and manual pages for more feature demonstrations. 

### Start julia in terminal with multi-threads 
```terminal
~ % julia --threads auto
```
You can also set enviornment variable by adding `export JULIA_NUM_THREADS=auto` in your bash profile, which will use multi-threads automatically when you start julia.

### Load the package in Julia
In you julia script or REPL:
```julia
julia> using Microstructure
```

### Read dMRI data and perform spherical mean

Provide full path to dMRI images and names of acquisition files with following extensions:
    dwiname.bvals, dwiname.bvecs, dwiname.techo, dwiname.tdelta, dwiname.tsmalldel 
    provide all or a subset of the files depending on the data and model you use. 

```julia
julia> (dMRI, protocol) = spherical_mean(
                            datadir * "/dwiname.nii.gz", 
                            normalize=true, 
                            save=true, 
                            datadir * "dwiname.bvals", 
                            datadir * "dwiname.bvecs", 
                            datadir * "dwiname.techo", 
                            datadir * "dwiname.tdelta", 
                            datadir * "dwiname.tsmalldel")
```
You might also need to read a tissue mask to define the region you want to process:

```julia
julia> using Fibers
julia> mask = mri_read(datadir * "/mask.nii.gz")
```

### Specify the model we want to use and get a MCMC sampler for it

Here, we use a multi-echo spherical mean model which is curently under testing as an example:
```julia
julia> model_start = MTE_SMT(axon = Stick(dpara = 1.7e-9, t2 = 90e-3), extra = Zeppelin(dpara = 1.7e-9, t2 = 60e-3))
julia> sampler_smt = Sampler(model_start)
```

### MCMC Estimation
```julia
julia> savename = datadir * "/mte_smt."
julia> threading(model_start, sampler_smt, dMRI, mask, protocol, Noisemodel(), savename)
```