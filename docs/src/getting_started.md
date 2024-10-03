# Minimal steps
Here includes the minimal steps for you to get started with your MRI dataset. Visit tutorial and manual pages for more feature demonstrations. 

### Start julia in terminal with multi-threads 
```terminal
~ % julia --threads auto
```
You can also set enviornment variable by adding `export JULIA_NUM_THREADS=auto` in your bash profile, which will use multi-threads automatically when you start julia.

### Load the package in Julia
In your julia script or REPL:
```julia
julia> using Microstructure
```

### Read dMRI data and perform spherical mean

Provide full path to the DWI file and acquisition files with following extensions: dwiname.bvals (.bval), dwiname.bvecs (.bvec), dwiname.techo, dwiname.tdelta and dwiname.tsmalldel. Provide all or a subset of the acquisition files depending on the data and model you use. 

```julia
julia> (dMRI, protocol) = spherical_mean(
                            joinpath(datadir, "dwiname.nii.gz"), 
                            save=true, 
                            joinpath(datadir, "dwiname.bvals"), 
                            joinpath(datadir, "dwiname.bvecs"), 
                            joinpath(datadir, "dwiname.techo"), 
                            joinpath(datadir, "dwiname.tdelta"), 
                            joinpath(datadir, "dwiname.tsmalldel")
                            )
```
You might also need to read a tissue mask to define the region you want to process:

```julia
julia> using Fibers
julia> mask = mri_read(joinpath(datadir, "mask.nii.gz"))
```

### Specify the model we want to use and get a MCMC sampler for it

Here, we use a multi-echo spherical mean model which is curently under testing as an example:
```julia
julia> model_start = MTE_SMT(axon = Stick(dpara = 1.7e-9, t2 = 90e-3), extra = Zeppelin(dpara = 1.7e-9, t2 = 60e-3))
julia> nsample, burnin, thinning = 20000, 1000, 100
julia> sampler_smt = Sampler(model_start, nsample, burnin, thinning)
```

### MCMC Estimation
```julia
julia> savename = joinpath(datadir, "mte_smt.")
julia> threading(model_start, sampler_smt, dMRI, mask, protocol, Noisemodel(), savename)
```