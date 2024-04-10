# Minimal steps

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
julia> (dMRIdata, protocol) = spherical_mean(infile_image, normalize=true, save=true, dwiname.bvals, dwiname.bvecs, dwiname.techo, dwiname.tdelta, dwiname.tsmalldel)
```

### Specify the model we want to use

Take MTE-SANDI with a Gaussian noise model for example:
```julia
julia> tissue_model = MTE_SANDI()
julia> noise_model = Noise_model()
```

### Estimation
