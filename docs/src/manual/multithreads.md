# Multi threads

### Start julia in terminal with multi-threads 
```terminal
~ % julia --threads auto
```
You can also set enviornment variable by adding `export JULIA_NUM_THREADS=auto` in your bash profile, which will use multi-threads automatically when you start julia.

### Multi-threads for MCMC estimation

```@docs
threading
```