# [SANDI] (@id guide-sandi)

This tutorial demonstrates how to fit the soma and neurite density imaging (SANDI) (Palombo et al., 2020) using the neural network estimator.

## Datasets 
We used high b-value data acquired on the MGH Connectome 1.0 scanner (Tian et al., 2022) for this demo. The datasets were acquired at two diffusion times, Δ = 19 or 49 ms, diffusion-encoding gradient duration δ = 8 ms, 8 different b-values for each diffusion time (50, 350, 800, 1500, 2400, 3450, 4750, and 6000 ``s/mm^2`` for Δ = 19 ms; 200, 950, 2300, 4250, 6750, 9850, 13,500, 17,800 ``s/mm^2`` for Δ = 49 ms), 32 (for b < 2400 ``s/mm^2``) or 64 (for b >= 2400 ``s/mm^2``) diffusion encoding directions uniformly distributed on a sphere. We used a subset of the data with shorter diffusion time (b = 0, 350, 800, 1500, 2400, 3450, 4750, and 6000 ``s/mm^2`` and Δ = 19 ms) for SANDI, considering that the assumption of no water exchange may be invalid at Δ = 49 in the GM (Palombo et al., 2020). We used datasets from 3 subjects with a scan and rescan to evaluate the stability of estimates. 

## Experiments 
The free parameters in the SANDI model are the intra-soma signal fraction ``f_{is}``, soma radius ``R_s``, intra-neurite signal fraction ``f_{in}``, intra-neurite parallel diffusivity ``D_∥^{in}`` and diffusivity of the isotropic extra-cellular space ``D_{ec}``. 

We generated 60,000 training samples with tissue parameters uniformly sampled from prior ranges of [2, 12] 𝜇m for ``R_s``, [1.5, 2.5] ``𝜇m^2/ms`` for ``D_∥^{in}``, and [0.5, 3.0] ``𝜇m^2/ms`` for ``D_{ec}``. The compartment fractions were sampled from a Dirichlet distribution with equal concentration parameters for the three compartments. We generated signals for the training samples using the forward model and added Gaussian noise to the signals to generate noisy measurements. The noise level was set based on the temporal SNR of the b=0 images and the number of measurements used for direction averaging.

```julia
# packages used in this demo
using Fibers, Microstructure
using Distributions, Random
using JLD2, DelimitedFiles
using Flux, Plots

# include utility functions for plotting
srcpath = dirname(pathof(Microstructure))
include(joinpath(srcpath, "../utils/utils.jl"))

# set the path to save generated figures
figdir = "/Users/tgong/Work/Projects/Microstructure.jl/demos/Toolbox/figures"
dpi = 600  # figure resoluton

# %%
### 1. Get imaging protocol 
datadir = "/Users/tgong/Work/Database/Connectome1/sub_001/dwi_real"
protocol = Protocol(joinpath(datadir, "d1_diravg_norm.btable"))

### 2. Setup model and estimator
model = SANDI(;
    soma=Sphere(; diff=3.0e-9, size=8.0e-9),
    neurite=Stick(; dpara=2.0e-9),
    extra=Iso(; diff=2.0e-9),
    fracs=[0.4, 0.4],
)

# parameters to estimate
params = ("fracs", "soma.size", "neurite.dpara", "extra.diff")
prior_range = ((0.0, 1.0), (2.0e-6, 12.0e-6), (1.5e-9, 2.5e-9), (0.5e-9, 3.0e-9))
prior_dist = (Dirichlet([1, 1, 1]), nothing, nothing, nothing)
paralinks = ()
noise_type = "Gaussian"
sigma_range = (0.002, 0.02)
sigma_dist = Normal(0.006, 0.002)

# network settings
nsamples = 60000
nin = 8
nout = 5
hidden_layers = (48, 48, 48)
dropoutp = (0.1, 0.1, 0.1)

netarg = NetworkArg(
    model,
    protocol,
    params,
    prior_range,
    prior_dist,
    paralinks,
    noise_type,
    sigma_range,
    sigma_dist,
    nsamples,
    nin,
    nout,
    hidden_layers,
    dropoutp,
    relu6,
)
trainarg = TrainingArg(; batchsize=128, lossf=losses_rmse, device=cpu)

# get mlp and training data
mlp, logs, inputs, labels = training(trainarg, netarg)

# visualize training log
logs_plt(logs, trainarg)
```
```@raw html
<img src="../../assets/package_demo/Training_log_plot_sandi.png" width=500> 
```

Performing fitting evaluation on synthetic data:
```julia
# get estimates
ntest = 100
posteriors = test(mlp, inputs, ntest)
est = mean(posteriors)
est_std = std(posteriors)

# get evaluation plot for each parameters in the model 
evalplots_mean, evalplots_std, para_range = eval_plt(netarg, est, est_std, labels)

# plotting evaluation 
titles = [L"f_{is}" L"f_{in}" L"R_{soma}" L"D_{\parallel}^{in}" L"D_{ec}"]
Plots.plot(
    evalplots_mean["fracs1"],
    evalplots_mean["fracs2"],
    evalplots_mean["soma.size"],
    evalplots_mean["neurite.dpara"],
    evalplots_mean["extra.diff"];
    layout=(2, 3),
    size=(900, 400),
    legend=:none,
    margin=5mm,
    xguidefontsize=8,
    yguidefontsize=10,
    titles=titles,
    xlabel=[L"GT" L"GT" L"GT\ [{\mu}m]" L"GT\ [{\mu}m^2/ms]" L"GT\ [{\mu}m^2/ms]"],
    ylabel=L"Estimates: Mean",
    dpi = dpi,
)

savefig(joinpath(figdir, "Eval_sandi_5p_" * info * "_mean.pdf"))

Plots.plot(
    evalplots_std["fracs1"],
    evalplots_std["fracs2"],
    evalplots_std["soma.size"],
    evalplots_std["neurite.dpara"],
    evalplots_std["extra.diff"];
    layout=(2, 3),
    size=(900, 400),
    legend=:none,
    margin=5mm,
    xguidefontsize=8,
    yguidefontsize=10,
    titles=titles,
    xlabel=[L"GT" L"GT" L"GT\ [{\mu}m]" L"GT\ [{\mu}m^2/ms]" L"GT\ [{\mu}m^2/ms]"],
    ylabel=L"Std/Prior\ Range",
    dpi = dpi,
)

savefig(joinpath(figdir, "Eval_sandi_5p_" * info * "_std.pdf"))
```

Apply trained network to real datasets:
```julia
studydir = "/Users/tgong/Work/Database/Connectome1"
subjs = (
    "sub_001", "sub_001_rescan", "sub_002", "sub_002_rescan", "sub_003", "sub_003_rescan"
)
modelname = "sandi.5p.uniform."

for subj in subjs
    datadir = joinpath(studydir, subj, "dwi_real")
    dmri = mri_read(joinpath(datadir, "d1_diravg_norm.nii.gz"))
    mask = mri_read(joinpath(datadir, subj*"_dwi_real_brainmask.nii.gz"))

    savedir = joinpath(datadir, "SANDI")
    mkpath(savedir)
    nn_estimate(dmri, mask, mlp, netarg, ntest, savedir, modelname)
end
```

## Results

**Fitting evaluation.** While other model parameters can be accurately estimated, the ``D_∥^{in}`` cannot be estimated with the evaluated protocol, and the estimates tend to be biased towards the mean values of the prior distribution.  

```@raw html
<embed src="../../assets/package_demo/Figure10_SANDI_eval.pdf" width="600px" height="300px">
```

Fitting evaluations of the SANDI model on synthetic training data with uninformative priors. (a) 2D histograms of ground-truth labels vs. estimates; (b) the 2D histogram of ground-truth labels vs. the standard deviation of posteriors relative to used prior range; (c) the distributions of training labels. The signal fractions of the three tissue compartments are summed to 1 and follow the Dirichlet distribution. This evaluation suggests that the protocol is not sensitive to ``D_∥^{in}``.

**Parameter maps on real dataset.** The uncertainty map of ``f_{in}`` features WM regions in the genu of corpus callosum with very low ``f_{in}`` estimates, which could potentially be outliers. The uncertainty maps of all parameters also feature a GM region with atypical values in the ``f_in``, ``R_s`` and ``D_{ec}``. The distributions of parameters in GM regions including caudate and putamen show high consistency among the 6 scans. 

```@raw html
<embed src="../../assets/package_demo/Figure11_SANDI_map.pdf" width="600px" height="400px">
```

SANDI model fitting on in vivo human data acquired with ultra-high gradient strength. Only the four parameters that can be estimated robustly are shown here. (a) Parameter maps from one subject. The WM region highlighted by orange arrows in the genu of corpus callosum with low ``f_{in}`` values shows high uncertainty, suggesting potential outlier estimates. The GM region highlighted by white arrows showing atypical values in the ``f_{in}``,  ``R_s`` and ``D_{ec}`` exhibits high uncertainties in all the parameters. (b) Probability density of estimates in GM ROIs (caudate and putamen) from 6 scans.

## References
Palombo, M., Ianus, A., Guerreri, M., Nunes, D., Alexander, D.C., Shemesh, N., Zhang, H., 2020. SANDI: A compartment-based model for non-invasive apparent soma and neurite imaging by diffusion MRI. Neuroimage 215. https://doi.org/10.1016/j.neuroimage.2020.116835

Tian, Q., Fan, Q., Witzel, T., Polackal, M.N., Ohringer, N.A., Ngamsombat, C., Russo, A.W., Machado, N., Brewer, K., Wang, F., Setsompop, K., Polimeni, J.R., Keil, B., Wald, L.L., Rosen, B.R., Klawiter, E.C., Nummenmaa, A., Huang, S.Y., 2022. Comprehensive diffusion MRI dataset for in vivo human brain microstructure mapping using 300 mT/m gradients. Scientific Data 2022 9:1 9, 1–11. https://doi.org/10.1038/s41597-021-01092-6