# [Two-stage MCMC] (@id guide-mcmc)

After fitting a microstructure model, it is important to assess the quality of the fitting by inspecting the parameter posterior samples and determine if the fitting assumptions should be adjusted to improve fitting quality. We demonstrate this process with a two-stage MCMC fitting method that we developed for estimating axon diameter indices in ex vivo tissue (Gong et al., 2025).  

## Inspecting quality of fitting and posterior samples
For estimating axon diameter index in ex vivo tissue, previous studies have used only the intra-axonal compartment with very few, ultra-high b-values (>=20 ms/𝜇m2 for ex vivo tissue)(Veraart et al., 2020). We use the multi-compartment model `ExCaliber`, with the additional consideration that a nonnegligible and spatially varying dot signal is present in ex vivo tissue at high b-values and needs to be differentiated from the intra-axonal signals. We therefore model the full signal decay from a range of low and high b-values. The five free parameters in the `ExCaliber` model are the axon diameter index, the intra-axonal parallel diffusivity, the extra-cellular perpendicular diffusivity, and the intra-axonal and dot signal fraction. Parallel diffusivities in the intra-axonal and extra-cellular space are assumed to be equal to the intrinsic diffusivity. 

We first simulate noisy measurements from a typical WM voxel with the `ExCaliber` model and test protocol:
```julia
# %% packages used in this demo
using Microstructure
using Random, Distributions
using Plots
using LaTeXStrings
using Plots.PlotMeasures

figdir = "/Users/tgong/Work/Projects/Microstructure.jl/demos/Toolbox/figures/"
dpi = 600  # figure resoluton

# make a imaging protocol (you can also read from a *.table file)
bval = [0, 1000, 2500, 5000, 7500, 11100, 18100, 25000, 43000] .* 1.0e6
nbval = length(bval)
techo = 40.0 .* ones(nbval) .* 1e-3
tdelta = 15.192 .* ones(nbval) .* 1e-3
tsmalldel = 11.0 .* ones(nbval) .* 1e-3
prot = Protocol(bval, techo, tdelta, tsmalldel)

# make a protocol 
bval = [0, 1000, 2500, 5000, 7500, 11100, 18100, 25000, 43000] .* 1.0e6
nbval = length(bval)
techo = 40.0 .* ones(nbval) .* 1e-3
tdelta = 15.192 .* ones(nbval) .* 1e-3
tsmalldel = 11.0 .* ones(nbval) .* 1e-3
prot = Protocol(bval, techo, tdelta, tsmalldel)

# declare a model object and generate signal 
model = ExCaliber(;
    axon=Cylinder(; da=2.0e-6), 
    extra=Zeppelin(; dperp_frac=0.3), 
    fracs=[0.7, 0.15]
) # ExCaliber(Cylinder(2.0e-6, 6.0e-10, 6.0e-10, 0.0), Zeppelin(6.0e-10, 0.3, 0.0), Iso(0.0, 0.0), [0.7, 0.15])
signals = model_signals(model, prot)

# add gaussian noise to simulate noisy measurements
sigma = 1.0/100.0 # SNR=100 
meas = signals .+ rand(Normal(0, sigma), size(signals))
meas = meas ./ meas[1]
```

We can them setup the MCMC sampler to estimate model parameters from the measurements

```julia
# set the tissue parameters you want to estimate in the model; 
paras = ("axon.da", "axon.dpara", "extra.dperp_frac", "fracs")
# set parameter links
paralinks = ("axon.d0" => "axon.dpara", "extra.dpara" => "axon.dpara")

# set the range of priors and proposal distributions
pararange = ((1.0e-7, 1.0e-5), (0.01e-9, 1.2e-9), (0.0, 1.0), (0.0, 1.0))
proposal = (
    Normal(0, 0.25e-6), Normal(0, 0.025e-9), Normal(0, 0.05), MvNormal([0.0025 0; 0 0.0001])
)

# The first sampler samples all parameters defined above
sampler_full = Sampler(;
    params=paras,
    prior_range=pararange,
    proposal=proposal,
    paralinks=paralinks,
    nsamples=70000,
    burnin=20000,
    thinning=100,
)

# The second sampler samples only the first and the fourth parameters while other settings are kept the same as the first sampler
sampler_sub = subsampler(sampler_full, [1, 4])

# you can combine the two samplers to run a two-stage sampling
sampler = (sampler_full, sampler_sub)

# setup the noise model the starting point (memory space) for the estimates
gaussian_noise = Noisemodel()
estimates = ExCaliber(; axon=Cylinder(; da=5.0e-6))
```

We can now run `mcmc!` with the two-stage `sampler` directly, but since we want to inspect the posteriors of each run,
we first run only the `sampler_full`:

```julia
# do mcmc using only the first sampler
chain = mcmc!(estimates, meas, prot, sampler_full, gaussian_noise, 1)

# show the estimates after first sampler 
@show estimates
```

`chain` is the whole mcmc chain while `estimates` is the model object with estimated parameters. We can now check the fitting curve and posteriors based on these two:

```julia
"""
    get the fraction for each compartment from the vector of fractions
"""
function getfrac(fracsvec::Vector{<:Any})

    # fracsvec = chain["fracs"][burnin:thinning:end] or chain["fracs"]
    fax = [];
    fdot=[];
    fex=[]
    for vec in fracsvec
        push!(fax, vec[1])
        push!(fdot, vec[2])
    end
    fex = 1 .- fax .- fdot
    return (fax, fex, fdot)
end

# get fractions
(fax, fex, fdot) = getfrac(chain["fracs"])

# %% check the samples from posterior distributions of each parameter and show the estimates as means from the posterior distributions
burnin = 20000
thinning = 100

nbin = 20
function histograms(samples, nbins, GT, label=[false false])
    histogram(samples; normalize=:pdf, bins=nbins, label=label[1], color=:gray)
    vline!([mean(samples) GT]; label=label[2], lw=2)
end

p1 = histograms(
    chain["axon.da"][burnin:thinning:end]*1e6, 0:0.5:10, 2.0, ["samples", ["mean" "GT"]]
)
p2 = histograms(fax[burnin:thinning:end], 0:0.05:1, 0.7)
p3 = histograms(chain["axon.dpara"][burnin:thinning:end]*1e9, 0:0.1:1.2, 0.6)
p4 = histograms(fdot[burnin:thinning:end], 0:0.05:1, 0.15)
p5 = histograms(fex[burnin:thinning:end], 0:0.05:1, 0.15)
p6 = histograms(chain["extra.dperp_frac"][burnin:thinning:end], 0:0.1:1, 0.3)
p7 = histograms(chain["sigma"][burnin:thinning:end], 0:0.005:0.05, 0.01)

p8 = histogram(
    chain["logp"][burnin:thinning:end];
    normalize=:pdf,
    bins=10:2:30,
    label="samples",
    color=:gray,
)
vline!([mean(chain["logp"][burnin:thinning:end])]; label="mean", lw=2)

titles =
    [L"d_{a} [{\mu}m]" L"f_{ia}" L"D_{\parallel}^{ia} [ms/{{\mu}m}^2]" L"f_{dot}" L"f_{ec}" L"D_{\perp}^{ec} to D_{\parallel}^{ia} fraction" L"sigma" L"log(p)"]
plot(
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p8;
    layout=(2, 4),
    titles=titles,
    left_margin=5mm,
    bottom_margin=5mm,
    xguidefontsize=10,
    size=(1200, 400),
    dpi=dpi,
)
savefig(figdir * "/2_samples_mcmc_stage1")

# %% get predicted signals and compare to the measurements 
pred = model_signals(estimates, prot)
bvals = bval * 1e-9
blabel = bval[1:2:end] * 1e-9
# assess goodness of fit
plot(
    bvals,
    pred;
    label="predictions",
    lc=:black,
    lw=2,
    xticks=blabel,
    size=(600, 400),
    dpi=dpi,
)
scatter!(bvals, meas; label="measurements", mc=:red, ms=4, ma=0.5)
title!("Goodness of fit")
xlabel!(L"b\ [ms/{{\mu}m}^2]")
ylabel!("Normalized signal")
savefig(figdir * "/2_curves_mcmc_stage1")
```

Now we run the second MCMC sampler `sampler_sub` while keeping the diffusivity parameters fixed 

```julia
chain2 = mcmc!(estimates, meas, prot, sampler_sub, gaussian_noise, 1)
@show estimates
```

As above, we use `chain2` and updated `estimates` to inspect the quality of fitting and posterior samples

## Results
In the first stage, where all five tissue parameters are sampled, we find high uncertainty of estimated intra-axonal fractions. By fixing the parallel diffusivity and extra-cellular perpendicular diffusivity to their posterior means and sampling only the distributions of other 3 tissue parameters, the second MCMC run achieves similar likelihood of measurements, but lower parameter uncertainty and higher accuracy for the axon diameter index and compartment signal fractions.

```@raw html
<embed src="../../assets/package_demo/Figure5_MCMC.pdf" width="800px" height="400px">
```

## References
Gong, T., Maffei, C., Dann, E., Lee, H.-H., Lee, H., Augustinack, J.C., Huang, S.Y., Haber, S.N., Yendiki, A., 2025. Interplay between MRI-based axon diameter and myelination estimates in macaque and human brain. Imaging Neuroscience. https://doi.org/10.1162/IMAG_A_00576

Veraart, J., Nunes, D., Rudrapatna, U., Fieremans, E., Jones, D.K., Novikov, D.S., Shemesh, N., 2020. Noninvasive quantification of axon radii using diffusion MRI. Elife 9. https://doi.org/10.7554/eLife.49855
