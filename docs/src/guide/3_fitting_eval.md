# [Fitting evaluation] (@id guide-eval)

Fitting on synthetic data is helpful for evaluating parameter estimation accuracy under different SNR levels or imaging protocols. 
Using the `ExCaliber` model as an example, we generate signals from the forward model with varying tissue properties and perform MCMC sampling. We vary the axon diameter indices from 1 μm to 10 μm at 1 μm intervals. This range of evaluation extends slightly beyond the [sensitivity ranges calculated before](1_sensitivity_range.md).

The tissue parameters chosen for the synthetic data were values typically observed in ex vivo WM tissue: intra-axonal signal fraction  ``f_{ia} = 0.7``, dot signal fraction ``f_{dot} = 0.15``, extra-cellular signal fraction ``f_{ec}`` = 0.15, parallel diffusivity ``D_∥ = 0.6 𝜇m^2/ms`` and perpendicular diffusivity ``D_⊥ = 0.6 * 0.3 𝜇m^2/ms``. We tested three protocols with a single diffusion time and multiple b-values, where the maximum b-value was chosen to reach Gmax = 660 mT/m. The first had δ/∆ = 9.6/12 ms and b = 1, 2.5, 5, 7.5, 11.1, 18.1, 25 ``ms/𝜇m^2`` (7 b-values total), the second had δ/∆ = 11/15.192 ms and an additional b = 43 ``ms/𝜇m^2`` (8 b-values total), and the third had δ/∆ = 11/21 ms and an additional b = 64 ``ms/𝜇m^2`` (9 b-values total). Gaussian-distributed noise was added to generate 100 realizations of noisy spherical mean measurements; we show results for a moderate and lower SNR of 100 and 50 for the spherical mean measurements. 

The code below demonstrates the evaluation for the shortest diffusion time and SNR =100:
```julia
# %% packages used in this demo
using Microstructure
using Random, Distributions
using StatsPlots
using LaTeXStrings
using Plots.PlotMeasures

figdir = "/Users/tgong/Work/Projects/Microstructure.jl/demos/Toolbox/figures" 
dpi = 600  # figure resoluton

#  change setting to test on different levels of SNR 
snr = 100

# make the imaging protocol 
bval = [0, 1000, 2500, 5000, 7500, 11100, 18100, 25000] .* 1.0e6
nvol = length(bval)
techo = 40.0 .* ones(nvol) .* 1e-3
tdelta = 12.0*ones(nvol) .* 1e-3
tsmalldel = 9.6 .* ones(nvol) .* 1e-3
prot = Protocol(bval, techo, tdelta, tsmalldel)

# setup the MCMC sampler
# set the tissue parameters you want to estimate in the model; 
paras = ("axon.da", "axon.dpara", "extra.dperp_frac", "fracs")
# set parameter links
paralinks = ("axon.d0" => "axon.dpara", "extra.dpara" => "axon.dpara")

# set the range of priors and proposal distributions
pararange = ((1.0e-7, 1.0e-5), (0.01e-9, 1.2e-9), (0.0, 1.0), (0.0, 1.0))
proposal = (
    Normal(0, 0.25e-6), Normal(0, 0.025e-9), Normal(0, 0.05), MvNormal([0.0025 0; 0 0.0001])
)

# setup sampler and noise model
sampler_full = Sampler(;
    params=paras,
    prior_range=pararange,
    proposal=proposal,
    paralinks=paralinks,
    nsamples=70000,
    burnin=20000,
    thinning=100,
)
sampler_sub = subsampler(sampler_full, [1, 4])
sampler = (sampler_full, sampler_sub)

gaussian_noise = Noisemodel()
model = ExCaliber(; axon=Cylinder(; da=5.0e-6))

# simulate signals with different configurations of tissue parameters
sigma = 1.0/snr  # noise level in simulation
da_test = 1.0e-6:1.0e-6:10.0e-6  # tested range of axon diameter index
dpara = 0.6e-9
dperp_frac = 0.3
fracs = [0.7, 0.15]

Random.seed!(1)
N = 100 # the number of noise realization 

# get estimations 
est_sim = []
est_std_sim = []
for ia in eachindex(da_test)
    simpara = ExCaliber(;
        axon=Cylinder(; da=da_test[ia], dpara=dpara, d0=dpara),
        extra=Zeppelin(; dpara=dpara, dperp_frac=dperp_frac),
        fracs=fracs,
    )
    signals = model_signals(simpara, prot)

    # add gaussian noise
    noise = rand(Normal(0, sigma), nvol, N)
    meas = repeat(signals, 1, N) .+ noise

    # normalizing measurements by the b0
    meas = meas ./ repeat(meas[1, :], 1, nvol)'

    # do MCMC sampling with multi-threads 
    est, est_std = threading(model, sampler, meas, prot, gaussian_noise)

    push!(est_sim, est)
    push!(est_std_sim, est_std) 
end
```

We can then visualize the ground truth parameters vs. estimates as boxplots:
```julia
# plot and save figure 
da_est = [est_sim[ia][1] for ia in eachindex(da_test)]
dpara_est = [est_sim[ia][2] for ia in eachindex(da_test)]
dperp_frac_est = [est_sim[ia][3] for ia in eachindex(da_test)]
frac_est = [reduce(hcat, est_sim[ia][4]) for ia in eachindex(da_test)]
fia = [frac_est[ia][1, :] for ia in eachindex(da_test)]
fdot = [frac_est[ia][2, :] for ia in eachindex(da_test)]

f1 = boxplot(da_est*1e6; label=false, ylabel=L"{\mu}m", ylims=(0, 10))
plot!(da_test*1e6; lw=2, label="GT")
#annotate!(1,-1,text(L"({\mu}m)", 10))

f2 = boxplot(dpara_est*1e9; label=false, ylabel=L"{{\mu}m}^2/ms", ylims=(0, 0.9))
hline!([dpara]*1e9; lw=2, label="GT")
annotate!(1, -1, text(L"({\mu}m)", 10))

f3 = boxplot(dperp_frac_est; label=false, ylims=(0, 1))
hline!([dperp_frac]; lw=2, label="GT")
annotate!(1, -1, text(L"({\mu}m)", 10))

f4 = boxplot(fia; label=false, ylims=(0, 1))
hline!([fracs[1]]; lw=2, label="GT")
annotate!(1, -1, text(L"({\mu}m)", 10))

f5 = boxplot(fdot; label=false, ylims=(0, 0.5))
hline!([fracs[2]]; lw=2, label="GT")
annotate!(1, -1, text(L"({\mu}m)", 10))

titles =
    [L"d_{a}" L"D_{\parallel}^{ia}" L"D_{\perp}^{ec} to D_{\parallel}^{ia} fraction " L"f_{ia}" L"f_{dot}"]
plot(
    f1,
    f2,
    f3,
    f4,
    f5;
    layout=(1, 5),
    size=(1400, 250),
    titles=titles,
    xlabel=L"diameter ({\mu}m)",
    dpi=dpi,
    left_margin=8mm,
    bottom_margin=8mm,
)
savefig(figdir * "/Estimation_3c_fitd0_dt12_snr" * string(snr))
```
```@raw html
<img src="../../assets/package_demo/Estimation_3c_fitd0_dt12_snr100.png" width=800> 
```

## Results
### Measurements with a single diffusion time
Smaller axons had more precise diameter estimates and more accurate intra-axonal signal fractions than larger axons. However, diameter estimates in the low range were biased in a way that made different diameters more difficult to discriminate. In comparison, the diameters and intra-axonal signal fractions of larger axons were always under-estimated, while the dot signal fractions were more accurate. Among the different diffusion times, the shorter ones (∆ = 12 and 15.2 ms) maintained a consistent trend of axon diameter index estimates within the resolution limit (about 2-8 𝜇m). Comparison of the two SNR levels shows that the lower SNR (SNR = 50) decreased estimation precision for smaller axons but increased the discriminability of axon diameter indices in the low range. The lower SNR also increased bias for larger axons. These findings highlight the importance of performing such evaluations to understand the effects of different acquisition parameters on model fitting. 

```@raw html
<embed src="../../assets/package_demo/Figure6_ExCaliber_eval_sdt.pdf" width="800px" height="400px">
```
Estimates of axon diameters from single diffusion time data when (A) SNR = 100 and (B) SNR = 50 for spherical mean signal. (i) Data generated with 7 values, bmax = 25 ms/𝜇m2 and δ/∆ = 9.6/12 ms; (ii) Data generated with 8 b-values, bmax = 43 ms/𝜇m2 and δ/∆ = 11/15.2 ms; (iii) Data generated with 9 b-values, bmax = 64 ms/𝜇m2 and δ/∆ = 11/21 ms. Parameter estimates from 100 noise realizations are shown as boxplots and ground-truth (GT) parameter values are shown as line plots. 

### Measurements with multiple diffusion timee
Including data with multiple diffusion times improved the accuracy of axon diameter estimation, with better discriminability between smaller axons and lower biases and variances for larger axons, particularly at high SNR.  However, at lower SNR, there was less to be gained by including multiple diffusion times vs. a single diffusion time, as the differences between signals with different diffusion times were very small. In real datasets, we need to consider if the SNR is sufficient for the differences between signals acquired with different diffusion times to be significant. When using data from all the diffusion times for fitting, both the bias and variance decreased substantially. 

```@raw html
<embed src="../../assets/package_demo/Figure7_ExCaliber_eval_mdt.pdf" width="800px" height="400px">
```
Estimates of axon diameter from multi-diffusion time data when (A) SNR = 100 and (B) SNR = 50 for the spherical mean signal. (i) Data combining two shorter diffusion times: δ/∆ = 9.6/12 ms and δ/∆ = 11/15.2 ms; (ii) Data combing two longer diffusion times δ/∆ = 11/15.2 ms and δ/∆ = 11/21 ms; (iii) Data combining all three diffusion times. Parameter estimates from 100 noise realizations are shown as boxplots and ground-truth parameter values are shown as line plots. 

## References
Gong, T., & Yendiki, A. (2024). Microstructure. jl: a Julia Package for Probabilistic Microstructure Model Fitting with Diffusion MRI. arXiv preprint arXiv:2407.06379.
