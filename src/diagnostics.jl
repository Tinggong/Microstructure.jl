using MCMCDiagnosticTools, Statistics
using DataFrames, Gadfly

export Sampler, run_diagnostics, plot_diagnostics

"""
    Change the number of samples in a sampler
"""
function Sampler(sampler::Sampler, nsamples::Int64)
    return Sampler(;
        params=sampler.params,
        prior_range=sampler.prior_range,
        proposal=sampler.proposal,
        paralinks=sampler.paralinks,
        nsamples=nsamples,
        burnin=sampler.burnin,
        thinning=sampler.thinning,
    )
end

"""
    run_diagnostics(
        meas::Vector{Float64},
        protocol::Protocol,
        model_start::BiophysicalModel,
        sampler::Sampler,
        draws::Vector{Int64},
        rng_seed::Int64=1,
        noise::Noisemodel=Noisemodel(),
    )

Return chain diagnostics tested with different sampling length. This function is useful for optimizing sampler for a given model.

# References

Vehtari, A., Gelman, A., Simpson, D., Carpenter, B. and Bürkner, P.C., 2021. Rank-normalization, folding, and localization: An improved R ̂ for assessing convergence of MCMC (with discussion). Bayesian analysis, 16(2), pp.667-718.

"""
function run_diagnostics(
    meas::Vector{Float64},
    protocol::Protocol,
    model_start::BiophysicalModel,
    sampler::Sampler,
    draws::Vector{Int64},
    rng_seed::Int64=1,
    noise::Noisemodel=Noisemodel(),
)
    ratio = zeros(length(draws))
    diagnostics = DataFrame(;
        Parameters=String[],
        NSamples=Int[],
        ESS=Float64[],
        SplitR=Float64[],
        MCSE=Float64[],
        Estimate=Float64[],
        ErrorRatio=Float64[],
    )

    for (i, nsamples) in enumerate(draws)
        sampler = Sampler(sampler, nsamples)
        chain = mcmc!(model_start, meas, protocol, sampler, noise, rng_seed)

        for para in sampler.params
            if chain[para][1] isa Vector
                fracs = reduce(hcat, chain[para])
                for i in axes(fracs, 1)
                    
                    diagns = ess_rhat(fracs[i, (sampler.burnin + 1):end]; split_chains=2)
                    se = mcse(fracs[i, (sampler.burnin + 1):end]; kind=Statistics.mean)
                    estimate = mean(fracs[i, (sampler.burnin + 1):(sampler.thinning):end])
                    push!(
                        diagnostics,
                        (
                            "f" * string(i),
                            nsamples,
                            diagns[:ess],
                            diagns[:rhat],
                            se,
                            estimate,
                            se / estimate,
                        ),
                    )
                end
            else
                diagns = ess_rhat(chain[para][(sampler.burnin + 1):end]; split_chains=2)
                se = mcse(chain[para][(sampler.burnin + 1):end]; kind=Statistics.mean)
                estimate = mean(chain[para][(sampler.burnin + 1):(sampler.thinning):end])

                push!(
                    diagnostics,
                    (
                        para,
                        nsamples,
                        diagns[:ess],
                        diagns[:rhat],
                        se,
                        estimate,
                        se / estimate,
                    ),
                )
            end
        end

        ratio[i] = sum(chain["move"]) / nsamples
    end
    return diagnostics, ratio
end

function run_diagnostics(
    meas::Vector{Float64},
    protocol::Protocol,
    model_start::BiophysicalModel,
    sampler::Tuple{Sampler,Sampler},
    draws::Vector{Int64},
    rng_seed::Int64=1,
    noise::Noisemodel=Noisemodel(),
)
    ratio = zeros(length(draws))
    diagnostics = DataFrame(;
        Parameters=String[],
        NSamples=Int[],
        ESS=Float64[],
        SplitR=Float64[],
        MCSE=Float64[],
        Estimate=Float64[],
        ErrorRatio=Float64[],
    )

    for (i, nsamples) in enumerate(draws)
        sampler_full = Sampler(sampler[1], nsamples)
        chain = mcmc!(model_start, meas, protocol, sampler_full, noise, rng_seed)

        sampler_sub = Sampler(sampler[2], nsamples)
        pertub = draw_samples(sampler_sub, noise, "dict")
        mcmc!(chain, model_start, meas, protocol, sampler_sub, pertub, noise)

        for para in sampler_full.params
            if chain[para][1] isa Vector
                fracs = reduce(hcat, chain[para])
                for i in axes(fracs, 1)
                    diagns = ess_rhat(
                        fracs[i, (sampler_full.burnin + 1):end]; split_chains=2
                    )
                    se = mcse(fracs[i, (sampler_full.burnin + 1):end]; kind=Statistics.mean)
                    estimate = mean(
                        fracs[i, (sampler_full.burnin + 1):(sampler_full.thinning):end]
                    )
                    push!(
                        diagnostics,
                        (
                            "f" * string(i),
                            nsamples,
                            diagns[:ess],
                            diagns[:rhat],
                            se,
                            estimate,
                            se / estimate,
                        ),
                    )
                end
            else
                diagns = ess_rhat(
                    chain[para][(sampler_full.burnin + 1):end]; split_chains=2
                )
                se = mcse(chain[para][(sampler_full.burnin + 1):end]; kind=Statistics.mean)
                estimate = mean(
                    chain[para][(sampler_full.burnin + 1):(sampler_full.thinning):end]
                )
                push!(
                    diagnostics,
                    (
                        para,
                        nsamples,
                        diagns[:ess],
                        diagns[:rhat],
                        se,
                        estimate,
                        se / estimate,
                    ),
                )
            end
        end

        ratio[i] = sum(chain["move"]) / nsamples
    end
    return diagnostics, ratio
end

"""
Visualize diagnostics for model parameter
"""
function plot_diagnostics(diagno::DataFrame)
    
    set_default_plot_size(30cm, 15cm)
    p0 = Gadfly.plot()
    p1 = Gadfly.plot(diagno, x=:NSamples, y=:ESS, color=:Parameters, Geom.point, Geom.line, linestyle=[:dash])
    p2 = Gadfly.plot(diagno, x=:NSamples, y=:SplitR, color=:Parameters, Geom.point, Geom.line, linestyle=[:dash])
    p3 = Gadfly.plot(diagno, x=:NSamples, y=:MCSE, color=:Parameters, Geom.point, Geom.line, linestyle=[:dash])
    p4 = Gadfly.plot(diagno, x=:NSamples, y=:Estimate, color=:Parameters, Geom.point, Geom.line, linestyle=[:dash])
    p5 = Gadfly.plot(diagno, x=:NSamples, y=:ErrorRatio, color=:Parameters, Geom.point, Geom.line, linestyle=[:dash])

    return gridstack([p1 p2 p0; p3 p4 p5])
end
