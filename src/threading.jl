
export pre_allocate, empty_chain!, create_chain, threading

"""
This method runs multi-threads MCMC estimation on dMRI data using a specified biophysical model, 
and save estimated parameters as nifti files.

    threading(
        model_start::BiophysicalModel,
        sampler::Sampler,
        dmri::MRI,
        mask::MRI,
        protocol::Protocol,
        noise_model::Noisemodel,
        savedir::String,
    )


This method returns mean and standard deviation of estimations from measurements array of size [Nmeas, Nvoxels].

    threading(
        model_start::BiophysicalModel,
        sampler::Sampler,
        measurements::Array{Float64,2},
        protocol::Protocol,
        noise_model::Noisemodel,
    )

"""
function threading(
    model_start::BiophysicalModel,
    sampler::Sampler,
    dmri::MRI,
    mask::MRI,
    protocol::Protocol,
    noise_model::Noisemodel,
    datadir::String,
)
    indexing = dropdims(mask.vol; dims=4)
    # put measurments in first dimension for faster iteration
    meas = Float64.(permutedims(dmri.vol[indexing .> 0, :], (2, 1)))

    # multi-threads processing of voxels within tissue mask
    est, est_std = threading(model_start, sampler, meas, protocol, noise_model)

    # save nifti
    for (ip, para) in enumerate(sampler.params)
        if est[ip][1] isa Vector
            mri = MRI(mask, length(est[ip][1]))

            mri.vol[indexing .> 0, :] .= reduce(vcat, est[ip])
            mri_write(mri, datadir * para * ".mean.nii.gz")

            mri.vol[indexing .> 0, :] .= reduce(vcat, est_std[ip])
            mri_write(mri, datadir * para * ".std.nii.gz")
        else
            mri = MRI(mask, 1)

            mri.vol[indexing .> 0] .= est[ip]
            mri_write(mri, datadir * para * ".mean.nii.gz")

            mri.vol[indexing .> 0] .= est_std[ip]
            mri_write(mri, datadir * para * ".std.nii.gz")
        end
    end

    return nothing
end

# threading voxels; this can be used alone for real or simulated data without reading data from nifti files
function threading(
    model_start::BiophysicalModel,
    sampler::Sampler,
    meas::Array{Float64,2},
    protocol::Protocol,
    noise_model::Noisemodel,
)
    datasize = size(meas)
    pertubations = draw_samples(sampler, noise_model)
    (measurements, estimates, chains, est, est_std) = pre_allocate(
        model_start, sampler, datasize
    )

    Threads.@threads for iv in 1:(datasize[2]::Int)

        # for voxels in the same thread, use the allocated space repeatedly
        td = Threads.threadid()
        measurements[td] .= meas[:, iv]

        # ignore voxels when normalized signals containing NaN or values larger than 1
        sum(measurements[td]) == NaN && continue
        maximum(measurements[td]) > 1 && continue

        # if want to use the same starting point for all voxels, add these two steps
        update!(estimates[td], model_start, sampler.params)
        update!(estimates[td], sampler.paralinks)

        mcmc!(chains[td], estimates[td], measurements[td], protocol, sampler, pertubations)

        for ip in 1:length(sampler.params)
            est[ip][iv] = mean(
                chains[td][ip][(sampler.burnin):(sampler.thinning):(sampler.nsamples)]
            )
            est_std[ip][iv] = std(
                chains[td][ip][(sampler.burnin):(sampler.thinning):(sampler.nsamples)]
            )
        end
    end

    return est, est_std
end

"""
    pre_allocate(
        model::BiophysicalModel, sampler::Sampler, datasize::Tuple{Int64,Int64}
    )

Allocating spaces for caching computing results based on 'model', 'sampler' and 'datasize'.
'datasize' is the size of data (Nmeas, Nvoxels) 
"""
function pre_allocate(
    model::BiophysicalModel, sampler::Sampler, datasize::Tuple{Int64,Int64}
)

    # temporal vectors to cache data in each mcmc; repeatedly used by threads
    measurements = [Vector{Float64}(undef, datasize[1]) for td in 1:Threads.nthreads()]
    estimates = [deepcopy(model) for td in 1:Threads.nthreads()]

    # chain space for each thread
    chains = [create_chain(sampler, "vec") for td in 1:Threads.nthreads()]

    # arrays hosting mean and std of samples
    #est = Array{Any}(undef,datasize[2:end]...,length(sampler.params))
    #est_std = Array{Any}(undef,datasize[2:end]...,length(sampler.params))

    est = []
    for i in eachindex(sampler.params)
        np = rand(sampler.proposal[i])
        if np isa Vector
            push!(est, fill(fill(NaN, length(np)), datasize[2]))
        else
            push!(est, [NaN for _ in 1:datasize[2]])
        end
    end
    est_std = deepcopy(est)

    return measurements, estimates, chains, est, est_std
end

"""
    create_chain(sampler, container)
create undefied container ("vec" or "dict") for saving mcmc chain
"""
function create_chain(sampler::Sampler, container::String)
    example = rand.(sampler.proposal)
    if container == "vec"
        # vec to store chains
        chain = []
        for i in eachindex(example)
            if example[i] isa Vector
                push!(chain, vec(similar(example[i], typeof(example[i]), sampler.nsamples)))
            else
                push!(chain, Vector{Float64}(undef, sampler.nsamples))
            end
        end
        # add space for sigma,logp and move for dignostics
        push!(chain, Vector{Float64}(undef, sampler.nsamples))
        push!(chain, Vector{Float64}(undef, sampler.nsamples))
        push!(chain, Vector{Int64}(undef, sampler.nsamples))

    elseif container == "dict"

        # dict to store chains
        chain = Dict()
        for i in eachindex(example)
            if example[i] isa Vector
                push!(
                    chain,
                    sampler.params[i] =>
                        vec(similar(example[i], typeof(example[i]), sampler.nsamples)),
                )
            else
                push!(chain, sampler.params[i] => Vector{Float64}(undef, sampler.nsamples))
            end
        end
        # add sigma,logp and move for dignostics
        push!(chain, "sigma" => Vector{Float64}(undef, sampler.nsamples))
        push!(chain, "logp" => Vector{Float64}(undef, sampler.nsamples))
        push!(chain, "move" => Vector{Float64}(undef, sampler.nsamples))
    else
        error("Use vec or dict")
    end
    return chain
end

"""
    empty chain while keeping keys
"""
function empty_chain!(chain::Dict{String,Vector{Any}})
    for key in keys(chain)
        empty!(chain[key])
    end
    return nothing
end
"""
    create empty chain
"""
function empty_chain(sampler::Sampler)
    # dict to store chains
    chain = Dict(para => [] for para in sampler.params)
    # add sigma,logp and move for dignostics
    push!(chain, "sigma" => [])
    push!(chain, "logp" => [])
    push!(chain, "move" => [])

    return chain
end
