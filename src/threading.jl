
export pre_allocate, empty_chain!, create_chain, threading

"""
This method runs multi-threads MCMC estimation on dMRI data using a specified biophysical model, calls the voxel threading 
method introduced in (2) and save estimated parameters as nifti files. "savedir" can include both output path and file name prefix.
Two-stage MCMC sampling methods are run if provided sampler is a Tuple of two samplers, where it will sample all the unknown parameters 
using the first sampler then sample target tissue parameters in the second sampler while fixing the rest parameters to posterior means in the first MCMC.  

    threading(
        model_start::BiophysicalModel,
        sampler::Union{Sampler,Tuple{Sampler,Sampler}},
        dmri::MRI,
        mask::MRI,
        protocol::Protocol,
        noise_model::Noisemodel,
        savedir::String,
    ) 

Methods that return mean and standard deviation of estimations from measurements array of size [Nmeas, Nvoxels] using single-stage or two-stage MCMC.

    threading(
        model_start::BiophysicalModel,
        sampler::Sampler,
        measurements::Array{Float64,2},
        protocol::Protocol,
        noise_model::Noisemodel,
    )

    
    threading(
        model_start::BiophysicalModel,
        sampler::Tuple{Sampler,Sampler},
        measurements::Array{Float64,2},
        protocol::Protocol,
        noise_model::Noisemodel,
    )

"""
function threading(
    model_start::BiophysicalModel,
    sampler::Union{Sampler,Tuple{Sampler,Sampler}},
    dmri::MRI,
    mask::MRI,
    protocol::Protocol,
    noise_model::Noisemodel,
    datadir::String,
    rng::Int64=1,
)
    
    indexing = dropdims(mask.vol; dims=4)
    # put measurments in first dimension for faster iteration
    meas = Float64.(permutedims(dmri.vol[indexing .> 0, :], (2, 1)))

    # multi-threads processing of voxels within tissue mask
    Random.seed!(rng)
    est, est_std = threading(model_start, sampler, meas, protocol, noise_model)

    sampler isa Tuple ? params = sampler[1].params : params = sampler.params

    # save nifti
    for (ip, para) in enumerate(params)
        if est[ip][1] isa Vector
            mri = MRI(mask, length(est[ip][1]), Float64)

            mri.vol[indexing .> 0, :] .= reduce(vcat, est[ip])
            mri_write(mri, datadir * para * ".mean.nii.gz")

            mri.vol[indexing .> 0, :] .= reduce(vcat, est_std[ip])
            mri_write(mri, datadir * para * ".std.nii.gz")
        else
            mri = MRI(mask, 1, Float64)

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
    pertubations = draw_samples(sampler, noise_model, "vec")
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

# voxel threading functions for two-stage MCMC sampling where you sample all the unknown parameters 
# in the first MCMC then fix and sample the other parameters in the second MCMC 
# Take dict pertubations and dict chain for reuse
function threading(
    model_start::BiophysicalModel,
    sampler::Tuple{Sampler, Sampler},
    meas::Array{Float64,2},
    protocol::Protocol,
    noise_model::Noisemodel,
)
    datasize = size(meas)
    pertubations = draw_samples(sampler[1], noise_model, "dict")
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
        update!(estimates[td], model_start, sampler[1].params)
        update!(estimates[td], sampler[1].paralinks)
        
        mcmc!(chains[td], estimates[td], measurements[td], protocol, sampler[1], pertubations)
        mcmc!(chains[td], estimates[td], measurements[td], protocol, sampler[2], pertubations)

        for (ip, para) in enumerate(sampler[1].params)
            est[ip][iv] = mean(
                chains[td][para][(sampler[1].burnin):(sampler[1].thinning):(sampler[1].nsamples)]
            )
            est_std[ip][iv] = std(
                chains[td][para][(sampler[1].burnin):(sampler[1].thinning):(sampler[1].nsamples)]
            )
        end
    end

    return est, est_std
end

"""
    pre_allocate(
        model::BiophysicalModel, sampler::Sampler, datasize::Tuple{Int64,Int64}
    )

    pre_allocate(
        model::BiophysicalModel, sampler::Tuple{Sampler,Sampler}, datasize::Tuple{Int64,Int64}
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

function pre_allocate(
    model::BiophysicalModel, sampler::Tuple{Sampler,Sampler}, datasize::Tuple{Int64,Int64}
)

    # temporal vectors to cache data in each mcmc; repeatedly used by threads
    measurements = [Vector{Float64}(undef, datasize[1]) for td in 1:Threads.nthreads()]
    estimates = [deepcopy(model) for td in 1:Threads.nthreads()]

    # chain space for each thread
    chains = [create_chain(sampler[1], "dict") for td in 1:Threads.nthreads()]

    # arrays hosting mean and std of samples
    est = []
    for i in eachindex(sampler[1].params)
        np = rand(sampler[1].proposal[i])
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
    empty some parameters in the chain while keeping keys
"""
function empty_chain!(chain::Dict{String,Vector{Any}}, keys::Tuple{Vararg{String}})
    for key in keys
        empty!(chain[key])
    end
    empty!(chain["sigma"])
    empty!(chain["logp"])
    empty!(chain["move"])
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
