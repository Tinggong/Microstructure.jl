
export pre_allocate, empty_chain!, create_chain

"""
    pre_allocate(model, sampler, datasize)

Allocating spaces for caching computing results based on model, sampler and datasize
    datasize: the size of data can be [Nmeas, Nvoxels] or [Nmeas,sizex,sizey,sizez]
"""
function pre_allocate(model::BiophysicalModel,sampler::Sampler,datasize::Tuple{Vararg{Int64}})
    
    # temporal vectors to cache data in each mcmc; repeatedly used by threads
    measurements = [Vector{Float64}(undef, datasize[1]) for td in 1:Threads.nthreads()]
    estimates = [deepcopy(model) for td in 1:Threads.nthreads()]

    # chain space for each thread
    chains = [create_chain(sampler,"vec") for td in 1:Threads.nthreads()]

    # arrays hosting mean and std of samples
    est = Array{Any}(undef,datasize[2:end]...,length(sampler.params))
    est_std = Array{Any}(undef,datasize[2:end]...,length(sampler.params))

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
                push!(chain,vec(similar(example[i],typeof(example[i]),sampler.nsamples)))
            else
                push!(chain,Vector{Float64}(undef,sampler.nsamples))
            end
        end
        # add space for sigma,logp and move for dignostics
        push!(chain,Vector{Float64}(undef,sampler.nsamples))
        push!(chain,Vector{Float64}(undef,sampler.nsamples))
        push!(chain,Vector{Int64}(undef,sampler.nsamples))

    elseif container == "dict"
        
        # dict to store chains
        chain = Dict()
        for i in eachindex(example)
            if example[i] isa Vector
                push!(chain,sampler.params[i] => vec(similar(example[i],typeof(example[i]),sampler.nsamples)))
            else
                push!(chain,sampler.params[i] => Vector{Float64}(undef,sampler.nsamples))
            end
        end
        # add sigma,logp and move for dignostics
        push!(chain,"sigma"=>Vector{Float64}(undef,sampler.nsamples))
        push!(chain,"logp"=>Vector{Float64}(undef,sampler.nsamples))
        push!(chain,"move"=>Vector{Float64}(undef,sampler.nsamples))
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
    chain =  Dict(para => [] for para in sampler.params)
    # add sigma,logp and move for dignostics
    push!(chain,"sigma"=>[])
    push!(chain,"logp"=>[])
    push!(chain,"move"=>[])  
    
    return chain
end

"""
    threading(model_start, sampler, meas, protocol, noise_model)

multi-threads processing of voxels or process simulated data
"""
function threading(model_start::BiophysicalModel, sampler::Sampler, meas::Array{Float64,2},protocol::Protocol, noise_model::Noisemodel)

    datasize = size(meas)
    pertubations = draw_samples(sampler,noise_model)
    (measurements, estimates, chains, est, est_std) = pre_allocate(model_start,sampler,datasize)

    Threads.@threads for iv in 1:datasize[2]::Int
            
        # for voxels in the same thread, use the allocated space repeatedly
        td = Threads.threadid()
        measurements[td] .= meas[:,iv]      
        
        # ignore voxels when normalized signals containing NaN or values larger than 1
        sum(measurements[td]) == NaN && continue      
        maximum(measurements[td]) > 1 && continue

        # if want to use the same starting point for all voxels, add these two steps
        update!(estimates[td],model_start,sampler.params)
        update!(estimates[td],sampler.paralinks)

        mcmc!(chains[td],estimates[td],measurements[td],protocol,sampler,pertubations)
        
        for ip in 1:length(sampler.params)
            est[iv,ip] = mean(chains[td][ip][sampler.burnin:sampler.thinning:sampler.nsamples])
            est_std[iv,ip] = std(chains[td][ip][sampler.burnin:sampler.thinning:sampler.nsamples])
        end
    end

    return est, est_std
end
