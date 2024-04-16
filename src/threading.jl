
export pre_allocate, empty_chain!, create_chain

"""
    pre_allocate(model, sampler, datasize)

Allocating spaces for caching computing results based on model and sampler
    measlen: the number of measurements in the protocol
    datasize: the size of data; for voxel-wise parallel computing, datasize is the number of voxels within a brain mask
        for slice-wise parallel computing, datasize is a vector of [sizex,sizey,sizez]
"""
function pre_allocate(model::BiophysicalModel,sampler::Sampler,datasize::Tuple{Vararg{Int64}})
    
    # temporal vectors to cache data in each mcmc; repeatedly used by threads
    measurements = [Vector{Float64}(undef, datasize[1]) for td in 1:Threads.nthreads()]
    estimates = [model for td in 1:Threads.nthreads()]

    # chain space for each thread
    chain =  create_chain(sampler,"vec")

    # chain space for each thread
    chains = [chain for td in 1:Threads.nthreads()]

    # arrays hosting fininal outputs from dataset
    outputs = similar(chain,typeof(chain),datasize[2:end])

    return measurements, estimates, chains, outputs
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
end

"""
    empty chain while keeping keys
"""
function empty_chain!(chain::Dict{String,Vector{Any}})
   
    for key in keys(chain)
        empty!(chain[key])
    end
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
end

# function threading_voxels
# function threading_slices
