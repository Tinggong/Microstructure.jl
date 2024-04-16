# MCMC estimation

using Random, Distributions, StaticArrays

export Sampler, Noisemodel, mcmc!, subsampler,logp_gauss, logp_rician, update!, increment!, getsubfield, draw_samples, draw_samples!, findsubfield

"""
Noise model used for modelling
    logpdf: Function to calculate log likelihood of measurements; set between logp_gauss and logp_rician
    sigma_start: starting value
    sigma_range: prior range
    proposal: proposal distribution
"""
Base.@kwdef struct Noisemodel
    logpdf::Function = logp_gauss
    sigma_start::Float64 = 0.01
    sigma_range::Tuple{Float64,Float64} = (0.005,0.1)
    proposal::Distribution = Normal(0,0.005)
end

"""
logpdf with Gaussian noise model.
sigma is the standard deviation of Gaussian noise

# using Distributions.jl
# logpdf(Product(Normal.(preds,sigma)),meas)
"""
function logp_gauss(meas::Vector{Float64},preds::Vector{Float64},sigma::Float64)
    n = length(preds)
    return -sum((preds .- meas).^2.0) ./2.0 ./ sigma.^2.0 .- n./2.0.*log.(2.0.*pi.*sigma.^2.0)
end

"""
logpdf with Rician noise model
sigma is the standard deviation of the Gaussian noise underlying the Rician noise
"""
function logp_rician(meas::Vector{Float64},preds::Vector{Float64},sigma::Float64)
    return logpdf(Product(Rician.(preds,sigma)),meas)
end

"""
Sampler for a biophysical model
    params: parameters (String) to sample in the model
    prior_range: bounds for each parameter
    proposal: Distribution to draw pertubations
    paralinks: linking parameters in the model
    nsamples: The total number of samples in a MCMC chain; default to 70000
    burnin: The number of samples that will be discarded in the beginning of the chain; default to 20000
    thinning: The interval to extract unrelated samples in the chain; default to 100
Example Sampler for ExCaliber
    Sampler(
        params = ("axon.da","axon.dpara","extra.dperp_frac","fracs")
        prior_range = ((1.0e-7,1.0e-5),(0.01e-9,0.9e-9),(0.0, 1.0),(0.0,1.0))
        proposal = (Normal(0,0.25e-6), Normal(0,0.025e-9), Normal(0,0.05), MvNormal([0.0025 0 0;0 0.0001 0; 0 0 0.0001])) #; (0,0.05),Normal(0,0.01),Normal(0,0.01)]]
        paralinks = ("axon.d0" => "axon.dpara", "extra.dpara" => "axon.dpara")
    )
"""
Base.@kwdef struct Sampler
    params::Tuple{Vararg{String}} # parameters to sample
    prior_range::Tuple{Vararg{Tuple{Float64,Float64}}} # range for priors 
    proposal::Tuple{Vararg{<:Any}} # proposal distributions
    paralinks:: Tuple{Vararg{Pair{String}}} = () # parameter links used in modelling
    nsamples::Int64 = 70000
    burnin::Int64 = 20000
    thinning::Int64 = 100 
end

"""
Draw pertubations used in MCMC 
mutating pertubations 
"""
function draw_samples!(pertubations::Vector{<:Any},sampler::Sampler,noise::Noisemodel=Noisemodel())
    
    @inbounds for (i,para) in enumerate(sampler.params)
       
        if para != "fracs"
            pertubations[i] = rand(sampler.proposal[i],sampler.nsamples)
        else
            # convert the fraction matrix to vectors
            pertubation = rand(sampler.proposal[i],sampler.nsamples)
            pertubations[i] = [vec(pertubation[:,i]) for i in 1:sampler.nsamples]
        end
    end 
    pertubations[end] = rand(noise.proposal,sampler.nsamples)

end

"""
Draw pertubations used in MCMC 
creating pertubations
"""
function draw_samples(sampler::Sampler,noise::Noisemodel=Noisemodel())
    
    pertubations = [Vector{Any}(undef,sampler.nsamples) for i in 1:(1+length(sampler.params))::Int]
    
    @inbounds for (i,para) in enumerate(sampler.params)
       
        if para != "fracs"
            pertubations[i] = rand(sampler.proposal[i],sampler.nsamples)
        else
            pertubation = rand(sampler.proposal[i],sampler.nsamples)
            pertubations[i] = [vec(pertubation[:,i]) for i in 1:sampler.nsamples]
        end
    end 
    pertubations[end] = rand(noise.proposal,sampler.nsamples)
    
    return pertubations
end

function draw_samples(sampler::Sampler,noise::Noisemodel,rng::Int64)
    
    Random.seed!(rng)
    pertubations = Dict()

    @inbounds for (i,para) in enumerate(sampler.params)
       
        if para != "fracs"
            push!(pertubations,para=>rand(sampler.proposal[i],sampler.nsamples))
        else
            if sampler.proposal[i] isa Vector
                pertubation = rand.(sampler.proposal[i],sampler.nsamples)
                push!(pertubations, para=> [[pertubation[j][i] for j in eachindex(pertubation)] for i in 1:sampler.nsamples])
            else
                pertubation = rand(sampler.proposal[i],sampler.nsamples)
                push!(pertubations, para=> [vec(pertubation[:,i]) for i in 1:sampler.nsamples])
            end
        end
    end 
    push!(pertubations,"sigma" => rand(noise.proposal,sampler.nsamples))
    return pertubations
end

"""
Define a subsampler sampling a subset of parameters in the sampler 
using index vector for keeping parameters
"""
function subsampler(sampler::Sampler,index::Vector{Int64},paralinks:: Tuple{Vararg{Pair{String}}} = ())
    params = sampler.params[index]
    prior_range = sampler.prior_range[index]
    proposal = sampler.proposal[index]
    Sampler(params=params,prior_range=prior_range,proposal=proposal,paralinks=paralinks)
end

"""
After optimizing sampler parameters for a model, add default sampler for the model here
    an example given here is ExCaliber with two-stage MCMC 
"""
function Sampler(model::BiophysicalModel,fracdis::String,sub::Bool=false)
    
    modeltype = typeof(model)
    # tesing
    if modeltype == ExCaliber
        # set the tissue parameters you want to estimate in the model; 
        paras = ("axon.da","axon.dpara","extra.dperp_frac","fracs")
        # set parameter links
        paralinks = ("axon.d0" => "axon.dpara", "extra.dpara" => "axon.dpara")
        # set the range of priors and proposal distributions
        pararange = ((1.0e-7,1.0e-5),(0.01e-9,0.9e-9),(0.0, 1.0),(0.0,1.0))
        # replace this
        if fracdis == "mv"
            proposal = (Normal(0,0.25e-6), Normal(0,0.025e-9), Normal(0,0.05), MvNormal([0.0025 0 0;0 0.0001 0; 0 0 0.0001])) #; (0,0.05),Normal(0,0.01),Normal(0,0.01)]]
        elseif fracdis == "uv"
            proposal = (Normal(0,0.25e-6), Normal(0,0.025e-9), Normal(0,0.05),[Normal(0,0.05),Normal(0,0.01),Normal(0,0.01)])
        end
        # setup sampler and noise model
        sampler = Sampler(params=paras,prior_range=pararange,proposal=proposal,paralinks=paralinks)
        !sub && return sampler 
        return (sampler,subsampler(sampler,[1,4],())) 
    elseif modeltype == MTE_SANDI
        # under testing
    elseif modeltype == SANDI
        # under testing
    else
        error("Model not defined")
    end

end

"""
Run mcmc for a model and sampler

Method 1 generates pertubations within function, creates and returns a dict chain, and modify final model estimates in place.
    This method is useful in checking a few voxels, e.g. for quality of fitting, chain dignostics and optimizing sampler for models. 

Method 2 takes chain and pertubations as input, mutating chain in place which can be used to calculate finial estimates and uncertainties. 
    This method is used for processing larger dataset, e.g. for whole-barin/slices. 
    This method is used together with multi-threads processing that pre-allocate spaces for caching chains, avoiding creating them for each voxel. 
    This method also reuses pertubations for faster speed, as we usually use a very large number of pertubations (e.g. 70000) to densely sample the proposal distributions. 
        
Method 1 
```julia-repl
julia> mcmc!(estimates,measurements,protocol,sampler,noise_model,rng)
```

Method 2: 'chain' can be Vector (modify elements) or Dict (push!); need to benchmark time difference
```julia-repl
julia> mcmc!(chain,estimates,meas,protocol,sampler,pertubations,noise_model))
```
"""
function mcmc!(estimates::BiophysicalModel,meas::Vector{Float64},protocol::Protocol,sampler::Sampler,noise::Noisemodel=Noisemodel(),rng::Int64=1)
    
    Random.seed!(rng)

    # create chain and pertubations
    chain = create_chain(sampler,"dict")
    pertubations = draw_samples(sampler,noise,rng)

    # get logp_start from the start model and sigma defined in sampler and noise model object
    sigma = noise.sigma_start
    logp_start = noise.logpdf(meas,model_signals(estimates,protocol),sigma)

    @inbounds for i in 1:sampler.nsamples::Int
    
        # get current pertubation
        pertubation = Tuple(para => pertubations[para][i] for para in sampler.params)

        # get the next sample location and check if it is within prior ranges
        outliers = increment!(estimates,pertubation,sampler.prior_range)
        sigma += pertubations["sigma"][i]
        
        if iszero(outliers) && !outlier_checking(sigma, noise.sigma_range)
    
            # update linked parameters in model
            update!(estimates,sampler.paralinks)
            
            # update logp
            logp_next = noise.logpdf(meas,model_signals(estimates,protocol),sigma)
            
            # acception ratio
            if rand(Float64) < min(1,exp(logp_next-logp_start))
                move = 1
                logp_start = copy(logp_next)
            else
                move = 0
                # move estimates back to previous location
                decrement!(estimates,pertubation)
                update!(estimates,sampler.paralinks)
                sigma -= pertubations["sigma"][i]
            end
        else
            move = 0
            # move next back to current location
            decrement!(estimates,pertubation)
            sigma -= pertubations["sigma"][i]
        end

        record_chain!(chain,estimates,sampler.params,i,move,sigma,logp_start)
    end

    #update model object as the mean values of selected samples
    update!(estimates,Tuple(para => mean(chain[para][sampler.burnin:sampler.thinning:end]) for para in sampler.params))
    update!(estimates,sampler.paralinks)

    return chain
end

# method 2.1: mutate vector chain and use provided vector pertubations; this is used in multi-threads processing large dataset
function mcmc!(chain::Vector{Any},estimates::BiophysicalModel,meas::Vector{Float64},protocol::Protocol,sampler::Sampler,pertubations::Vector{Vector{Any}},noise::Noisemodel=Noisemodel())

    # get logp_start from the start model and sigma defined in sampler and noise model object
    sigma = noise.sigma_start
    logp_start = noise.logpdf(meas,model_signals(estimates,protocol),sigma)
    
    N = length(sampler.params)
    @inbounds for i in 1:sampler.nsamples::Int
    
        # get current pertubation
        pertubation = Tuple(sampler.params[j] => pertubations[j][i] for j in 1:N::Int)

        # get the next sample location and check if it is within prior ranges
        outliers = increment!(estimates,pertubation,sampler.prior_range)
        
        sigma += pertubations[end][i]
        
        if iszero(outliers) && !outlier_checking(sigma, noise.sigma_range)
    
            # update linked parameters in model
            update!(estimates,sampler.paralinks)
            
            # update logp
            logp_next = noise.logpdf(meas,model_signals(estimates,protocol),sigma)
            
            # acception ratio
            if rand(Float64) < min(1,exp.(logp_next-logp_start))
                move = 1
                logp_start = copy(logp_next)
            else
                move = 0
                # move estimates back to previous location
                decrement!(estimates,pertubation)
                update!(estimates,sampler.paralinks)
                sigma -= pertubations[end][i]
            end
        else
            move = 0
            # move next back to current location
            decrement!(estimates,pertubation)
            sigma -= pertubations[end][i]
        end

        record_chain!(chain,estimates,sampler.params,i,move,sigma,logp_start)
        
    end
    #update model object as the mean values of selected samples
    update!(estimates,Tuple(para => mean(chain[para][sampler.burnin:sampler.thinning:end]) for para in sampler.params))
    update!(estimates,sampler.paralinks)
end

# method 2.2: mutate dict chain and use provided pertubations; this is useful when doing two stage mcmc demonstration
function mcmc!(chain::Dict{Any,Any},estimates::BiophysicalModel,meas::Vector{Float64},protocol::Protocol,sampler::Sampler,pertubations::Dict{Any,Any},noise::Noisemodel=Noisemodel())

    #empty_chain!(chain)
    # get logp_start from the start model and sigma defined in sampler and noise model object
    sigma = noise.sigma_start
    logp_start = noise.logpdf(meas,model_signals(estimates,protocol),sigma)

    @inbounds for i in 1:sampler.nsamples::Int
    
        # get current pertubation
        pertubation = Tuple(para => pertubations[para][i] for para in sampler.params)
        
        # get the next sample location and check if it is within prior ranges
        outliers = increment!(estimates,pertubation,sampler.prior_range)
        sigma += pertubations[end][i]
        
        if iszero(outliers) && !outlier_checking(sigma, noise.sigma_range)
    
            # update linked parameters in model
            update!(estimates,sampler.paralinks)
            
            # update logp
            logp_next = noise.logpdf(meas,model_signals(estimates,protocol),sigma)
            
            # acception ratio
            if rand(Float64) < min(1,exp.(logp_next-logp_start))
                move = 1
                logp_start = copy(logp_next)
            else
                move = 0
                # move estimates back to previous location
                decrement!(estimates,pertubation)
                update!(estimates,sampler.paralinks)
                sigma -= pertubations[end][i]
            end
        else
            move = 0
            # move next back to current location
            decrement!(estimates,pertubation)
            sigma -= pertubations[end][i]
        end

        record_chain!(chain,estimates,sampler.params,i,move,sigma,logp_start)
        
    end
    #update model object as the mean values of selected samples
    update!(estimates,Tuple(para => mean(chain[para][sampler.burnin:sampler.thinning:end]) for para in sampler.params))
    update!(estimates,sampler.paralinks)
end

function record_chain!(chain::Dict{String,Vector{Any}},estimates::BiophysicalModel,params::Tuple{Vararg{String}},move::Int64,sigma::Float64,logp::Float64)
    # record estimates to chain
    for para in params
        push!(chain[para], getsubfield(estimates,para))
    end
    push!(chain["sigma"], sigma)
    push!(chain["logp"],logp)
    push!(chain["move"],move)

end

function record_chain!(chain::Dict{Any,Any},estimates::BiophysicalModel,params::Tuple{Vararg{String}},i::Int64,move::Int64,sigma::Float64,logp::Float64)

    for para in params
        chain[para][i] = getsubfield(estimates,para)
    end
    chain["sigma"][i] = sigma
    chain["logp"][i] = logp
    chain["move"][i] = move
end

function record_chain!(chain::Vector{Any},estimates::BiophysicalModel,params::Tuple{Vararg{String}},i::Int64,move::Int64,sigma::Float64,logp::Float64)

    for (j,para) in enumerate(params)
        chain[j][i] = getsubfield(estimates,para)
    end
    chain[end-2][i] = sigma
    chain[end-1][i] = logp
    chain[end][i] = move
end

"""
update fields and subfields of a model object using 
    1. given pairs containing the fieldnames and values to update; can be values or parameter links
        update!(ExCaliber(),("axon.da" => 3e-6, "axon.d0" => "axon.dpara"))
    2. another model object and fieldnames
        update!(modeltarget,modelsource,fieldnames)
"""
# update parameter and values pairs; allow mixed type in specification
function update!(model::BiophysicalModel,allfields::Tuple{Vararg{Pair{String,<:Any}}})
    
    for pair in allfields
        update!(model,pair)
    end 
    
end

function update!(model::BiophysicalModel,pair::Pair{String,Float64})

    # find the compartment and corresponding field to update
    ind = findfirst('.',pair[1])
    if !isnothing(ind)
        compname = Symbol(pair[1][1:ind-1])
        subfield = Symbol(pair[1][ind+1:end])
   
        # update subfield
        comp = getfield(model,compname) 
        setfield!(comp,subfield,pair[2])

    else
        setfield!(model,Symbol(pair[1]),pair[2])
    end
end

# update only fracs
function update!(model::BiophysicalModel,pair::Pair{String,Vector{Float64}})
    setfield!(model,Symbol(pair[1]),pair[2])
end

# update parameters using given parameter links
function update!(model::BiophysicalModel,pair::Pair{String,String})
    
    # find the compartment and corresponding field to update
    compname, field = findsubfield(pair[1])

    # find the value from given parameter link
    compname2,field2 = findsubfield(pair[2])
    value = getfield(getfield(model,compname2),field2)
    
    # update subfield
    comp=getfield(model,compname)
    setfield!(comp,field,value)

end

# Updating a model object using values from another object
function update!(model::BiophysicalModel,source::BiophysicalModel,fields::Tuple{Vararg{String}})
    
    for field in fields
        value = getsubfield(source,field)
        update!(model,field => value)
    end
    
end

"""
Get field/subfield values from a model object that can be used to update fields
""" 
function getsubfield(model::BiophysicalModel,field::String)
    
    # find the compartment and corresponding field to update
    ind = findfirst('.',field)
    if !isnothing(ind)
        compname = Symbol(field[1:ind-1])
        subfield = Symbol(field[ind+1:end])
        # update subfield
        value = getfield(getfield(model,compname),subfield)
    else
        value = getfield(model,Symbol(field))
    end
    
    return value
end

function findsubfield(field::String)
    
    ind = findfirst('.',field)
    compname = Symbol(field[1:ind-1])
    subfield = Symbol(field[ind+1:end])
    
    return compname,subfield
end

"""
Move estimates back to previous location before current pertubation. No bounds checking.
"""
function decrement!(model::BiophysicalModel,allfields::Tuple{Vararg{Pair{String,<:Any}}})
    
    for pair in allfields
        decrement!(model,pair)
    end 
    return nothing
end

# Decrement fields
function decrement!(model::BiophysicalModel,pair::Pair{String,Float64})

    # find the compartment and corresponding field to update
    ind = findfirst('.',pair[1])
    if !isnothing(ind)
        # update subfield
        compname = Symbol(pair[1][1:ind-1])
        field = Symbol(pair[1][ind+1:end])

        comp = getfield(model,compname) 
        value = getfield(comp,field) - pair[2]
        setfield!(comp,field, value)
    else
        #update field
        field = Symbol(pair[1])
        value = getfield(model,field) - pair[2]
        setfield!(model, field, value)
    end
    return nothing
end

function decrement!(model::BiophysicalModel,pair::Pair{String,Vector{Float64}})
    
    field = Symbol(pair[1])
    value = getfield(model,field) .- pair[2]
    setfield!(model, field, value)
    return nothing

end

"""
increment model estimates in place and return outliers by checking prior ranges
    model: a biophysical model
    pairs: paras of fields/subfiedls and values to add; fieldname => value2add
    ranges: prior range
"""
function increment!(model::BiophysicalModel,allfields::Tuple{Vararg{Pair{String,<:Any}}},bounds::Tuple{Vararg{Tuple{Float64,Float64}}})
    
    outliers = 0
    for (i,pair) in enumerate(allfields)
        outliers += increment!(model,pair,bounds[i]) 
    end 
    return outliers
end

# for fraction vectors
function increment!(model::BiophysicalModel,pair::Pair{String,Vector{Float64}},bounds::Tuple{Float64,Float64})

    field = Symbol(pair[1])
    value =  getfield(model,field) .+ pair[2]

    setfield!(model, field, value)
    return outlier_checking(value,bounds)  
end

# for other fields 
function increment!(model::BiophysicalModel,pair::Pair{String,Float64},bounds::Tuple{Float64,Float64})

    # find the compartment and corresponding field to updates
    ind = findfirst('.',pair[1])
    
    if !isnothing(ind)
        compname = Symbol(pair[1][1:ind-1])
        field = Symbol(pair[1][ind+1:end])

        # get updated subfield value
        comp = getfield(model,compname) 
        value = getfield(comp,field) + pair[2]

        setfield!(comp, field, value)
        return outlier_checking(value,bounds)  
    else
        field = Symbol(pair[1])
        value = getfield(model,field) + pair[2]

        setfield!(model, field, value)
        return outlier_checking(value,bounds)  
    end   
end

"""
    outlier_checking(value,(lowerbound,upperbound))

Check if a value is an outlier given a range (lowerbound,upperbound); return true if considered outlier.
When 'value' is a vector which means it represents fractions, the method checks if any elements 
or the sum of all the elements contain an outlier; return the number of outliers encounted.
"""
function outlier_checking(value::Float64,bounds::Tuple{Float64,Float64})
    
    return (value<bounds[1] || value>bounds[2])
end

function outlier_checking(fracs::Vector{Float64},bounds::Tuple{Float64,Float64}) 
    
    s = sum(fracs)
    outliers = outlier_checking(s,bounds) 
    
    for i in eachindex(fracs)
        outliers += outlier_checking(fracs[i],bounds) 
    end

    return outliers
end
 