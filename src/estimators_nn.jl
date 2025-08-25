
using ProgressMeter
using Flux, Distributions, Random, Statistics

export NetworkArg,
    TrainingArg,
    prepare_training,
    create_mlp,
    sigma_level,
    generate_samples,
    train_loop!,
    training,
    test,
    save_mlp,
    load_mlp,
    nn_estimate,
    save_nn_maps,
    losses_rmse,
    losses_corr,
    losses_rmse_kl,
    losses_rmse_corr,
    get_backend

"""
    NetworkArg(
        model::BiophysicalModel
        protocol::Protocol
        params::Tuple{Vararg{String}}
        prior_range::Tuple{Vararg{Tuple{Float64,Float64}}} # range for priors 
        prior_dist::Tuple{Vararg{<:Any}}
        paralinks::Tuple{Vararg{Pair{String,<:String}}} = ()
        noise_type::String = "Gaussian" # "Rician"    
        sigma_range::Tuple{Float64, Float64}
        sigma_dist::Distribution
        nsamples::Int64
        nin::Int64
        nout::Int64
        hidden_layers::Tuple{Vararg{Int64}}
        dropoutp::Union{<:AbstractFloat, Tuple{Vararg{<:AbstractFloat}}}
        actf::Function
    )

Return a `NetworkArg` object with necessary parameters to construct a neural network model 
and generate training samples for specifc biophysical model. A test network architecture and training samples can be automaticlly determined from the modelling task by using function
    
    NetworkArg(model, protocol, params, prior_range, prior_dist, paralinks, noisetype, sigma_range, sigma_dist)
"""
Base.@kwdef struct NetworkArg
    model::BiophysicalModel
    protocol::Protocol
    params::Tuple{Vararg{String}}
    prior_range::Tuple{Vararg{Tuple{Float64,Float64}}} # range for priors 
    prior_dist::Tuple{Vararg{<:Any}}
    paralinks::Union{Pair{String},Tuple{Vararg{Pair{String}}}} = ()
    noise_type::String = "Gaussian" # "Rician"    
    sigma_range::Tuple{Float64,Float64}
    sigma_dist::Distribution
    nsamples::Int64
    nin::Int64
    nout::Int64
    hidden_layers::Tuple{Vararg{Int64}}
    dropoutp::Union{<:AbstractFloat,Tuple{Vararg{<:AbstractFloat}}}
    actf::Function = relu6 # activate function for output layer
end

"""
    TrainingArg(
        batchsize::Int64 
        lossf::Function
        lr::Float64
        epoch::Int64
        tv_split::Float64
        patience::Tuple{Int64,Int64} 
        device::Function
    )

Return `TrainingArg` Type object with fields related to how network will be trained.
batch size; loss function; learning rate; number of epoches; validation/training data split;
patience for train loss plateau, patience for validation loss to increase. 
Patiences are currently not applied when training and validating on generated training samples from uniform parameter distributions, 
therefore training will stop when reaching the number of epoches. 
The patience parameter will be considered in the future when training with real data or generated data with other distributions. 
"""
Base.@kwdef struct TrainingArg
    batchsize::Int64 = 128
    lossf::Function = Flux.Losses.mse
    lr::Float64 = 0.001
    epoch::Int64 = 100
    tv_split::Float64 = 0.2
    patience::Tuple{Int64,Int64} = (10, 30)
    device::Function = cpu
end

"""
    NetworkArg(model, protocol,params,paralinks,tissuetype,sigma,noise_type,dropoutp=0.2)
Use the inputs related to biophysical models to determine network architecture and number of training samples for test
return a full defined NetworkArg struct 

Reference for adjusting the number of training samples:
Shwartz-Ziv, R., Goldblum, M., Bansal, A., Bruss, C.B., LeCun, Y., & Wilson, A.G. (2024). Just How Flexible are Neural Networks in Practice?

(Easier task and smaller MLPs have higher effective model complexity (can fit more training samples than network parameters; 
for more complex tasks and larger MLPs, the number of training samples can be set as similar to the number of network parameters to improve training efficiency)
"""
function NetworkArg(
    model::BiophysicalModel,
    protocol::Protocol,
    params::Tuple{Vararg{String}},
    prior_range::Tuple{Vararg{Tuple{Float64,Float64}}},
    prior_dist::Tuple{Vararg{<:Any}},
    paralinks::Union{Pair{String},Tuple{Vararg{Pair{String}}}},
    noise_type::String,
    sigma_range::Tuple{Float64,Float64},
    sigma_dist::Distribution,
    dropoutp=0.2,
    actf=relu6,
)
    nin = length(protocol.bval)
    nout = 0
    for para in params
        nout += length(getsubfield(model, para))
    end

    hidden_layers = (nin * 4, nin * nout, nout * 8)
    num = (nin, hidden_layers..., nout)

    # the number of trainable parameters in the network 
    npar = 0
    for i in 1:(length(num) - 1)
        npar += (num[i] + 1) * num[i + 1]
    end

    # adjust "50"
    nsamples = npar * 50
    arg = NetworkArg(
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
        actf,
    )

    return arg
end

"""
    prepare_training(arg::NetworkArg)

Return (`mlp`, `inputs`, `labels`, `gt`); `mlp` is the multi-layer perceptron network model for the biophysical model; 
`inputs` and `labels` are arrays of signals and scaled tissue parameters used for supervised training; and
`gt` is a dict containing the ground truth tissue parameters without applying scaling. Scaling is applied in the
training labels to ensure different tissue parameters are roughly in the same range as they are optimized together. 
"""
function prepare_training(arg::NetworkArg, rng_seed::Int)
    mlp = create_mlp(arg.nin, arg.nout, arg.hidden_layers, arg.dropoutp, arg.actf)

    (inputs, labels, gt) = generate_samples(
        arg.model,
        arg.protocol,
        arg.params,
        arg.prior_range,
        arg.prior_dist,
        arg.nsamples,
        arg.paralinks,
        arg.sigma_range,
        arg.sigma_dist,
        arg.noise_type,
        rng_seed,
    )

    return mlp, inputs, labels, gt
end

"""
    create_mlp(
        ninput::Int, 
        noutput::Int, 
        hiddenlayers::Tuple{Vararg{Int}}, 
        dropoutp::Union{<:AbstractFloat,Tuple{Vararg{<:AbstractFloat}}}
        )

Return a `mlp` with `ninput`/`noutput` as the number of input/output channels, and number of units in each layer specified in `hiddenlayers`; 
'dropoutp' contains the dropout probalibities for dropout layers; it can be a single value (one dropout layer before output) or same length as the hidden layers 
"""
function create_mlp(
    ninput::Int,
    noutput::Int,
    hiddenlayers::Tuple{Vararg{Int}},
    dropoutp::Union{<:AbstractFloat,Tuple{Vararg{<:AbstractFloat}}},
    out_actf::Function=relu6,
)
    num = (ninput, hiddenlayers...)
    layers_dense = [Dense(num[i] => num[i + 1], relu) for i in 1:(length(num) - 1)]
    if length(dropoutp) == 1
        mlp = Chain(
            layers_dense...,
            Dropout(Float32.(dropoutp)),
            Dense(hiddenlayers[end] => noutput, out_actf),
        )

    elseif length(dropoutp)==length(hiddenlayers)
        layers_dropout = [Dropout(Float32.(dropoutp[i])) for i in eachindex(dropoutp)]
        layers = Any[]
        for i in eachindex(dropoutp)
            push!(layers, layers_dense[i])
            push!(layers, layers_dropout[i])
        end
        mlp = Chain(layers..., Dense(hiddenlayers[end] => noutput, out_actf))
    else
        error("Numbers of dropout and hidden layer don't match")
    end

    return mlp
end

"""
    sigma_level(
        snr::MRI, 
        mask::MRI, 
    )

Return the limits (sigma_range) and distribution (sigma_distribution) of noise sigma based on SNR map defined on the b0 images, 
which defines the level of noise added to synthtic training data for neural network estimator. 
"""
function sigma_level(snr::MRI, mask::MRI)
    snrs = snr.vol[mask.vol .> 0]
    snrs = snrs[.!isnan.(snrs)]

    # The mean sigma level decided by mean SNR across tissue mask 
    sigma = 1.0/mean(snrs)

    # varing noise levels 
    minsnr, iqr1, iqr2, maxsnr = quantile(snrs, (0.05, 0.25, 0.75, 0.95))
    sigma_dist = Normal(sigma, (1.0/iqr1-1.0/iqr2)/2.0)
    sigma_range = (1/maxsnr, 1/minsnr)

    return sigma_range, sigma_dist
end

function sigma_level(snr::Array{<:AbstractFloat,3}, mask::Array{<:Real,3})
    snrs = snr[mask .> 0]
    snrs = snrs[.!isnan.(snrs)]

    # The mean sigma level decided by mean SNR across tissue mask 
    sigma = 1.0/mean(snrs)

    # varing noise levels 
    minsnr, iqr1, iqr2, maxsnr = quantile(snrs, (0.05, 0.25, 0.75, 0.95))
    sigma_dist = Normal(sigma, (1.0/iqr1-1.0/iqr2)/2.0)
    sigma_range = (1/maxsnr, 1/minsnr)

    return sigma_range, sigma_dist
end

"""
    generate_samples(
        model::BiophysicalModel,
        protocol::Protocol,
        params::Tuple{Vararg{String}},
        prior_range::Tuple{Vararg{Tuple{Float64,Float64}}}, 
        prior_dist::Tuple{Vararg{<:Any}},
        nsample::Int,
        paralinks::Union{Pair{String},Tuple{Vararg{Pair{String}}}},
        sigma_range::Tuple{Float64, Float64},
        sigma::Distribution,
        noise_type::String="Gaussian",
        rng_seed,
    )
Generate and return training samples for a model using given priors of tissue parameters 
and specified noise model (`"Gaussian"` or `"Rician"`) and noise level.
"""
function generate_samples(
    model::BiophysicalModel,
    protocol::Protocol,
    params::Tuple{Vararg{String}},
    prior_range::Tuple{Vararg{Tuple{Float64,Float64}}},
    prior_dist::Tuple{Vararg{<:Any}},
    nsample::Int,
    paralinks::Union{Pair{String},Tuple{Vararg{Pair{String}}}},
    sigma_range::Tuple{Float64,Float64},
    sigma_dist::Distribution,
    noise_type::String="Gaussian",
    rng_seed::Int=1,
)
    params_labels = []
    params_gt = Dict()
    Random.seed!(rng_seed)

    for (p, para) in enumerate(params)
        if !hasfield(typeof(model), Symbol(para))
            if isnothing(prior_dist[p])
                vecs =
                    prior_range[p][1] .+
                    (prior_range[p][2] - prior_range[p][1]) .* rand(Float64, nsample)
            else
                vecs = rand(
                    truncated(prior_dist[p], prior_range[p][1], prior_range[p][2]), nsample
                )
            end

            push!(params_labels, (vecs ./ prior_range[p][2])')
            push!(params_gt, para => vecs)

        elseif para == "fracs"
            if model.fracs isa Vector
                vecs = rand(prior_dist[p], nsample)
                push!(params_labels, vecs[1:(end - 1), :])
                vecs = [vecs[1:(end - 1), i] for i in 1:nsample]
            else
                if isnothing(prior_dist[p])
                    vecs = rand(Uniform(prior_range[p][1], prior_range[p][2]), nsample)
                else
                    vecs = rand(
                        truncated(prior_dist[p], prior_range[p][1], prior_range[p][2]),
                        nsample,
                    )
                end
                push!(params_labels, vecs')
            end
            push!(params_gt, para => vecs)
        end
    end

    params_labels = reduce(vcat, params_labels)

    # simulate signals
    nvol = length(protocol.bval)
    signals = zeros(nvol, nsample)
    for i in 1:nsample
        update!(model, Tuple(para => params_gt[para][i] for para in params))
        update!(model, paralinks)
        signals[:, i] = model_signals(model, protocol)
    end

    # the noise level "sigma" is defined by SNRs on b=0 measurements
    noise_level = rand(truncated(sigma_dist, sigma_range[1], sigma_range[2]), nsample)

    # adding noise to rotational invariants according to the number of measurements and the SH order
    noise_norm = sqrt.((2.0 .* protocol.lmeas .+ 1) .* protocol.nmeas)

    if (noise_type == "Gaussian") | (noise_type == "gaussian")
        for i in 1:nsample
            signals[:, i] .=
                signals[:, i] .+ rand(Normal(0, noise_level[i]), nvol) ./ noise_norm
            signals[:, i] = signals[:, i] ./ signals[1, i]
        end
    elseif (noise_type == "Rician") | (noise_type == "rician")
        for i in 1:nsample
            signals[:, i] .= sqrt.(
                (signals[:, i] .+ rand(Normal(0, noise_level[i]), nvol) ./ noise_norm) .^
                2.0 .+ (rand(Normal(0, noise_level[i]), nvol)) ./ noise_norm .^ 2.0,
            )
            signals[:, i] = signals[:, i] ./ signals[1, i]
        end
    else
        error("Noise type not indentified")
    end

    return Float32.(signals), Float32.(params_labels), params_gt
end

"""
    train_loop!(
        mlp::Chain, 
        arg::TrainingArg, 
        inputs::Array{Float64,2}, 
        labels::Array{Float64,2}
    )
Train and update the `mlp` and return a Dict of training logs with train loss, training data loss and validation data loss for each epoch.
This function works on cpu, which is sufficiently fast for most cases.
"""
function train_loop!(
    mlp::Chain{T},
    arg::TrainingArg,
    inputs::Array{<:AbstractFloat,2},
    labels::Array{<:AbstractFloat,2},
) where {T}
    opt_state = Flux.setup(Adam(arg.lr), mlp)
    tv_index = floor(Int64, size(inputs, 2) * arg.tv_split)

    val_set = Flux.DataLoader(
        (@views inputs[:, 1:tv_index], @views labels[:, 1:tv_index]);
        batchsize=arg.batchsize,
    )
    train_set = Flux.DataLoader(
        (@views inputs[:, (tv_index + 1):end], @views labels[:, (tv_index + 1):end]);
        batchsize=arg.batchsize,
    )

    # function to calculate validation/training data loss
    loss(mlp, x, y) = arg.lossf(mlp(x), y)
    data_loss(mlp, dataset) = mean(loss(mlp, data...) for data in dataset)

    train_log = Dict("train_loss" => [], "val_data_loss" => [], "train_data_loss" => [])

    print("Training on cpu ...")
    @showprogress for epoch in 1:(arg.epoch)
        losses = 0.0
        for (i, data) in enumerate(train_set)
            input, label = data
            val, grads = Flux.withgradient(mlp) do m
                # Any code inside here is differentiated.
                arg.lossf(m(input), label)
            end

            # add batch loss
            losses += val

            # Detect loss of Inf or NaN. Print a warning, and then skip update!
            if !isfinite(val)
                @warn "loss is $val on item $i" epoch
                continue
            end
            Flux.update!(opt_state, mlp, grads[1])
        end

        #println("Epoch #" * string(epoch) * "; training loss: " * string(losses / length(train_set)))
        # Save the epoch train/val loss to log
        push!(train_log["train_loss"], losses / length(train_set))
        push!(train_log["val_data_loss"], data_loss(mlp, val_set))
        push!(train_log["train_data_loss"], data_loss(mlp, train_set))
    end
    return train_log
end

"""
    training(
        arg::TrainingArg, 
        net::NetworkArg, 
        rng_seed::Int
    )
Train and return a trained `mlp` model, a Dict of training `logs` with train loss, training data loss and validation data loss for each epoch,
and the `inputs` and `labels` (training data) the mlp was trained on. This function allows for both cpu and gpu training.
"""
function training(arg::TrainingArg, net::NetworkArg, rng_seed::Int=1)

    # get model and training data
    mlp, inputs, labels, gt = prepare_training(net, rng_seed)

    mlp = arg.device(mlp)
    opt_state = Flux.setup(Adam(arg.lr), mlp)
    tv_index = floor(Int64, size(inputs, 2) * arg.tv_split)

    val_set = arg.device(Flux.DataLoader(
        (@views inputs[:, 1:tv_index], @views labels[:, 1:tv_index]);
        batchsize=arg.batchsize,
    ))
    train_set = arg.device(Flux.DataLoader(
        (@views inputs[:, (tv_index + 1):end], @views labels[:, (tv_index + 1):end]);
        batchsize=arg.batchsize,
    ))

    # function to calculate validation/training data loss
    loss(mlp, x, y) = arg.lossf(mlp(x), y)
    data_loss(mlp, dataset) = mean(loss(mlp, data...) for data in dataset)

    train_log = Dict("train_loss" => [], "val_data_loss" => [], "train_data_loss" => [])

    print("Training on " * string(arg.device) * " ...")
    @showprogress for epoch in 1:(arg.epoch)
        losses = 0.0
        for (i, data) in enumerate(train_set)
            input, label = data
            val, grads = Flux.withgradient(mlp) do m
                # Any code inside here is differentiated.
                arg.lossf(m(input), label)
            end

            # add batch loss
            losses += val

            # Detect loss of Inf or NaN. Print a warning, and then skip update!
            if !isfinite(val)
                @warn "loss is $val on item $i" epoch
                continue
            end
            Flux.update!(opt_state, mlp, grads[1])
        end

        #println("Epoch #" * string(epoch) * "; training loss: " * string(losses / length(train_set)))
        # Save the epoch train/val loss to log
        push!(train_log["train_loss"], losses / length(train_set))
        push!(train_log["val_data_loss"], data_loss(mlp, val_set))
        push!(train_log["train_data_loss"], data_loss(mlp, train_set))
    end
    mlp = cpu(mlp)
    return mlp, train_log, inputs, labels, gt
end

"""
    test(mlp::Chain, data::Array{<:AbstractFloat,2}, ntest)

Return probabilistic estimates by applying a trained `mlp` to test data for `ntest` times with dropout layers on.

    test(mlp::Chain, data::Array{<:AbstractFloat,2})

Get deterministic estimates with dropout layers off

"""
function test(mlp::Chain{T}, data::Array{<:AbstractFloat,2}, ntest) where {T}
    est = []
    Flux.trainmode!(mlp)
    for j in 1:ntest
        push!(est, mlp(data))
    end
    return est
end

function test(mlp::Chain{T}, data::Array{<:AbstractFloat,2}) where {T}
    Flux.testmode!(mlp)
    return mlp(data)
end

"""
Save a trained mlp model
"""
function save_mlp(
    mlp::Chain{T}, netarg::NetworkArg, savedir::String, modelname::String
) where {T}

    # save as simpler data struct
    mlp_state = Flux.state(mlp)
    jldsave(joinpath(savedir, modelname * ".jld2"); mlp_state, netarg)
end

"""
Load a mlp model
"""
function load_mlp(savedir::String, modelname::String)

    # load mlp_state, netarg
    loaded = JLD2.load(joinpath(savedir, modelname * ".jld2"))

    # get the mlp model
    mlp = create_mlp(
        netarg.nin, netarg.nout, netarg.hidden_layers, netarg.dropoutp, netarg.actf
    )

    # set model state
    Flux.loadmodel!(mlp, loaded["mlp_state"])

    return mlp, loaded["netarg"]
end

"""
   nn_estimate(
        dmri::MRI,
        mask::MRI,
        mlp::Chain{T},
        netarg::NetworkArg,
        ntest::Int,
        savedir::String,
        savename::String,
    ) 
Apply a trained mlp model to data and save estimated parameter maps as nifti
"""
function nn_estimate(
    dmri::MRI,
    mask::MRI,
    mlp::Chain{T},
    netarg::NetworkArg,
    ntest::Int,
    savedir::String,
    modelname::String,
) where {T}
    mkpath(savedir)

    # get test data
    indexing = dropdims(mask.vol; dims=4)
    meas = Float32.(permutedims(dmri.vol[indexing .> 0, :], (2, 1)))

    # inference and save maps
    posteriors = test(mlp, meas, ntest)
    est = mean(posteriors)
    est_std = std(posteriors)

    # save maps in microstructure unit
    save_nn_maps(netarg, mask, est, est_std, savedir, modelname)
end

"""
    save_nn_maps(
        netarg::NetworkArg,
        mask::MRI,
        est::Array{<:AbstractFloat,2},
        est_std::Array{<:AbstractFloat,2},
        savedir::String,
        savename::String,
    )
Save parameter maps in microstructure unit
"""
function save_nn_maps(
    netarg::NetworkArg,
    mask::MRI,
    est::Array{<:AbstractFloat,2},
    est_std::Array{<:AbstractFloat,2},
    savedir::String,
    modelname::String,
)

    ### scaling with prior maximum and microstructure unit
    scaling = Microstructure.scalings["in_vivo"]
    indexing = dropdims(mask.vol; dims=4)

    j = 1
    for (i, para) in enumerate(netarg.params)
        if !hasfield(typeof(netarg.model), Symbol(para))
            (~, subfield) = Microstructure.findsubfield(para)
            subfieldname = String(subfield)

            mri = MRI(mask, 1, Float32)

            mri.vol[indexing .> 0] .=
                est[j, :] .* netarg.prior_range[i][2] .* scaling[subfieldname][2]
            mri_write(mri, joinpath(savedir, modelname * para * ".mean.nii.gz"))

            mri.vol[indexing .> 0] .=
                est_std[j, :] .* netarg.prior_range[i][2] .* scaling[subfieldname][2]
            mri_write(mri, joinpath(savedir, modelname * para * ".std.nii.gz"))
            j = j + 1

        elseif getfield(netarg.model, Symbol(para)) isa Vector
            nf = length(getfield(netarg.model, Symbol(para)))
            mri = MRI(mask, nf, Float32)

            mri.vol[indexing .> 0, :] = est[j:(j + nf - 1), :]' .* netarg.prior_range[i][2]
            mri_write(mri, joinpath(savedir, modelname * para * ".mean.nii.gz"))

            mri.vol[indexing .> 0, :] =
                est_std[j:(j + nf - 1), :]' .* netarg.prior_range[i][2]
            mri_write(mri, joinpath(savedir, modelname * para * ".std.nii.gz"))

            j = j + nf
        else
            mri = MRI(mask, 1, Float32)

            mri.vol[indexing .> 0, :] .= est[j, :] .* netarg.prior_range[i][2]
            mri_write(mri, joinpath(savedir, modelname * para * ".mean.nii.gz"))

            mri.vol[indexing .> 0, :] .= est_std[j, :] .* netarg.prior_range[i][2]
            mri_write(mri, joinpath(savedir, modelname * para * ".std.nii.gz"))
            j = j + 1
        end
    end

    if ("extra.dperp_frac" in netarg.params)
        mean_frac = mri_read(joinpath(savedir, modelname * "extra.dperp_frac.mean.nii.gz"))
        std_frac = mri_read(joinpath(savedir, modelname * "extra.dperp_frac.std.nii.gz"))

        if ("extra.dpara" in netarg.params)
            mean_dpara = mri_read(joinpath(savedir, modelname * "extra.dpara.mean.nii.gz"))
            std_dpara = mri_read(joinpath(savedir, modelname * "extra.dpara.std.nii.gz"))

            mean_frac.vol = mean_frac.vol .* mean_dpara.vol
            std_frac.vol = sqrt.(
                std_dpara.vol .^ 2.0 .* std_frac.vol .^ 2.0 .+
                std_dpara.vol .^ 2.0 .* mean_frac.vol .^ 2.0 .+
                std_frac.vol .^ 2.0 .* mean_dpara.vol .^ 2.0,
            )

        elseif ("axon.dpara" in netarg.params) &&
            !isempty(findall(netarg.paralinks .== ("extra.dpara" => "axon.dpara")))
            mean_dpara = mri_read(joinpath(savedir, modelname * "axon.dpara.mean.nii.gz"))
            std_dpara = mri_read(joinpath(savedir, modelname * "axon.dpara.std.nii.gz"))

            mean_frac.vol = mean_frac.vol .* mean_dpara.vol
            std_frac.vol = sqrt.(
                std_dpara.vol .^ 2.0 .* std_frac.vol .^ 2.0 .+
                std_dpara.vol .^ 2.0 .* mean_frac.vol .^ 2.0 .+
                std_frac.vol .^ 2.0 .* mean_dpara.vol .^ 2.0,
            )
        else

            # for fixed extra.dpara
            mean_frac.vol = mean_frac.vol .* netarg.model.extra.dpara
            std_frac.vol = std_frac.vol .* netarg.model.extra.dpara
        end

        mri_write(mean_frac, joinpath(savedir, modelname * "extra.dperp.mean.nii.gz"))
        mri_write(std_frac, joinpath(savedir, modelname * "extra.dperp.std.nii.gz"))
    end
end

"""
    RMSE loss
"""
function losses_rmse(y, yy)
    return sqrt.(Flux.Losses.mse(y, yy))
end

# test loss
function losses_corr(y, yy)
    n=size(y, 1)
    corr =
        (n*sum(y .* yy; dims=1) - sum(y; dims=1) .* sum(yy; dims=1)) ./ sqrt.(
            (n*sum(y .^ 2; dims=1)-sum(y; dims=1) .^ 2) .*
            (n*sum(yy .^ 2; dims=1)-sum(yy; dims=1) .^ 2),
        )
    return -mean(corr)
end

function losses_rmse_kl(y, yy)
    return 0.8*sqrt.(Flux.Losses.mse(y, yy)) + 0.2*Flux.Losses.kldivergence(y, yy)
end

function losses_rmse_corr(y, yy)
    return 0.8*losses_rmse(y, yy) + 0.2*losses_corr(y, yy)
end

"""
Identify from device to use cpu or gpu for training 
"""
function get_backend()
    device = Flux.get_device(; verbose=true)
    if typeof(device) <: Flux.FluxCPUDevice
        backend = cpu
    elseif typeof(device) <:
        Union{Flux.FluxCUDADevice,Flux.FluxMetalDevice,Flux.FluxAMDGPUDevice}
        backend = gpu
    end

    return backend
end
