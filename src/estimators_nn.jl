
using Flux, Distributions, Random, Statistics

export NetworkArg,
    TrainingArg,
    prepare_training,
    create_mlp,
    generate_samples,
    train_loop!,
    test,
    losses_rmse

"""
    NetworkArg(
        model::BiophysicalModel
        protocol::Protocol
        params::Tuple{Vararg{String}}
        paralinks::Tuple{Vararg{Pair{String,<:String}}} = ()
        tissuetype::String = "ex_vivo" # "in_vivo"
        sigma::Float64
        noise_type::String = "Gaussian" # "Rician"
        hidden_layers::Tuple{Vararg{Int64}}
        nsamples::Int64
        nin::Int64
        nout::Int64
        dropoutp::Float64 = 0.2
    )

Return a NetworkArg object with necessary parameters to constructe a neural network model 
and generate training samples for specifc biophysical model. Network architecture and training 
samples can be automaticlly determined from the modelling task by using function
    
    NetworkArg(model, protocol, params, paralinks, tissuetype, sigma, noise_type)
"""
Base.@kwdef struct NetworkArg
    model::BiophysicalModel
    protocol::Protocol
    params::Tuple{Vararg{String}}
    paralinks::Tuple{Vararg{Pair{String,<:String}}} = ()
    tissuetype::String = "ex_vivo" # "in_vivo"
    sigma::Float64
    noise_type::String = "Gaussian" # "Rician"
    hidden_layers::Tuple{Vararg{Int64}}
    nsamples::Int64
    nin::Int64
    nout::Int64
    dropoutp::Float64 = 0.2
end

"""
    TrainingArg(
        batchsize::Int64 
        lossf::Function
        lr::Float64
        epoch::Int64
        tv_split::Float64
        patience::Tuple{Int64,Int64} 
    )

Return TrainingArg Type object with fields related to how network will be trained.
batch size; loss function; learning rate; number of epoches; validation/training data split;
patience for train loss plateau, patience for validation loss to increase. 
Patiences are currently not apply when training and validating on generated training samples from uniform parameter distributions, 
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
end

"""
    NetworkArg(model, protocol,params,paralinks,tissuetype,sigma,noise_type,dropoutp=0.2)
Use the inputs related to biophysical models to determine network architecture and number of training samples
return a full defined NetworkArg struct 
"""
function NetworkArg(
    model::BiophysicalModel,
    protocol::Protocol,
    params::Tuple{Vararg{String}},
    paralinks::Tuple{Vararg{Pair{String}}},
    tissuetype::String,
    sigma::Float64,
    noise_type::String="Gaussian",
    dropoutp=0.2,
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
        paralinks,
        tissuetype,
        sigma,
        noise_type,
        hidden_layers,
        nsamples,
        nin,
        nout,
        dropoutp,
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
function prepare_training(arg::NetworkArg)
    mlp = create_mlp(arg.nin, arg.nout, arg.hidden_layers, arg.dropoutp)

    (inputs, labels, gt) = generate_samples(
        arg.model,
        arg.protocol,
        arg.params,
        arg.nsamples,
        arg.paralinks,
        arg.tissuetype,
        arg.sigma,
        arg.noise_type,
    )

    return mlp, inputs, labels, gt
end

"""
    create_mlp(
        ninput::Int64, 
        noutput::Int64, 
        hiddenlayers::Tuple{Vararg{Int64}}, 
        dropoutp::Float64=0.2
        )

Return a `mlp` with `ninput`/`noutput` as the number of input/output channels, and number of units in each layer specified in `hiddenlayers`; 
a dropout layer is inserted before the output layer with dropout probability `dropoutp`.
"""
function create_mlp(
    ninput::Int64, noutput::Int64, hiddenlayers::Tuple{Vararg{Int64}}, dropoutp::Float64=0.2
)
    num = (ninput, hiddenlayers...)
    mlp = [Dense(num[i] => num[i + 1], relu) for i in 1:(length(num) - 1)]

    mlp = Flux.f64(Chain(mlp..., Dropout(dropoutp), Dense(hiddenlayers[end] => noutput)))

    return mlp
end

"""
    generate_samples(
        model::BiophysicalModel,
        protocol::Protocol,
        params::Tuple{Vararg{String}},
        nsample::Int64,
        paralinks::Tuple{Vararg{Pair{String}}},
        tissuetype::String,
        sigma::Float64,
        noise_type::String,
    )
Generate and return training samples for a model using uniform coverage of tissue parameters 
and specified noise model ("Gaussian" or "Rician") and noise level `sigma`.
"""
function generate_samples(
    model::BiophysicalModel,
    protocol::Protocol,
    params::Tuple{Vararg{String}},
    nsample::Int64,
    paralinks::Tuple{Vararg{Pair{String}}},
    tissuetype::String,
    sigma::Float64,
    noise_type::String="Gaussian",
)
    params_labels = []
    params_gt = Dict()
    scaling = Microstructure.scalings[tissuetype]
    for para in params
        if !hasfield(typeof(model), Symbol(para))
            (~, subfield) = Microstructure.findsubfield(para)
            subfieldname = String(subfield)
            vecs =
                scaling[subfieldname][1][1] .+
                (scaling[subfieldname][1][2] - scaling[subfieldname][1][1]) .*
                rand(Float64, nsample)
            push!(
                params_labels,
                (vecs .* scaling[subfieldname][2] .* scaling[subfieldname][3])',
            )
            push!(params_gt, para => vecs)

        elseif para == "fracs"
            vecs = rand(Dirichlet(length(model.fracs) + 1, 1), nsample)
            push!(params_labels, vecs[1:(end - 1), :])
            if model.fracs isa Vector
                vecs = [vecs[1:(end - 1), i] for i in 1:nsample]
            else
                vecs = [vecs[1, i] for i in 1:nsample]
            end
            push!(params_gt, para => vecs)
        else 
            vecs = 
                scaling[para][1][1] .+ 
                (scaling[para][1][2] - scaling[para][1][1]) .* 
                rand(Float64, nsample)
            push!(params_labels, (vecs .* scaling[para][2] .* scaling[para][3])')
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

    # add gaussian noise to get training inputs
    if noise_type == "Gaussian"
        signals .= signals .+ rand(Normal(0, sigma), nvol, nsample)
    elseif noise_type == "Rician"
        signals .=
            sqrt.(
                (signals .+ rand(Normal(0, sigma), nvol, nsample)) .^ 2.0 .+
                rand(Normal(0, sigma), nvol, nsample) .^ 2.0
            )
    end

    return signals, params_labels, params_gt
end

"""
    train_loop!(
        mlp::Chain, 
        arg::TrainingArg, 
        inputs::Array{Float64,2}, 
        labels::Array{Float64,2}
    )
Train and update the `mlp` and return a Dict of training logs with train loss, training data loss and validation data loss for each epoch.
"""
function train_loop!(
    mlp::Chain{T}, arg::TrainingArg, inputs::Array{Float64,2}, labels::Array{Float64,2}
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
    for epoch in 1:(arg.epoch)
        losses = 0.0
        for (i, data) in enumerate(train_set)
            input, label = data
            val, grads = Flux.withgradient(mlp) do m
                # Any code inside here is differentiated.
                result = m(input)
                arg.lossf(result, label)
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

        # Save the epoch train/val loss to log
        push!(train_log["train_loss"], losses / length(train_set))
        push!(train_log["val_data_loss"], data_loss(mlp, val_set))
        push!(train_log["train_data_loss"], data_loss(mlp, train_set))
    end
    return train_log
end

"""
    test(mlp::Chain, data::Array{Float64,2}, ntest)

Return mean and standard deviation of estimations by applying a trained `mlp` to test data for `ntest` times
with dropout layer on.
"""
function test(mlp::Chain{T}, data::Array{Float64,2}, ntest) where {T}
    est = []
    est_std = []
    Flux.trainmode!(mlp)
    for i in 1:size(data, 2)
        test = mlp(data[:, i])
        for j in 1:(ntest - 1)
            test = hcat(test, mlp(data[:, i]))
        end
        push!(est, dropdims(mean(test; dims=2); dims=2))
        push!(est_std, dropdims(std(test; dims=2); dims=2))
    end

    return est, est_std
end

"""
    RMSE loss
"""
function losses_rmse(y, yy)
    return sqrt.(Flux.Losses.mse(y, yy))
end
