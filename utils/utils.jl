# Needed packages
using Microstructure, Fibers, Flux
using Plots, Statistics
using LaTeXStrings
using Plots.PlotMeasures

export axon_da_sensitivity,
    sensitivity_plot, eval_plt, logs_plt, nn_eval, nn_estimate, sigma_level_smt

"""
Calculate the sensitivity range for axon diameter index with PA signal from a single b-value.
Inputs: 
    SNR: SNR level of data
    n: the number of unique gradient directions for the b-value measurements
    bval: b-value
    tdelta: gradient seperation time
    tsmalldel: gradient duration time
    d0: the intrinsic diffusivity within intra-axonal compartment
Outputs:
    lower bound, upper bound, G
"""
function axon_da_sensitivity(SNR, n, bval, tdelta, tsmalldel, d0)
    za = 1.64
    sigma = za ./ SNR ./ sqrt.(n)

    # make a Protocol type object from the single-b measurement
    prot = Protocol([bval], zeros(length(bval)), [tdelta], [tsmalldel])

    # generate signals from a Stick compartment and the protocol
    Sstick = compartment_signals(Stick(; dpara=d0), prot)

    # maximal signal
    upperb = Sstick[1] - sigma

    # the range of diameter indices for searching the lower and upper bounds
    range_lb = 0.5e-6:0.01e-6:4.0e-6
    range_ub = 5.0e-6:0.01e-6:20.0e-6

    cyl = Cylinder(; dpara=d0, d0=d0)
    for ia in eachindex(range_lb)
        cyl.da = range_lb[ia]
        signal = compartment_signals(cyl, prot)
        signal[1] < upperb && break
    end
    da_min = cyl.da

    for ia in eachindex(range_ub)
        cyl.da = range_ub[ia]
        signal = compartment_signals(cyl, prot)
        signal[1] < sigma && break
    end

    da_max = cyl.da
    return da_min, da_max, prot.gvec[1]
end

# example 
axon_da_sensitivity(37, 32, 43.0*1e9, 15.192e-3, 11.0e-3, 0.6*1e-9)

# %%
"""
Visulize the sensitivity profile of axon diameter given several b-values
    call the axon_da_sensitivity function and plot figures
"""
function sensitivity_plot(SNR, d0, bval_mea, tdelta, tsmalldel, n=32)
    da_min_mea = zeros(length(bval_mea))
    da_max_mea = zeros(length(bval_mea))
    g_mea = zeros(length(bval_mea))

    for ib in eachindex(bval_mea)
        da_min_mea[ib], da_max_mea[ib], g_mea[ib] = axon_da_sensitivity(
            SNR, n, bval_mea[ib], tdelta, tsmalldel, d0
        )
    end

    # plotting bval
    bval = (000.0 * 1.0e6):(1000.0 * 1.0e6):max(bval_mea...)
    da_min = zeros(length(bval))
    da_max = zeros(length(bval))
    G = zeros(length(bval))
    for ib in eachindex(bval)
        da_min[ib], da_max[ib], G[ib] = axon_da_sensitivity(
            SNR, n, bval[ib], tdelta, tsmalldel, d0
        )
    end

    xticks = bval_mea*1.0e-9
    xlabel = L"b: ms/{{\mu}m}^2"
    yticks = [0.0, 2.0, 5.0, 8.0, 11.0, 15.0]
    ylabel = L"diameter: {\mu}m"
    label = ["lower bound" "upper bound"]

    p = plot(
        bval*1.0e-9,
        [da_min, da_max]*1.0e6;
        xticks=xticks,
        xlabel=xlabel,
        yticks=yticks,
        ylabel=ylabel,
        label=label,
        legend=:topright,
        lw=2,
        ylims=(0, 18),
    )
    scatter!(bval_mea*1.0e-9, [da_min_mea, da_max_mea]*1.0e6; label=false)
    annotate!(
        bval_mea*1.0e-9,
        da_min_mea*1.0e6 .+ 1.5,
        text.(round.(da_min_mea*1.0e6; digits=2), :top, :green, 8),
    )
    annotate!(
        bval_mea[2:end]*1.0e-9,
        da_max_mea[2:end]*1.0e6 .- 1.0,
        text.(round.(da_max_mea[2:end]*1.0e6; digits=2), :right, :purple, 8),
    )

    vline!([xticks[1], xticks[end]]; line=(:dash), labels=false)
    annotate!(
        xticks[1]-2.5, 1, text("G="*string(round(Int, g_mea[1]*1000))*"mT/m", :top, 7, :red)
    )
    annotate!(
        xticks[end]-4,
        1,
        text("G="*string(round(Int, g_mea[end]*1000))*"mT/m", :top, 7, :red),
    )
    return p
end

"""
Return evaluation plots for each parameter as ground truth vs. estimates
Standard units are used in computation while they are scaled to microstructure units in plots
"""
function eval_plt(netarg::NetworkArg, est, est_std, labels)
    P_mean = Dict{String,Any}()
    P_std = Dict{String,Any}()
    para_range = Dict{String,Any}()

    # only using unit scaling; same for in vivo or ex vivo
    scaling = Microstructure.scalings["in_vivo"]
    j = 1
    for (i, para) in enumerate(netarg.params)
        if !hasfield(typeof(netarg.model), Symbol(para))
            # the unit scaling factor for this parameter
            (~, subfield) = Microstructure.findsubfield(para)
            scale_factor = [netarg.prior_range[i]...] .* scaling[String(subfield)][2]

            p = histogram2d(
                vec(labels[j, :]) .* scale_factor[2],
                vec(est[j, :]) .* scale_factor[2];
                bins=(range(scale_factor..., 20)),
                show_empty_bins=false,
            )
            plot!(
                range(scale_factor..., 20),
                range(scale_factor..., 20);
                legend=false,
                xticks=[scale_factor...],
                yticks=[scale_factor...],
            )
            push!(P_mean, para => p)

            push!(
                P_std,
                para => histogram2d(
                    vec(labels[j, :]) .* scale_factor[2],
                    vec(est_std[j, :]) .* scale_factor[2] ./
                    (scale_factor[2]-scale_factor[1]);
                    bins=(range(scale_factor..., 20), range(0, 0.25, 25)),
                    show_empty_bins=false,
                    xticks=[scale_factor...],
                    yticks=[0, 0.1, 0.2],
                ),
            )
            push!(para_range, para => (j, scale_factor))
            j = j + 1
        else
            scale_factor = [netarg.prior_range[i]...] .* scaling[para][2]
            nf = length(getfield(netarg.model, Symbol(para)))

            for n in 1:nf
                p = histogram2d(
                    vec(labels[j + n - 1, :]) .* scale_factor[2],
                    vec(est[j + n - 1, :]) .* scale_factor[2];
                    bins=(range(scale_factor..., 20)),
                    show_empty_bins=false,
                )
                plot!(
                    range(scale_factor..., 20),
                    range(scale_factor..., 20);
                    legend=false,
                    xticks=[scale_factor...],
                    yticks=[scale_factor...],
                )

                push!(P_mean, para*string(n)=>p)

                p = histogram2d(
                    vec(labels[j + n - 1, :]) .* scale_factor[2],
                    vec(est_std[j + n - 1, :]) .* scale_factor[2] ./
                    (scale_factor[2]-scale_factor[1]);
                    bins=(range(scale_factor..., 20), range(0, 0.25, 25)),
                    show_empty_bins=false,
                    xticks=[scale_factor...],
                    yticks=[0, 0.1, 0.2],
                )

                push!(P_std, para*string(n) => p)
                push!(para_range, para*string(n) => (j+n-1, scale_factor))
            end
            j = j + nf
        end
    end

    # save the extra.dperp if it was represented and estimated as a fraction of extra-/intra-cellular parallel diffusivity
    if ("extra.dperp_frac" in netarg.params)
        if ("extra.dpara" in netarg.params)
            ind = [para_range["extra.dperp_frac"][1], para_range["extra.dpara"][1]]
            scale_factor = para_range["extra.dpara"][2]

        elseif ("axon.dpara" in netarg.params) &&
            !isempty(findall(netarg.paralinks .== ("extra.dpara" => "axon.dpara")))
            ind = [para_range["extra.dperp_frac"][1], para_range["axon.dpara"][1]]
            scale_factor = para_range["axon.dpara"][2]

        else
            ind = missing
        end

        if !ismissing(ind)
            p = histogram2d(
                labels[ind[1], :] .* labels[ind[2], :] .* scale_factor[2],
                est[ind[1], :] .* est[ind[2], :] .* scale_factor[2];
                bins=(range(0, 3.0..., 20)),
                show_empty_bins=false,
            )
            plot!(
                range(0, 3.0..., 20),
                range(0, 3.0..., 20);
                legend=false,
                xticks=[0, 3.0],
                yticks=[0, 3.0],
            )

            push!(P_mean, "extra.dperp" => p)

            p = histogram2d(
                labels[ind[1], :] .* labels[ind[2], :] .* scale_factor[2],
                sqrt.(
                    est_std[ind[1], :] .^ 2.0 .* est_std[ind[2], :] .^ 2.0 .+
                    est_std[ind[1], :] .^ 2.0 .* est[ind[2], :] .^ 2.0 .+
                    est_std[ind[2], :] .^ 2.0 .* est[ind[1], :] .^ 2.0,
                ) .* scale_factor[2] ./ (scale_factor[2] - scale_factor[1]);
                bins=(range(0, 3.0..., 20), range(0, 0.25, 25)),
                show_empty_bins=false,
                xticks=[0, 3.0],
                yticks=[0, 0.1, 0.2],
            )

            push!(P_std, "extra.dperp" => p)
        end
    end

    return P_mean, P_std, para_range
end

"""
plot training logs
"""
function logs_plt(logs, trainarg)
    p = plot(logs["train_loss"]; label="Training loss")
    plot!(logs["val_data_loss"]; label="Validation data loss")
    plot!(logs["train_data_loss"]; label="Training data loss")
    p = plot(
        p;
        margin=5mm,
        xguidefontsize=8,
        yguidefontsize=10,
        title="Training performance",
        xlabel="Epoch / n",
        ylabel=String(Symbol(trainarg.lossf)),
    )
    return p
end

"""
Evaluate training and estimation accuray and precision 
    return an evaluation plot and the training logs 
"""
function nn_eval(netarg, trainarg, ntest)

    # get mlp and training data
    mlp, logs, inputs, labels = training(trainarg, netarg)

    # evaluation on synthetic data
    posteriors = test(mlp, inputs, ntest)
    eval_est = mean(posteriors)
    eval_est_std = std(posteriors)
    plots_mean, plots_std, para_range = eval_plt(netarg, eval_est, eval_est_std, labels)
    plots_logs = logs_plt(logs, trainarg)

    return plots_mean, plots_std, para_range, plots_logs, mlp
end

"""
Deciding the level of noise to add to synthetic training data based on target real datasets
"""
function sigma_level(snr::MRI, mask::MRI, protocol::Protocol)
    snrs = snr.vol[mask.vol .> 0]
    snrs_smt = snrs[snrs .> 0]*sqrt(mean(protocol.nmeas))

    # The mean sigma level decided by mean SNR across tissue mask 
    sigma = 1.0/mean(snrs_smt)

    # varing noise levels 
    minsnr, iqr1, iqr2, maxsnr = quantile(snrs_smt, (0.05, 0.25, 0.75, 0.95))
    sigma_dist = Normal(sigma, (1.0/iqr1-1.0/iqr2)/2.0)
    sigma_range = (1/maxsnr, 1/minsnr)

    return sigma_range, sigma_dist
end

"""
Deciding the level of noise to add to synthetic training data based on target real datasets
This is the sigma on spherical mean
"""
function sigma_level_smt(snr::MRI, mask::MRI, nmeas::Int)
    index = dropdims(mask.vol; dims=4) .> 0

    snrs = snr.vol[index, 1]
    snrs_smt = snrs[snrs .> 0]*sqrt(nmeas)

    # The mean sigma level decided by mean SNR across tissue mask 
    sigma = 1.0/mean(snrs_smt)

    # varing noise levels 
    minsnr, iqr1, iqr2, maxsnr = quantile(snrs_smt, (0.05, 0.25, 0.75, 0.95))
    sigma_dist = Normal(sigma, (1.0/iqr1-1.0/iqr2)/2.0)
    sigma_range = (1/maxsnr, 1/minsnr)

    return sigma_range, sigma_dist
end
