# Definitions of biophysical models.
#
# This script builds model structs with fields of tissue compartments and signal fractions,
# and forward functions inferencing signals from the model struct and imaging protocol.
# 
# quickly fetch literature models
# you can also add your models with desired combinations of compartments here

export model_signals,
    SANDI, 
    SANDIdot, 
    MTE_SANDI, 
    ExCaliber, 
    MTE_SMT, 
    print_model, 
    BiophysicalModel,  
    MultiTensor1,
    MultiTensor2,
    MultiTensor3

"""
All models in this page belong to the BiophysicalModel Type. You can also build your models with desired combinations of compartments using a similar syntax. 
In each model, all compartmental parameters can be considered "free parameters" and sampled using MCMC. 
This is designed to offer maximum flexibility in adjusting model assumptions, but it doesn't guarantee reliable estimation of all parameters. 
It's common that we need to fix or link some tissue parameters based on our data measurement protocols and our tissue parameters of interest.
Parameter fixing and linking can be achieved by settings in MCMC sampler in the estimator module.
"""
abstract type BiophysicalModel end

"""
    SANDI(
        soma::Sphere,
        neurite::Stick,
        extra::Iso,
        fracs::Vector{Float64}
        )

The soma and neurite density imaging (SANDI) model uses a sphere compartment to model the cell soma, 
a stick compartment to model the neurite and an isotropic diffusion compartment for the extra-cellular space; 
It includes all the tissue parameters in each compartment and a `fracs` vector representing the fraction of 
intra-soma signal and intra-neurite signal (the extra-cellular signal fraction is 1-sum(fracs)).
For SANDI model, ignore the field of `t2` in all compartments and set them to 0.

# Reference
Palombo, M., Ianus, A., Guerreri, M., Nunes, D., Alexander, D.C., Shemesh, N., Zhang, H., 2020. SANDI: A compartment-based model for non-invasive apparent soma and neurite imaging by diffusion MRI. Neuroimage 215. https://doi.org/10.1016/j.neuroimage.2020.116835
"""
Base.@kwdef mutable struct SANDI <: BiophysicalModel
    soma::Sphere = Sphere(; diff=3.0e-9)
    neurite::Stick = Stick()
    extra::Iso = Iso()
    fracs::Vector{Float64} = [0.4, 0.3]
end

"""
    MTE_SANDI(
        soma::Sphere 
        neurite::Stick
        extra::Iso 
        fracs::Vector{Float64} 
        )

For Multi-echo-SANDI (MTE-SANDI) model, consider the `t2` values in all compartments, 
and the fractions estimated will be non-T2-weighted compartment fractions in comparison to the model mentioned above. 

# Reference
Gong, T., Tax, C.M., Mancini, M., Jones, D.K., Zhang, H., Palombo, M., 2023. Multi-TE SANDI: Quantifying compartmental T2 relaxation times in the grey matter. Toronto.
"""
Base.@kwdef mutable struct MTE_SANDI <: BiophysicalModel
    soma::Sphere = Sphere(; diff=3.0e-9)
    neurite::Stick = Stick()
    extra::Iso = Iso()
    fracs::Vector{Float64} = [0.4, 0.3]
end

"""
    SANDIdot(
        soma::Sphere 
        neurite::Stick
        extra::Iso 
        dot::Iso
        fracs::Vector{Float64} 
    )

SANDIdot model includes additionally a dot compartment for SANDI model; the dot compartment is considered as immobile water and is more commonly seen in ex vivo imaging.
For SANDIdot model, ignore the field of t2 in all compartments and set them to 0. The fraction vector represents fractions of the soma, 
neurite and dot with the fraction of extra being 1-sum(fracs).

# Reference
Alexander, D.C., Hubbard, P.L., Hall, M.G., Moore, E.A., Ptito, M., Parker, G.J.M., Dyrby, T.B., 2010. Orientationally invariant indices of axon diameter and density from diffusion MRI. Neuroimage 52, 1374–1389. https://doi.org/10.1016/j.neuroimage.2010.05.043

Panagiotaki, E., Schneider, T., Siow, B., Hall, M.G., Lythgoe, M.F., Alexander, D.C., 2012. Compartment models of the diffusion MR signal in brain white matter: A taxonomy and comparison. Neuroimage 59, 2241–2254. 

Palombo, M., Ianus, A., Guerreri, M., Nunes, D., Alexander, D.C., Shemesh, N., Zhang, H., 2020. SANDI: A compartment-based model for non-invasive apparent soma and neurite imaging by diffusion MRI. Neuroimage 215. https://doi.org/10.1016/j.neuroimage.2020.116835
"""
Base.@kwdef mutable struct SANDIdot <: BiophysicalModel
    soma::Sphere = Sphere(; diff=2.0e-9)
    neurite::Stick = Stick(; dpara=0.6e-9)
    extra::Iso = Iso(; diff=0.8e-9)
    dot::Iso = Iso(; diff=0.0)
    fracs::Vector{Float64} = [0.5, 0.3, 0.1]
end

"""
    ExCaliber(
        axon::Cylinder,
        extra::Zeppelin,
        dot::Iso,
        fracs::Vector{Float64}
        )

ExCaliber is a multi-compartment model for estimating axon diameter. It can be used for ex vivo imaging when the diffusivity in the ISO compartment is set to 0 (dot compatment),
and for in vivo imaging if the diffusivity of the ISO compartment is set to free water in tissue (CSF compartment).  

# Reference
Gong, T., Maffei, C., Dann, E., Lee, H.-H., Lee, H., Augustinack, J.C., Huang, S.Y., Haber, S.N., Yendiki, A., 2025. Interplay between MRI-based axon diameter and myelination estimates in macaque and human brain. Imaging Neuroscience. https://doi.org/10.1162/IMAG_A_00576

Fan, Q., Nummenmaa, A., Witzel, T., Ohringer, N., Tian, Q., Setsompop, K., ... & Huang, S. Y. (2020). Axon diameter index estimation independent of fiber orientation distribution using high-gradient diffusion MRI. Neuroimage, 222, 117197.
"""
Base.@kwdef mutable struct ExCaliber <: BiophysicalModel
    axon::Cylinder = Cylinder()
    extra::Zeppelin = Zeppelin()
    dot::Iso = Iso(; diff=0.0)
    fracs::Vector{Float64} = [0.7, 0.1]
end

"""
    MTE_SMT(
        axon::Stick = Stick()
        extra::Zeppelin = Zeppelin()
        fracs::Float64 = 0.5
        )
    
This is a model using multi-TE spherical mean technique for lower b-value in vivo imaging. Compartmental T2s are considered. 
There is not a specific reference for this model yet, but you can refer to previous work related to this topic:

Kaden, E., Kruggel, F., Alexander, D.C., 2016. Quantitative mapping of the per-axon diffusion coefficients in brain white matter. Magn Reson Med 75, 1752–1763. https://doi.org/10.1002/MRM.25734

Kaden, E., Kelm, N. D., Carson, R. P., Does, M. D., & Alexander, D. C. (2016). Multi-compartment microscopic diffusion imaging. NeuroImage, 139, 346-359.

Veraart, J., Novikov, D.S., Fieremans, E., 2017. TE dependent Diffusion Imaging (TEdDI) distinguishes between compartmental T 2 relaxation times. https://doi.org/10.1016/j.neuroimage.2017.09.030

Gong, T., Tong, Q., He, H., Sun, Y., Zhong, J., Zhang, H., 2020. MTE-NODDI: Multi-TE NODDI for disentangling non-T2-weighted signal fractions from compartment-specific T2 relaxation times. Neuroimage 217. https://doi.org/10.1016/j.neuroimage.2020.116906
"""
Base.@kwdef mutable struct MTE_SMT <: BiophysicalModel
    axon::Stick = Stick()
    extra::Zeppelin = Zeppelin()
    fracs::Float64 = 0.5
end

"""
    model_signals(model::BiophysicalModel,prot::Protocol[,links])

Reture predicted model signals from BiophysicalModel `model` and imaging protocol 'prot'.
    `links` is a optional argument that specify parameter links in the model.
"""
function model_signals(excaliber::ExCaliber, prot::Protocol)
    fextra = 1 - sum(excaliber.fracs)
    signals =
        excaliber.fracs[1] .* compartment_signals(excaliber.axon, prot) .+
        fextra .* compartment_signals(excaliber.extra, prot) .+ excaliber.fracs[2]
    return signals ./ signals[1]
end
# normalizing to b0 signal makes models compatible for MTE imaging

function model_signals(sandi::SANDIdot, prot::Protocol)
    fextra = 1.0 - sum(sandi.fracs)
    signals =
        sandi.fracs[1] .* compartment_signals(sandi.soma, prot) .+
        sandi.fracs[2] .* compartment_signals(sandi.neurite, prot) .+
        fextra .* compartment_signals(sandi.extra, prot) .+ sandi.fracs[3]
    return signals ./ signals[1]
end

function model_signals(sandi::SANDI, prot::Protocol)
    fextra = 1.0 - sum(sandi.fracs)
    signals =
        sandi.fracs[1] .* compartment_signals(sandi.soma, prot) .+
        sandi.fracs[2] .* compartment_signals(sandi.neurite, prot) .+
        fextra .* compartment_signals(sandi.extra, prot)
    return signals ./ signals[1]
end

function model_signals(sandi::MTE_SANDI, prot::Protocol)
    fextra = 1.0 - sum(sandi.fracs)
    signals =
        (
            sandi.fracs[1] .* compartment_signals(sandi.soma, prot) .+
            sandi.fracs[2] .* compartment_signals(sandi.neurite, prot) .+
            fextra .* compartment_signals(sandi.extra, prot)
        ) 
    return signals ./ signals[1]
end

function model_signals(model::MTE_SMT, prot::Protocol)
    signals =
        (
            model.fracs .* compartment_signals(model.axon, prot) .+
            (1.0 .- model.fracs) .* compartment_signals(model.extra, prot)
        ) 
    return signals ./ signals[1] 
end

"""
    print_model(model::BiophysicalModel)
    
Helper function to check all tissue parameters in a model
"""
function print_model(model::BiophysicalModel)
    println(typeof(model), ":")
    for field in fieldnames(typeof(model))
        comp = getfield(model, field)
        subfield = fieldnames(typeof(comp))

        println(field, subfield)
    end
end

# update parameter links and get model signals
function model_signals(
    model::BiophysicalModel, prot::Protocol, links::Tuple{Vararg{Pair{String,String}}}
)
    PMI.update!(model, links)
    return model_signals(model, prot)
end


#####################################################################################
# testing multi-tensor data synthesis
Base.@kwdef mutable struct MultiTensor1 <: BiophysicalModel
    fascicle1::Tensor
    extra::Iso # an isotropic compartment for excluding partial volume effects from GM or CSF
    fracs::AbstractFloat = 0.5
end

Base.@kwdef mutable struct MultiTensor2 <: BiophysicalModel
    fascicle1::Tensor
    fascicle2::Tensor
    extra::Iso # an isotropic compartment for excluding partial volume effects from GM or CSF
    fracs::Vector{<:AbstractFloat} = [0.5, 0.3]
end

Base.@kwdef mutable struct MultiTensor3 <: BiophysicalModel
    fascicle1::Tensor
    fascicle2::Tensor
    fascicle3::Tensor
    extra::Iso # an isotropic compartment for excluding partial volume effects from GM or CSF
    fracs::Vector{<:AbstractFloat} = [0.5, 0.2, 0.2]
end

function model_signals(model::MultiTensor3, prot::Protocol)
    fextra = 1.0 - sum(model.fracs)
    signals =
        (
            model.fracs[1] .* compartment_signals(model.fascicle1, prot) .+
            model.fracs[2] .* compartment_signals(model.fascicle2, prot) .+
            model.fracs[3] .* compartment_signals(model.fascicle3, prot) .+

            fextra .* compartment_signals(model.extra, prot)
        ) 
    return signals ./ signals[1]
end

function model_signals(model::MultiTensor2, prot::Protocol)
    fextra = 1.0 - sum(model.fracs)
    signals =
        (
            model.fracs[1] .* compartment_signals(model.fascicle1, prot) .+
            model.fracs[2] .* compartment_signals(model.fascicle2, prot) .+
            fextra .* compartment_signals(model.extra, prot)
        ) 
    return signals ./ signals[1]
end

function model_signals(model::MultiTensor1, prot::Protocol)
    fextra = 1.0 - sum(model.fracs)
    signals =
        (
            model.fracs[1] .* compartment_signals(model.fascicle1, prot) .+
            fextra .* compartment_signals(model.extra, prot)
        ) 
    return signals ./ signals[1]
end