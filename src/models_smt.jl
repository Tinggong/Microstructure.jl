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
    BiophysicalModel  #ExCaliber_beta, model_signals!
"""
All models belong to BiophysicalModel type
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
        S0norm::Float64
        )

For Multi-echo-SANDI (MTE-SANDI) model, consider the `t2` values in all compartments, 
and the fractions estimated will be non-T2-weighted compartment fractions in comparison to the model mentioned above. 
`S0norm` is the relaxation-weighting free signal from all compartments S(b=0,t=0) normalised by S(b=0,t=TEmin).
"""
Base.@kwdef mutable struct MTE_SANDI <: BiophysicalModel
    soma::Sphere = Sphere(; diff=3.0e-9)
    neurite::Stick = Stick()
    extra::Iso = Iso()
    fracs::Vector{Float64} = [0.4, 0.3]
    S0norm::Float64  ## if generalize:  S0norm = S(b=0,t=TE)/S(b=0,t=TEmin) = 1 for single-TE imaging; S0 = S(b=0,t=0)/S(b=0,t=TEmin) for multi-TE imaging (1~3)
end

"""
    SANDIdot(
        soma::Sphere 
        neurite::Stick
        extra::Iso 
        dot::Iso
        fracs::Vector{Float64} 
    )

SANDIdot model includes additionally a dot compartment for ex vivo SANDI imaging; the dot compartment is considered as immobile water.
For SANDIdot model, ignore the field of t2 in all compartments and set them to 0. The fraction vector represents fractions of the soma, 
neurite and dot with the fraction of extra being 1-sum(fracs).
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
        csf::Iso,
        dot::Iso,
        fracs::Vector{Float64}
        )

ExCaliber is a model for axon diameter estimation in ex vivo tissue; the dot compartment is considered.
The fraction vector represents fractions of the axon, CSF and dot with the fraction of extra being 1-sum(fracs)
"""
Base.@kwdef mutable struct ExCaliber <: BiophysicalModel
    axon::Cylinder = Cylinder()
    extra::Zeppelin = Zeppelin()
    csf::Iso = Iso(; diff=2.0e-9)
    dot::Iso = Iso(; diff=0.0)
    fracs::Vector{Float64} = [0.7, 0.1, 0.1]
end

Base.@kwdef mutable struct ExCaliber_beta <: BiophysicalModel
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
        S0norm::Float64 = 2.0
        )
    
This is a model using multi-TE spherical mean technique for lower b-value in vivo imaging. Compartmental T2s are considered.
"""
Base.@kwdef mutable struct MTE_SMT <: BiophysicalModel
    axon::Stick = Stick()
    extra::Zeppelin = Zeppelin()
    fracs::Float64 = 0.5
    S0norm::Float64 = 2.0
end

"""
    model_signals(model::BiophysicalModel,prot::Protocol[,links])

Reture predicted model signals from BiophysicalModel `model` and imaging protocol 'prot'.
    `links` is a optional argument that specify parameter links in the model.
"""
function model_signals(excaliber::ExCaliber_beta, prot::Protocol)
    fextra = 1 - sum(excaliber.fracs)
    signals =
        excaliber.fracs[1] .* compartment_signals(excaliber.axon, prot) .+
        fextra .* compartment_signals(excaliber.extra, prot) .+ excaliber.fracs[2]
    return signals
end

function model_signals(excaliber::ExCaliber, prot::Protocol)
    fextra = 1 - sum(excaliber.fracs)
    signals =
        excaliber.fracs[1] .* compartment_signals(excaliber.axon, prot) .+
        fextra .* compartment_signals(excaliber.extra, prot) .+
        excaliber.fracs[2] .* compartment_signals(excaliber.csf, prot) .+ excaliber.fracs[3]
    return signals
end

function model_signals(sandi::SANDIdot, prot::Protocol)
    fextra = 1.0 - sum(sandi.fracs)
    signals =
        sandi.fracs[1] .* compartment_signals(sandi.soma, prot) .+
        sandi.fracs[2] .* compartment_signals(sandi.neurite, prot) .+
        fextra .* compartment_signals(sandi.extra, prot) .+ sandi.fracs[3]
    return signals
end

function model_signals(sandi::SANDI, prot::Protocol)
    fextra = 1.0 - sum(sandi.fracs)
    signals =
        sandi.fracs[1] .* compartment_signals(sandi.soma, prot) .+
        sandi.fracs[2] .* compartment_signals(sandi.neurite, prot) .+
        fextra .* compartment_signals(sandi.extra, prot)
    return signals
end

function model_signals(sandi::MTE_SANDI, prot::Protocol)
    fextra = 1.0 - sum(sandi.fracs)
    signals =
        (
            sandi.fracs[1] .* compartment_signals(sandi.soma, prot) .+
            sandi.fracs[2] .* compartment_signals(sandi.neurite, prot) .+
            fextra .* compartment_signals(sandi.extra, prot)
        ) .* sandi.S0norm
    return signals
end

function model_signals(model::MTE_SMT, prot::Protocol)
    signals =
        (
            model.fracs .* compartment_signals(model.axon, prot) .+
            (1.0 .- model.fracs) .* compartment_signals(model.extra, prot)
        ) .* model.S0norm
    return signals
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

##########################
# dev
# test mutating implementation
function model_signals!(signals::Vector{Float64}, excaliber::ExCaliber, prot::Protocol)
    signals .= 0.0
    signals_com = similar(signals)
    fextra = 1 - sum(excaliber.fracs)

    compartment_signals!(signals_com, excaliber.axon, prot)
    signals .= signals .+ excaliber.fracs[1] .* signals_com

    compartment_signals!(signals_com, excaliber.extra, prot)
    signals .= signals .+ fextra .* signals_com

    compartment_signals!(signals_com, excaliber.csf, prot)
    signals .= signals .+ excaliber.fracs[2] .* signals_com

    signals .= signals .+ excaliber.fracs[3]

    return nothing
end

# test mutating implementation
function model_signals!(
    signals::Vector{Float64},
    signals_com::Vector{Float64},
    excaliber::ExCaliber,
    prot::Protocol,
)
    signals .= 0.0
    fextra = 1 - sum(excaliber.fracs)

    compartment_signals!(signals_com, excaliber.axon, prot)
    signals .= signals .+ excaliber.fracs[1] .* signals_com

    compartment_signals!(signals_com, excaliber.extra, prot)
    signals .= signals .+ fextra .* signals_com

    compartment_signals!(signals_com, excaliber.csf, prot)
    signals .= signals .+ excaliber.fracs[2] .* signals_com

    signals .= signals .+ excaliber.fracs[3]

    return nothing
end

# update parameter links and get model signals
function model_signals(
    model::BiophysicalModel, prot::Protocol, links::Tuple{Vararg{Pair{String,String}}}
)
    PMI.update!(model, links)
    return model_signals(model, prot)
end
