# Definitions of biophysical models.
#
# This script builds model structs with fields of tissue compartments and signal fractions,
# and forward functions inferencing signals from the model struct and imaging protocol.
# 
# quickly fetch literature models
# you can also add your models with desired combinations of compartments here


export model_signals, SANDI, SANDIdot, MTE_SANDI, ExCaliber, MTE_SMT, print_model, BiophysicalModel  #ExCaliber_beta, model_signals!

# Models belong to BiophysicalModel type
abstract type BiophysicalModel end

"""
SANDI models or MTE-SANDI models including three tissue compartments for in vivo imaging
For SANDI model, ignore the field of t2 in all compartments and set them to 0
"""
Base.@kwdef mutable struct SANDI <: BiophysicalModel
    soma::Sphere = Sphere(diff=3.0e-9)
    neurite::Stick = Stick()
    extra::Iso = Iso()
    fracs::Vector{Float64} = [0.4,0.3]
end

"""
For MTE-SANDI model, consider the t2 values in all compartments
The fraction vector represents fractions of the soma and neurite with the fraction of extra being 1-sum(fracs)
S0norm is the relaxation-weighting free signal from all compartments S(b=0,t=0) normalised by S(b=0,t=TEmin)

"""
Base.@kwdef mutable struct MTE_SANDI <: BiophysicalModel
    soma::Sphere = Sphere(diff=3.0e-9)
    neurite::Stick = Stick()
    extra::Iso = Iso()
    fracs::Vector{Float64} = [0.4,0.3]
    S0norm::Float64  ## if generalize:  S0norm = S(b=0,t=TE)/S(b=0,t=TEmin) = 1 for single-TE imaging; S0 = S(b=0,t=0)/S(b=0,t=TEmin) for multi-TE imaging (1~3)
end

"""
SANDIdot models or MTE-SANDIdot models including additionally a dot compartment for ex vivo imaging
For SANDIdot model, ignore the field of t2 in all compartments and set them to 0
For MTE-SANDIdot model, consider the t2 values in all compartments
The fraction vector represents fractions of the soma, neurite and dot with the fraction of extra being 1-sum(fracs)
"""
Base.@kwdef mutable struct SANDIdot <: BiophysicalModel
    soma::Sphere = Sphere(diff=2.0e-9)
    neurite::Stick = Stick(dpara=0.6e-9)
    extra::Iso = Iso(diff=0.8e-9)
    dot::Iso = Iso(diff=0.0)
    fracs::Vector{Float64} = [0.5,0.3,0.1]
end

"""
ExCaliber model for ex vivo tissue; dot signal considered
The fraction vector represents fractions of the axon, CSF and dot with the fraction of extra being 1-sum(fracs)
"""
Base.@kwdef mutable struct ExCaliber <: BiophysicalModel
    axon::Cylinder = Cylinder()
    extra::Zeppelin = Zeppelin()
    csf::Iso = Iso(diff=2.0e-9)
    dot::Iso = Iso(diff=0.0)
    fracs::Vector{Float64} = [0.7,0.1,0.1]
end

Base.@kwdef mutable struct ExCaliber_beta <: BiophysicalModel
    axon::Cylinder = Cylinder()
    extra::Zeppelin = Zeppelin()
    dot::Iso = Iso(diff=0.0)
    fracs::Vector{Float64} = [0.7,0.1]
end

"""
To test multi-TE spherical mean technique for low-b in vivo imaging
"""
Base.@kwdef mutable struct MTE_SMT <: BiophysicalModel
    axon::Cylinder = Stick()
    extra::Zeppelin = Zeppelin()
    fracs::Float64 = 0.5
    S0norm::Float64 = 2.0
end

"""
    model_signals(model,prot[,links])

Predict model signals from BiophysicalModel `model` and imaging protocol 'prot'.
    `links` is a optional argument that specify parameter links in the model
"""
function model_signals(excaliber::ExCaliber_beta, prot::Protocol)
    fextra = 1-sum(excaliber.fracs)
    signals = excaliber.fracs[1].*compartment_signals(excaliber.axon,prot) .+ fextra.*compartment_signals(excaliber.extra,prot) .+ excaliber.fracs[2]
    return signals
end

function model_signals(excaliber::ExCaliber, prot::Protocol)
    fextra = 1-sum(excaliber.fracs)
    signals = excaliber.fracs[1].*compartment_signals(excaliber.axon,prot) .+ fextra.*compartment_signals(excaliber.extra,prot) .+ excaliber.fracs[2].*compartment_signals(excaliber.csf,prot) .+ excaliber.fracs[3]
    return signals
end

function model_signals(sandi::SANDIdot, prot::Protocol)
    fextra = 1.0-sum(sandi.fracs)
    signals = sandi.fracs[1].*compartment_signals(sandi.soma,prot) .+ sandi.fracs[2].*compartment_signals(sandi.neurite,prot) .+ fextra.*compartment_signals(sandi.extra,prot) .+ sandi.fracs[3]
    return signals
end

function model_signals(sandi::SANDI, prot::Protocol)

    fextra = 1.0-sum(sandi.fracs)
    signals = sandi.fracs[1].*compartment_signals(sandi.soma,prot) .+ sandi.fracs[2].*compartment_signals(sandi.neurite,prot) .+ fextra.*compartment_signals(sandi.extra,prot)
    return signals
end

function model_signals(sandi::MTE_SANDI, prot::Protocol)

    fextra = 1.0-sum(sandi.fracs)
    signals = (sandi.fracs[1].*compartment_signals(sandi.soma,prot) .+ sandi.fracs[2].*compartment_signals(sandi.neurite,prot) .+ fextra.*compartment_signals(sandi.extra,prot)) .* sandi.S0norm
    return signals
end

function model_signals(model::MTE_SMT, prot::Protocol)

    signals = (model.fracs.*compartment_signals(model.axon,prot) .+ (1.0.-model.fracs).*compartment_signals(model.extra,prot)) .* model.S0norm
    return signals
end

"""
    print_model(model)
    
Helper function to check all parameters in a model
"""
function print_model(model::BiophysicalModel)
    
    println(typeof(model),":")
    for field in fieldnames(typeof(model))
        
        comp = getfield(model,field)
        subfield = fieldnames(typeof(comp))
     
        println(field,subfield)
    end

end

##########################
# dev
# test mutating implementation
function model_signals!(signals::Vector{Float64},excaliber::ExCaliber, prot::Protocol)
    
    signals .= 0.0
    signals_com = similar(signals)
    fextra = 1-sum(excaliber.fracs)

    compartment_signals!(signals_com,excaliber.axon,prot)
    signals .= signals .+ excaliber.fracs[1].*signals_com

    compartment_signals!(signals_com,excaliber.extra,prot)
    signals .= signals .+ fextra.*signals_com

    compartment_signals!(signals_com,excaliber.csf,prot)
    signals .= signals .+ excaliber.fracs[2].*signals_com

    signals .= signals .+ excaliber.fracs[3]

    return
end

# test mutating implementation
function model_signals!(signals::Vector{Float64},signals_com::Vector{Float64},excaliber::ExCaliber, prot::Protocol)
    
    signals .= 0.0
    fextra = 1-sum(excaliber.fracs)

    compartment_signals!(signals_com,excaliber.axon,prot)
    signals .= signals .+ excaliber.fracs[1].*signals_com

    compartment_signals!(signals_com,excaliber.extra,prot)
    signals .= signals .+ fextra.*signals_com

    compartment_signals!(signals_com,excaliber.csf,prot)
    signals .= signals .+ excaliber.fracs[2].*signals_com

    signals .= signals .+ excaliber.fracs[3]

    return
end

# update parameter links and get model signals
function model_signals(model::BiophysicalModel,prot::Protocol,links::Tuple{Vararg{Pair{String,String}}})
    PMI.update!(model,links)
    model_signals(model,prot)
end


