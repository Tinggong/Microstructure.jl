# Tissue models
#
# This script builds compartment structs with fields of relevant tissue parameters and 
# forward functions inferencing signals from the compartment model and imaging protocol. 
#
# Featuring spherical mean based models with compartmental relaxation-weighting.

using LinearAlgebra, SpecialFunctions

export Protocol, Cylinder, Stick, Zeppelin, Iso, Sphere, compartment_signals, Compartment, smt_signals
# export compartment_signals!, smt_signals!

"""
Tissue compartments belong to Compartment type. A compartment contains relevant tissue parameters that affect MRI signals
"""
abstract type Compartment end

"""
    Cylinder(da,dpara,d0,t2)
Cylinder model using Van Gelderen, P.
parameters: cylinder diameter 'da'
            parallel diffusivity 'dpara'
            intrinsic diffusivity 'd0'
            T2 relaxation time 't2'
"""
Base.@kwdef mutable struct Cylinder <: Compartment
    da::Float64 = 3.0e-6
    dpara::Float64 = 0.6e-9
    d0::Float64 = 0.6e-9
    t2::Float64 = 0.0
end

"""
    Stick(dpara,t2)
stick model with zero perpendicular diffusivity
parameters:  parallel diffusivity 'dpara'
            T2 relaxation time 't2'
"""
Base.@kwdef mutable struct Stick <: Compartment
    dpara::Float64 = 0.6e-9
    t2::Float64 = 0.0
end

"""
    Zeppelin(dpara,dperp_frac,t2)
zeppelin/tensor model
parameters: parallel diffusivity 'dpara'
            perpendicular diffusivity represented as a fraction of dpara 'dperp_frac'
            T2 relaxation time 't2'
"""
Base.@kwdef mutable struct Zeppelin <: Compartment
    dpara::Float64 = 0.6e-9
    dperp_frac::Float64 = 0.5
    t2::Float64 = 0.0
end

"""
    Sphere(diff,size,t2)
sphere model (Neuman)
parameters: diffusivity within sphere 'diff'
            sphere radius 'size'
            T2 relaxation time 't2'
"""
Base.@kwdef mutable struct Sphere <: Compartment
    diff::Float64 = 2e-9
    size::Float64 = 4e-6
    t2::Float64 = 0.0
end

"""
    Iso(diff,t2)
dot/isotropic tensor
parameters: diffusivity 'diff'
            T2 relaxation time 't2'
This compartment can be used to represent CSF (diff = free water) or dot compartment (diff = 0). 
The latter is for immobile typically seen in ex vivo tissue
"""
Base.@kwdef mutable struct Iso <: Compartment
    diff::Float64 = 2e-9
    t2::Float64 = 0.0
end

"""
    compartment_signals(model,protocol)
Return compartment signals given a tissue compartment <: Compartment and imaging protocl::Protocol
Models: Cylinder/Zeppelin/Stick/Sphere/Iso
When t2 in compartment is set as default (0), relaxation-weighting not considered and conventional dMRI modelling used
"""
# Cylinder signals
function compartment_signals(model::Cylinder, prot::Protocol)
    
    # use this vector repeatedly to collect signals
    signals = zeros(length(prot.bval),)

    # these two steps are not counted in allocations when using StaticVectors (for BesselJ_Roots)
    alphm = BesselJ_RootsCylinder ./ ( model.da ./ 2.0 )
    c1 = 1.0 ./ ( model.d0.^ 2.0 .* alphm.^6.0 .* ( (model.da./2.0).^2.0 .* alphm.^2.0 .- 1.0))

    # these two will not be allocated if protocol is decleared using SVector
    c2 = .-prot.tsmalldel .- prot.tdelta
    c3 = prot.tsmalldel .- prot.tdelta

    ### this is not faster
    #signals = MVector{N,Float64}(zeros(length(prot.bval),)) 
    #c2 = SVector{N,Float64}(-prot.tsmalldel .- prot.tdelta)
    #c3 = SVector{N,Float64}(prot.tsmalldel .- prot.tdelta)

    for i in 1:10::Int # up to 10th order

        a = model.d0 .* alphm[i].^2.0
        signals .= signals .+ c1[i] .* (2.0 .* a.* prot.tsmalldel .- 2.0 .+ 2.0.*exp.(-a.* prot.tsmalldel) .+ 2.0.*exp.(-a.* prot.tdelta) .- exp.(a.*c2) .- exp.(a.*c3))

    end

    signals .= -2 .* gmr.^2 .* prot.gvec.^2 .* signals
    signals .= exp.(signals) .* sqrt.(pi./4.0./(prot.bval.*model.dpara.+signals)) .* erf.( sqrt.(prot.bval.*model.dpara.+signals) )

    signals[findall(iszero,prot.bval)] .= 1.0

    iszero(model.t2) && return signals 
    return signals.*exp.(-prot.techo./model.t2)
end

# Stick signals
function compartment_signals(model::Stick, prot::Protocol)

    signals = smt_signals(prot,model.dpara,0.0)

    iszero(model.t2) && return signals 
    return signals.*exp.(-prot.techo./model.t2)
end

# Zeppelin signals
function compartment_signals(model::Zeppelin,prot::Protocol)

    dperp = model.dpara.*model.dperp_frac
    signals = smt_signals(prot,model.dpara,dperp)

    iszero(model.t2) && return signals # t2 not considered
    return signals.*exp.(-prot.techo./model.t2)
end

# Sphere signals
function compartment_signals(model::Sphere,prot::Protocol)

    signals = zeros(length(prot.tdelta),)
    
    alphm = BesselJ_RootsSphere ./ model.size
    c1 = alphm.^(-4.0) ./ (alphm.^2.0 .* model.size.^2.0 .- 2.0)
   
    c2 = -prot.tsmalldel .- prot.tdelta 
    c3 = prot.tsmalldel .- prot.tdelta

    for i in 1:31::Int #eachindex(alphm)

        a = model.diff .* alphm[i].^2.0
        signals .= signals .+ c1[i] .* (2.0 .* prot.tsmalldel .- (2.0 .+ exp.(a.*c3) .- 2.0.*exp.(-a.*prot.tsmalldel) .- 2.0.*exp.(-a.*prot.tdelta) .+ exp.(a.*c2))./a)
    
    end

    signals .= exp.( -2.0 .* gmr.^2.0 .* prot.gvec.^2.0 ./ model.diff .* signals) 
   
    signals[findall(iszero,prot.bval)] .= 1.0

    iszero(model.t2) && return signals # t2 not considered
    return signals.*exp.(-prot.techo./model.t2)
end

# Isotropic signals
function compartment_signals(model::Iso,prot::Protocol)
   
    iszero(model.t2) && return exp.(-prot.bval.*model.diff)
    return exp.(-prot.bval.*model.diff.-prot.techo./model.t2)

end

# Spherical mean signals given parallel and perpendicular diffusivities
function smt_signals(prot::Protocol, dpara::Float64, dperp::Float64)

    signals = prot.bval.*(dpara.-dperp)

    signals .= exp.(-prot.bval.*dperp).*sqrt.(pi./4.0./signals).*erf.(sqrt.(signals))
    
    signals[findall(iszero,prot.bval)] .= 1.0
   
    return signals

end


###################### to test mutating functions ######################################
"""
under testing mutating signals
"""
function compartment_signals!(signals::Vector{Float64},model::Cylinder, prot::Protocol)
    
    # set to 0 and collect signals
    signals .= 0.0

    # these two steps are not counted in allocations when using StaticVectors (for BesselJ_Roots)
    alphm = BesselJ_RootsCylinder ./ ( model.da ./ 2.0 )
    c1 = 1.0 ./ ( model.d0.^ 2.0 .* alphm.^6.0 .* ( (model.da./2.0).^2.0 .* alphm.^2.0 .- 1.0))

    # these two will not be allocated if protocol is decleared using SVector
    c2 = .-prot.tsmalldel .- prot.tdelta
    c3 = prot.tsmalldel .- prot.tdelta

    for i in 1:10::Int # up to 10th order

        a = model.d0 .* alphm[i].^2.0
        signals .= signals .+ c1[i] .* (2.0 .* a.* prot.tsmalldel .- 2.0 .+ 2.0.*exp.(-a.* prot.tsmalldel) .+ 2.0.*exp.(-a.* prot.tdelta) .- exp.(a.*c2) .- exp.(a.*c3))

    end

    signals .= -2 .* gmr.^2 .* prot.gvec.^2 .* signals
    signals .= exp.(signals) .* sqrt.(pi./4.0./(prot.bval.*model.dpara.+signals)) .* erf.( sqrt.(prot.bval.*model.dpara.+signals) )

    signals[findall(iszero,prot.bval)] .= 1.0

    iszero(model.t2) && return signals
    signals .= signals.*exp.(-prot.techo./model.t2)
    return signals
end

# Stick signals
function compartment_signals!(signals::Vector{Float64},model::Stick, prot::Protocol)

    smt_signals!(signals,prot,model.dpara,0.0)

    iszero(model.t2) && return signals
    signals .= signals.*exp.(-prot.techo./model.t2)
    return signals
end

# Zeppelin signals
function compartment_signals!(signals::Vector{Float64},model::Zeppelin,prot::Protocol)

    dperp = model.dpara.*model.dperp_frac
    smt_signals!(signals,prot,model.dpara,dperp)

    iszero(model.t2) && return signals # t2 not considered
    signals .= signals.*exp.(-prot.techo./model.t2)
    return signals
end

# Sphere signals
function compartment_signals!(signals::Vector{Float64},model::Sphere,prot::Protocol)

    signals .= 0.0
    
    alphm = BesselJ_RootsSphere ./ model.size
    c1 = alphm.^(-4.0) ./ (alphm.^2.0 .* model.size.^2.0 .- 2.0)
   
    c2 = -prot.tsmalldel .- prot.tdelta 
    c3 = prot.tsmalldel .- prot.tdelta

    for i in 1:31::Int #eachindex(alphm)

        a = model.diff .* alphm[i].^2.0
        signals .= signals .+ c1[i] .* (2.0 .* prot.tsmalldel .- (2.0 .+ exp.(a.*c3) .- 2.0.*exp.(-a.*prot.tsmalldel) .- 2.0.*exp.(-a.*prot.tdelta) .+ exp.(a.*c2))./a)
    
    end

    signals .= exp.( -2.0 .* gmr.^2.0 .* prot.gvec.^2.0 ./ model.diff .* signals) 
   
    signals[findall(iszero,prot.bval)] .= 1.0

    iszero(model.t2) && return signals # t2 not considered
    signals .= signals.*exp.(-prot.techo./model.t2)
    return signals
end

# Isotropic signals
function compartment_signals!(signals::Vector{Float64},model::Iso,prot::Protocol)
   
    signals .= exp.(-prot.bval.*model.diff)
    iszero(model.t2) && return
    signals .= signals.*exp.(-prot.techo./model.t2)
    return signals
end

# Spherical mean signals given parallel and perpendicular diffusivities
function smt_signals!(signals::Vector{Float64},prot::Protocol, dpara::Float64, dperp::Float64)

    signals .= prot.bval.*(dpara.-dperp)
    signals .= exp.(-prot.bval.*dperp).*sqrt.(pi./4.0./signals).*erf.(sqrt.(signals))
    signals[findall(iszero,prot.bval)] .= 1.0
   
    return signals
end
