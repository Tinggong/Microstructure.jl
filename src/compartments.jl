# This script builds compartment structs with fields of relevant tissue parameters and 
# forward functions inferencing signals from the compartment model and imaging protocol. 
#
# Featuring spherical mean based models with compartmental relaxation-weighting.

using LinearAlgebra, SpecialFunctions

export Cylinder, 
    Stick, 
    Zeppelin, 
    Iso, 
    Sphere, 
    compartment_signals, 
    Compartment, 
    smt_signals
# export compartment_signals!, smt_signals!

"""
Compartment Type is an abstract type that includes the Cylinder, Stick, Zeppelin, Sphere and Iso type. 
A Compartment Type object contains relevant tissue parameters that affect the MRI signals.
Each type of compartment contain a 't2' field for combined-diffusion-relaxometry imaging. 
When your data supports only T2-weighted compartment modelling, i.e. acquired with single-TE, 
set the 't2' field to zero for conventional dMRI modelling. 
"""
abstract type Compartment end

"""
    Cylinder(
        da::Float64, 
        dpara::Float64, 
        d0::Float64, 
        t2::Float64
        )

Return a Cylinder Type object with the cylinder diameter 'da', parallel diffusivity 'dpara', 
the intrinsic diffusivity 'd0' and the T2 relaxation time 't2'. 

# Examples
```julia-repl
julia> Cylinder(da = 3.0e-6, dpara = 1.8e-9, d0 = 1.7e-9, t2 = 90e-3)
Cylinder(3.0e-6, 1.8e-9, 1.7e-9, 0.09)
```
"""
Base.@kwdef mutable struct Cylinder <: Compartment
    da::Float64 = 3.0e-6
    dpara::Float64 = 0.6e-9
    d0::Float64 = 0.6e-9
    t2::Float64 = 0.0
end

"""
    Stick(dpara::Float64, t2::Float64)

Return a Stick Type object with parallel diffusivity 'dpara' and T2 relaxation time 't2'. 
The perpendicular diffusivity of a Stick model is zero. 

# Examples
```julia-repl
julia> Stick(dpara = 1.7e-6, t2 = 60e-3)
Stick(1.7e-6, 0.06)
```
"""
Base.@kwdef mutable struct Stick <: Compartment
    dpara::Float64 = 0.6e-9
    t2::Float64 = 0.0
end

"""
    Zeppelin(
        dpara::Float64, 
        dperp_frac::Float64, 
        t2::Float64
        )

Return a Zeppelin Type object with parallel diffusivity 'dpara', axially symmetric 
perpendicular diffusivity represented as a fraction of the parallel diffusivity 'dperp_frac',
and the T2 relaxation time 't2'.

# Examples
```julia-repl
julia> Zeppelin(dpara = 1.7e-6, dperp_frac = 0.5, t2 = 0.0)
Zeppelin(1.7e-6, 0.5, 0.0)
```
"""
Base.@kwdef mutable struct Zeppelin <: Compartment
    dpara::Float64 = 0.6e-9
    dperp_frac::Float64 = 0.5
    t2::Float64 = 0.0
end

"""
    Sphere(
        diff::Float64, 
        size::Float64, 
        t2::Float64
        )

Return a Sphere Type object with diffusivity within sphere 'diff', spherical radius 'size',
and T2 relaxation time 't2'.

# Examples
```julia-repl
julia> Sphere(diff = 3.0e-9, size = 8.0e-6, t2 = 45e-3)
Sphere(3.0e-9, 8.0e-6, 0.045)
```
"""
Base.@kwdef mutable struct Sphere <: Compartment
    diff::Float64 = 2e-9
    size::Float64 = 4e-6
    t2::Float64 = 0.0
end

"""
    Iso(diff::Float64, t2=Float64)

Return an isotropic tensor with diffusivity 'diff' and T2 relaxation time 't2'. 
This compartment can be used to represent CSF (diff = free water) or dot compartment (diff = 0). 
The latter is for immobile water typically seen in ex vivo tissue.

# Examples
```julia-repl
julia> Iso(diff = 3.0e-9,t2 = 2000.0e-3)
Iso(3.0e-9, 2.0)
```
```julia-repl
julia> Iso(diff = 0.0)
Iso(0.0, 0.0)
```
"""
Base.@kwdef mutable struct Iso <: Compartment
    diff::Float64 = 2e-9
    t2::Float64 = 0.0
end

"""
    compartment_signals(model::Compartment,protocol::Protocol)

Return compartment signals given a compartment object 'model' and a imaging 'protocol'. 
'model' can be the Cylinder/Zeppelin/Stick/Sphere/Iso Type. When t2 in compartment 'model' is set as default (0), 
relaxation-weightings are not considered.
"""
function compartment_signals(model::Cylinder, prot::Protocol)

    # use this vector repeatedly to collect signals
    signals = zeros(length(prot.bval))

    # these two steps are not counted in allocations when using StaticVectors (for BesselJ_Roots)
    alphm = BesselJ_RootsCylinder ./ (model.da ./ 2.0)
    c1 =
        1.0 ./ (
            model.d0 .^ 2.0 .* alphm .^ 6.0 .*
            ((model.da ./ 2.0) .^ 2.0 .* alphm .^ 2.0 .- 1.0)
        )

    # these two will not be allocated if protocol is decleared using SVector
    c2 = .-prot.tsmalldel .- prot.tdelta
    c3 = prot.tsmalldel .- prot.tdelta

    ### this is not faster
    #signals = MVector{N,Float64}(zeros(length(prot.bval),)) 
    #c2 = SVector{N,Float64}(-prot.tsmalldel .- prot.tdelta)
    #c3 = SVector{N,Float64}(prot.tsmalldel .- prot.tdelta)

    for i in 1:(10::Int) # up to 10th order
        a = model.d0 .* alphm[i] .^ 2.0
        signals .=
            signals .+
            c1[i] .* (
                2.0 .* a .* prot.tsmalldel .- 2.0 .+ 2.0 .* exp.(-a .* prot.tsmalldel) .+
                2.0 .* exp.(-a .* prot.tdelta) .- exp.(a .* c2) .- exp.(a .* c3)
            )
    end

    signals .= -2 .* gmr .^ 2 .* prot.gvec .^ 2 .* signals
    signals .=
        exp.(signals) .* sqrt.(pi ./ 4.0 ./ (prot.bval .* model.dpara .+ signals)) .*
        erf.(sqrt.(prot.bval .* model.dpara .+ signals))

    signals[findall(iszero, prot.bval)] .= 1.0

    iszero(model.t2) && return signals
    return signals .* exp.(-prot.techo ./ model.t2)
end

# Stick signals
function compartment_signals(model::Stick, prot::Protocol)
    signals = smt_signals(prot, model.dpara, 0.0)

    iszero(model.t2) && return signals
    return signals .* exp.(-prot.techo ./ model.t2)
end

# Zeppelin signals
function compartment_signals(model::Zeppelin, prot::Protocol)
    dperp = model.dpara .* model.dperp_frac
    signals = smt_signals(prot, model.dpara, dperp)

    iszero(model.t2) && return signals # t2 not considered
    return signals .* exp.(-prot.techo ./ model.t2)
end

# Sphere signals
function compartment_signals(model::Sphere, prot::Protocol)
    signals = zeros(length(prot.tdelta))

    alphm = BesselJ_RootsSphere ./ model.size
    c1 = alphm .^ (-4.0) ./ (alphm .^ 2.0 .* model.size .^ 2.0 .- 2.0)

    c2 = -prot.tsmalldel .- prot.tdelta
    c3 = prot.tsmalldel .- prot.tdelta

    for i in 1:(31::Int) #eachindex(alphm)
        a = model.diff .* alphm[i] .^ 2.0
        signals .=
            signals .+
            c1[i] .* (
                2.0 .* prot.tsmalldel .-
                (
                    2.0 .+ exp.(a .* c3) .- 2.0 .* exp.(-a .* prot.tsmalldel) .-
                    2.0 .* exp.(-a .* prot.tdelta) .+ exp.(a .* c2)
                ) ./ a
            )
    end

    signals .= exp.(-2.0 .* gmr .^ 2.0 .* prot.gvec .^ 2.0 ./ model.diff .* signals)

    signals[findall(iszero, prot.bval)] .= 1.0

    iszero(model.t2) && return signals # t2 not considered
    return signals .* exp.(-prot.techo ./ model.t2)
end

# Isotropic signals
function compartment_signals(model::Iso, prot::Protocol)
    iszero(model.t2) && return exp.(-prot.bval .* model.diff)
    return exp.(-prot.bval .* model.diff .- prot.techo ./ model.t2)
end

# Spherical mean signals given parallel and perpendicular diffusivities
function smt_signals(prot::Protocol, dpara::Float64, dperp::Float64)
    signals = prot.bval .* (dpara .- dperp)

    signals .=
        exp.(-prot.bval .* dperp) .* sqrt.(pi ./ 4.0 ./ signals) .* erf.(sqrt.(signals))

    signals[findall(iszero, prot.bval)] .= 1.0

    return signals
end

###################### to test mutating functions ######################################
"""
under testing mutating signals
"""
function compartment_signals!(signals::Vector{Float64}, model::Cylinder, prot::Protocol)

    # set to 0 and collect signals
    signals .= 0.0

    # these two steps are not counted in allocations when using StaticVectors (for BesselJ_Roots)
    alphm = BesselJ_RootsCylinder ./ (model.da ./ 2.0)
    c1 =
        1.0 ./ (
            model.d0 .^ 2.0 .* alphm .^ 6.0 .*
            ((model.da ./ 2.0) .^ 2.0 .* alphm .^ 2.0 .- 1.0)
        )

    # these two will not be allocated if protocol is decleared using SVector
    c2 = .-prot.tsmalldel .- prot.tdelta
    c3 = prot.tsmalldel .- prot.tdelta

    for i in 1:(10::Int) # up to 10th order
        a = model.d0 .* alphm[i] .^ 2.0
        signals .=
            signals .+
            c1[i] .* (
                2.0 .* a .* prot.tsmalldel .- 2.0 .+ 2.0 .* exp.(-a .* prot.tsmalldel) .+
                2.0 .* exp.(-a .* prot.tdelta) .- exp.(a .* c2) .- exp.(a .* c3)
            )
    end

    signals .= -2 .* gmr .^ 2 .* prot.gvec .^ 2 .* signals
    signals .=
        exp.(signals) .* sqrt.(pi ./ 4.0 ./ (prot.bval .* model.dpara .+ signals)) .*
        erf.(sqrt.(prot.bval .* model.dpara .+ signals))

    signals[findall(iszero, prot.bval)] .= 1.0

    iszero(model.t2) && return signals
    signals .= signals .* exp.(-prot.techo ./ model.t2)
    return signals
end

# Stick signals
function compartment_signals!(signals::Vector{Float64}, model::Stick, prot::Protocol)
    smt_signals!(signals, prot, model.dpara, 0.0)

    iszero(model.t2) && return signals
    signals .= signals .* exp.(-prot.techo ./ model.t2)
    return signals
end

# Zeppelin signals
function compartment_signals!(signals::Vector{Float64}, model::Zeppelin, prot::Protocol)
    dperp = model.dpara .* model.dperp_frac
    smt_signals!(signals, prot, model.dpara, dperp)

    iszero(model.t2) && return signals # t2 not considered
    signals .= signals .* exp.(-prot.techo ./ model.t2)
    return signals
end

# Sphere signals
function compartment_signals!(signals::Vector{Float64}, model::Sphere, prot::Protocol)
    signals .= 0.0

    alphm = BesselJ_RootsSphere ./ model.size
    c1 = alphm .^ (-4.0) ./ (alphm .^ 2.0 .* model.size .^ 2.0 .- 2.0)

    c2 = -prot.tsmalldel .- prot.tdelta
    c3 = prot.tsmalldel .- prot.tdelta

    for i in 1:(31::Int) #eachindex(alphm)
        a = model.diff .* alphm[i] .^ 2.0
        signals .=
            signals .+
            c1[i] .* (
                2.0 .* prot.tsmalldel .-
                (
                    2.0 .+ exp.(a .* c3) .- 2.0 .* exp.(-a .* prot.tsmalldel) .-
                    2.0 .* exp.(-a .* prot.tdelta) .+ exp.(a .* c2)
                ) ./ a
            )
    end

    signals .= exp.(-2.0 .* gmr .^ 2.0 .* prot.gvec .^ 2.0 ./ model.diff .* signals)

    signals[findall(iszero, prot.bval)] .= 1.0

    iszero(model.t2) && return signals # t2 not considered
    signals .= signals .* exp.(-prot.techo ./ model.t2)
    return signals
end

# Isotropic signals
function compartment_signals!(signals::Vector{Float64}, model::Iso, prot::Protocol)
    signals .= exp.(-prot.bval .* model.diff)
    iszero(model.t2) && return nothing
    signals .= signals .* exp.(-prot.techo ./ model.t2)
    return signals
end

# Spherical mean signals given parallel and perpendicular diffusivities
function smt_signals!(
    signals::Vector{Float64}, prot::Protocol, dpara::Float64, dperp::Float64
)
    signals .= prot.bval .* (dpara .- dperp)
    signals .=
        exp.(-prot.bval .* dperp) .* sqrt.(pi ./ 4.0 ./ signals) .* erf.(sqrt.(signals))
    signals[findall(iszero, prot.bval)] .= 1.0

    return signals
end
