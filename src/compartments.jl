# This script builds compartment structs with fields of relevant tissue parameters and 
# forward functions inferencing signals from the compartment model and imaging protocol. 

using LinearAlgebra, SpecialFunctions

export Cylinder,
    Stick,
    Zeppelin,
    Iso,
    Sphere,
    compartment_signals,
    Compartment,
    smt_signals,
    Tensor,
    Stick_kernel,
    Zeppelin_kernel,
    Iso_kernel,
    fODF

"""
Compartment Type is an abstract type that includes the `Cylinder`, `Stick`, `Zeppelin`, `Sphere` and `Iso` type. 
A Compartment Type object contains relevant tissue parameters that affect the MRI signals.
Each type of compartment contain a `t2` field for combined-diffusion-T2 imaging. 
When your data supports only T2-weighted compartment modelling, i.e. acquired with single-TE, 
set the `t2` field to zero for conventional dMRI modelling. 
"""
abstract type Compartment end

"""
    Cylinder(
        da::Float64, 
        dpara::Float64, 
        d0::Float64, 
        t2::Float64
        )

Return a Cylinder Type object with the cylinder diameter `da`, parallel diffusivity `dpara`, 
the intrinsic diffusivity `d0` and the T2 relaxation time `t2`. 

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

Return a Stick Type object with parallel diffusivity `dpara` and T2 relaxation time `t2`. 
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

Return a Zeppelin Type object with parallel diffusivity `dpara`, axially symmetric 
perpendicular diffusivity represented as a fraction of the parallel diffusivity `dperp_frac`,
and the T2 relaxation time `t2`.

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

Return a Sphere Type object with diffusivity within sphere `diff`, spherical radius `size`,
and T2 relaxation time `t2`.

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

Return an isotropic tensor with diffusivity `diff` and T2 relaxation time `t2`. 
This compartment can be used to represent CSF (`diff` = free water) or dot compartment (`diff` = 0). 
The latter is for immobile water typically seen in ex vivo tissue. 
This compartment can also represent an isotropic extra-cellular environment with diffusivity `diff` slower than free water.

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

Return compartment signals given a compartment object `model` and a imaging `protocol`. 
`model` can be the `Cylinder`/`Zeppelin`/`Stick`/`Sphere`/`Iso` Type. When `t2` in compartment `model` is set as default (0), 
relaxation-weightings are not considered in the signal equation.

### References
If you use these compartments to build models, please cite the recommended references. 

# For using any compartment in current release, please cite the following references for expressions of spherical mean/power averaging:

Callaghan, P.T., Jolley, K.W., Lelievre, J., 1979. Diffusion of water in the endosperm tissue of wheat grains as studied by pulsed field gradient nuclear magnetic resonance. Biophys J 28, 133. https://doi.org/10.1016/S0006-3495(79)85164-4

Kroenke, C.D., Ackerman, J.J.H., Yablonskiy, D.A., 2004. On the nature of the NAA diffusion attenuated MR signal in the central nervous system. Magn Reson Med 52, 1052–1059. https://doi.org/10.1002/MRM.20260

Kaden, E., Kruggel, F., Alexander, D.C., 2016. Quantitative mapping of the per-axon diffusion coefficients in brain white matter. Magn Reson Med 75, 1752–1763. https://doi.org/10.1002/MRM.25734

# Consider the following reference for overview of all tissue compartments:

Panagiotaki, E., Schneider, T., Siow, B., Hall, M.G., Lythgoe, M.F., Alexander, D.C., 2012. Compartment models of the diffusion MR signal in brain white matter: A taxonomy and comparison. Neuroimage 59, 2241–2254. 

# Cylinder compartment:
Van Gelderen, P., Des Pres, D., Van Zijl, P.C.M., Moonen, C.T.W., 1994. Evaluation of Restricted Diffusion in Cylinders. Phosphocreatine in Rabbit Leg Muscle. J Magn Reson B 103, 255–260. https://doi.org/10.1006/JMRB.1994.1038

Alexander, D.C., Hubbard, P.L., Hall, M.G., Moore, E.A., Ptito, M., Parker, G.J.M., Dyrby, T.B., 2010. Orientationally invariant indices of axon diameter and density from diffusion MRI. Neuroimage 52, 1374–1389. https://doi.org/10.1016/j.neuroimage.2010.05.043

Fan, Q., Nummenmaa, A., Witzel, T., Ohringer, N., Tian, Q., Setsompop, K., Klawiter, E.C., Rosen, B.R., Wald, L.L., Huang, S.Y., 2020. Axon diameter index estimation independent of fiber orientation distribution using high-gradient diffusion MRI. Neuroimage 222. 

Andersson, M., Pizzolato, M., Kjer, H.M., Skodborg, K.F., Lundell, H., Dyrby, T.B., 2022. Does powder averaging remove dispersion bias in diffusion MRI diameter estimates within real 3D axonal architectures? Neuroimage 248. 

# Sphere compartment:
Neuman, C.H., 1974. Spin echo of spins diffusing in a bounded medium. J Chem Phys 4508–4511. https://doi.org/10.1063/1.1680931

Balinov, B., Jönsson, B., Linse, P., Söderman, O., 1993. The NMR Self-Diffusion Method Applied to Restricted Diffusion. Simulation of Echo Attenuation from Molecules in Spheres and between Planes. J Magn Reson A 104, 17–25. https://doi.org/10.1006/JMRA.1993.1184

# Stick compartment:

Behrens, T.E.J., Woolrich, M.W., Jenkinson, M., Johansen-Berg, H., Nunes, R.G., Clare, S., Matthews, P.M., Brady, J.M., Smith, S.M., 2003. Characterization and Propagation of Uncertainty in Diffusion-Weighted MR Imaging. Magn Reson Med 50, 1077–1088. https://doi.org/10.1002/MRM.10609

Panagiotaki, E., Schneider, T., Siow, B., Hall, M.G., Lythgoe, M.F., Alexander, D.C., 2012. Compartment models of the diffusion MR signal in brain white matter: A taxonomy and comparison. Neuroimage 59, 2241–2254. 

Zhang, H., Schneider, T., Wheeler-Kingshott, C.A., Alexander, D.C., 2012. NODDI: Practical in vivo neurite orientation dispersion and density imaging of the human brain. Neuroimage 61, 1000–1016. https://doi.org/10.1016/j.neuroimage.2012.03.072

# Zeppelin & Iso:

Alexander, D.C., 2008. A General Framework for Experiment Design in Diffusion MRI and Its Application in Measuring Direct Tissue-Microstructure Features. Magn Reson Med 60, 439–448. https://doi.org/10.1002/mrm.21646

Panagiotaki, E., Schneider, T., Siow, B., Hall, M.G., Lythgoe, M.F., Alexander, D.C., 2012. Compartment models of the diffusion MR signal in brain white matter: A taxonomy and comparison. Neuroimage 59, 2241–2254. 

Zhang, H., Schneider, T., Wheeler-Kingshott, C.A., Alexander, D.C., 2012. NODDI: Practical in vivo neurite orientation dispersion and density imaging of the human brain. Neuroimage 61, 1000–1016. https://doi.org/10.1016/j.neuroimage.2012.03.072

# Compartmental T2-weighting: 

Veraart, J., Novikov, D.S., Fieremans, E., 2017. TE dependent Diffusion Imaging (TEdDI) distinguishes between compartmental T2 relaxation times. https://doi.org/10.1016/j.neuroimage.2017.09.030

Lampinen, B., Szczepankiewicz, F., Novén, M., van Westen, D., Hansson, O., Englund, E., Mårtensson, J., Westin, C.F., Nilsson, M., 2019. Searching for the neurite density with diffusion MRI: Challenges for biophysical modeling. Hum Brain Mapp 40, 2529–2545. https://doi.org/10.1002/hbm.24542

Gong, T., Tong, Q., He, H., Sun, Y., Zhong, J., Zhang, H., 2020. MTE-NODDI: Multi-TE NODDI for disentangling non-T2-weighted signal fractions from compartment-specific T2 relaxation times. Neuroimage 217. https://doi.org/10.1016/j.neuroimage.2020.116906

Gong, T., Tax, C.M., Mancini, M., Jones, D.K., Zhang, H., Palombo, M., 2023. Multi-TE SANDI: Quantifying compartmental T2 relaxation times in the grey matter. Toronto.

Kernel functions of the Zeppelin/Stick/Iso compartments are also included for standard model imaging using higher order rotational invariants:

Novikov, D.S., Veraart, J., Jelescu, I.O. and Fieremans, E., 2018. Rotationally-invariant mapping of scalar and orientational metrics of neuronal microstructure with diffusion MRI. NeuroImage, 174, pp.518-538.

Novikov, D.S., Fieremans, E., Jespersen, S.N. and Kiselev, V.G., 2019. Quantifying brain microstructure with diffusion MRI: Theory and parameter estimation. NMR in Biomedicine, 32(4), p.e3998.

Coelho, S., Baete, S.H., Lemberskiy, G., Ades-Aron, B., Barrol, G., Veraart, J., Novikov, D.S. and Fieremans, E., 2022. Reproducibility of the standard model of diffusion in white matter on clinical MRI systems. NeuroImage, 257, p.119290.
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

###################### to add orientated tissue compartments to test #####################################
"""
A tensor compartment 
"""
Base.@kwdef mutable struct Tensor <: Compartment
    dir_polar::Float64        # constrain this to [0, pi/2] so the orientations are towards one of the hemisphere
    dir_azimuth::Float64      # within [0, 2*pi]
    λ1::Float64 = 1.7e-9      # lambda 1
    λ2_frac::Float64 = 0.1    # lambda 2 to 1 fraction
    λ3_frac::Float64 = 0.1    # lambda 3 to 1 fraction
    t2::Float64 = 0.0         # t2 of the tensor compartment
end

function compartment_signals(model::Tensor, protocol::Protocol)

    # get rotation matrix based on tensor orientation
    R = rotation(model.dir_polar, model.dir_azimuth)

    # diffusivities of the tensor
    D = [
        bundle.λ1 0 0
        0 bundle.λ1*bundle.λ2_frac 0
        0 0 bundle.λ1*bundle.λ3_frac
    ]

    # get the diffusion tensor 
    DT = R*D*R'

    iszero(model.t2) && return dt_signals(DT, protocol) # t2 not considered
    return dt_signals(DT, protocol) .* exp.(-prot.techo ./ model.t2)
end

"""
Return rotation matrix for rotating x vector ([pi/2, 0]/[1, 0, 0]) to the given orienation
"""
function rotation(polar::AbstractFloat, azimuth::AbstractFloat)
    ϕ = polar - pi/2
    θ = azimuth

    Ry = [
        cos(ϕ) 0 sin(ϕ)
        0 1 0
        -sin(ϕ) 0 cos(ϕ)
    ]

    Rz = [
        cos(θ) -sin(θ) 0
        sin(θ) cos(θ) 0
        0 0 1
    ]

    return Ry*Rz
end

"""
Return diffusion tensor signals
"""
function dt_signals(dt::Matrix{<:AbstractFloat}, prot::Protocol)
    signals = exp.(
        .- prot.bval .* (
            prot.bvec[:, 1] .^ 2.0 .* dt[1, 1] .+ prot.bvec[:, 2] .^ 2.0 .* dt[2, 2] .+
            prot.bvec[:, 3] .^ 2.0 .* dt[3, 3] .+
            2.0 .* prot.bvec[:, 1] .* prot.bvec[:, 2] .* dt[1, 2] .+
            2.0 .* prot.bvec[:, 1] .* prot.bvec[:, 3] .* dt[1, 3] .+
            2.0 .* prot.bvec[:, 2] .* prot.bvec[:, 3] .* dt[2, 3]
        ),
    )

    signals[findall(iszero, prot.bval)] .= 1.0

    return signals
end

"""
Kernel functions from each compartment in standard model imaging. 
These compartments contains the same parameters but the compartment_signals predicts rotational invariants at orders higher than zero
"""
Base.@kwdef mutable struct Stick_kernel <: Compartment
    dpara::Float64 = 0.6e-9
    t2::Float64 = 0.0
end

Base.@kwdef mutable struct Zeppelin_kernel <: Compartment
    dpara::Float64 = 0.6e-9
    dperp_frac::Float64 = 0.5
    t2::Float64 = 0.0
end

Base.@kwdef mutable struct Iso_kernel <: Compartment
    diff::Float64 = 3.0e-9
    t2::Float64 = 2.0
end

"""
The fiber orientation distribution functions represented in SH basis

lmax: the maximum SH order represented
p2: a measure of anisotropy; rotational invariant of fODF at l=2
p4_frac: a fraction of p2; rotational invariant of fODF at l=4
plm: all the SH coefficients

"""
Base.@kwdef mutable struct fODF <: Compartment
    lmax::Int = 2
    p2::Float64 = 0.5
    p4_frac::Float64 = 0.5   # p4 is represented as p2 multiplied by a fraction between 0-1; only meaningful when lmax = 4
    plm::Vector{Float64} = ones(Int((lmax+1)*(lmax+2)/2)) # the SH coefficents of the fODF
end

function compartment_signals(model::Stick_kernel, prot::Protocol)
    Klb = zeros(length(prot.lmeas))

    for i in 1:length(prot.lmeas)
        Klb[i] = abs(
            sum(
                Microstructure.Pl["weights"] .*
                exp.(-prot.bval[i] .* model.dpara .* Microstructure.Pl["nodes"] .^ 2) .*
                Microstructure.Pl[prot.lmeas[i]],
            ),
        )
    end

    Klb[findall(iszero, prot.bval)] .= 1.0

    iszero(model.t2) && return Klb
    return Klb .* exp.(-prot.techo ./ model.t2)
end

function compartment_signals(model::Zeppelin_kernel, prot::Protocol)
    Klb = zeros(length(prot.lmeas))

    for i in 1:length(prot.lmeas)
        Klb[i] = abs(
            sum(
                Microstructure.Pl["weights"] .* exp.(
                    -prot.bval[i] .* model.dpara .* (
                        Microstructure.Pl["nodes"] .^ 2 .+
                        model.dperp_frac .* (1.0 .- Microstructure.Pl["nodes"] .^ 2)
                    ),
                ) .* Microstructure.Pl[prot.lmeas[i]],
            ),
        )
    end

    Klb[findall(iszero, prot.bval)] .= 1.0

    iszero(model.t2) && return Klb
    return Klb .* exp.(-prot.techo ./ model.t2)
end

function compartment_signals(model::Iso_kernel, prot::Protocol)
    Klb = zeros(length(prot.lmeas))

    for i in 1:length(prot.lmeas)
        Klb[i] = abs(
            sum(
                Microstructure.Pl["weights"] .* exp.(-prot.bval[i] .* model.diff) .*
                Microstructure.Pl[prot.lmeas[i]],
            ),
        )
    end

    Klb[findall(iszero, prot.bval)] .= 1.0

    iszero(model.t2) && return Klb
    return Klb .* exp.(-prot.techo ./ model.t2)
end
