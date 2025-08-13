# I/O functions for images and protocols
# Including functions to fit measurements to spherical harmonic bases

using Fibers, DelimitedFiles, Statistics, StaticArrays

export dMRI,
    Protocol,
    spherical_fit,
    spherical_mean,
    dmri_write,
    dmri_read,
    dmri_read_times,
    dmri_read_times!,
    dmri_read_time
"""
    dMRI(nifti::MRI, 
    tdelta::Vector{Float64}, 
    dsmalldel::Vector{Float64}, 
    techo::Vector{Float64}, 
    smt::Bool,
    nmeas::Vector{Int}
    lmeas::Vector{Int})

Return a dMRI Type object with MRI object `nifti`, and additional volume-wise 
experimental settings `tdelta`, `tsmalldel`, `techo`, `smt` for identifing smt signals,
`nmeas` for the number of measurements, and `lmeas` for the order of rotational invariants. 
"""
mutable struct dMRI
    nifti::MRI
    tdelta::Vector{Float64}
    tsmalldel::Vector{Float64}
    techo::Vector{Float64}
    smt::Bool
    nmeas::Vector{Int}
    lmeas::Vector{Int}
end

"""
initialize dMRI structure from MRI or Array
"""
function dMRI(mri::MRI)
    dMRI(
        mri,
        Vector{Float64}(zeros(mri.nframes)),
        Vector{Float64}(zeros(mri.nframes)),
        Vector{Float64}(zeros(mri.nframes)),
        false,
        Vector{Int}(ones(mri.nframes)),
        Vector{Int}(zeros(mri.nframes)),
    )
end

"""
Read nifti and text files to dMRI object; variable number of input text files
"""
function dmri_read(imagefile::String, infiles::String...)
    mri = mri_read(imagefile)
    dmri = dmri_read_times(mri, infiles)
    return dmri
end

"""
Called by dmri_read; Tuple holds variable number of input text files
Run alone to construct a dMRI from mri object and text files
"""
function dmri_read_times(mri::MRI, infiles::Tuple{Vararg{String}})
    dmri = dMRI(mri)
    dmri_read_times!(dmri, infiles)

    # round b table; .element-wise operation
    dmri.nifti.bval .= round.(dmri.nifti.bval ./ 50.0) .* 50.0
    # dmri.nifti.bval[dmri.nifti.bval .<= 100.0] .= 0.0

    # set delta/smalldel to 0 when b = 0
    index = iszero.(dmri.nifti.bval)
    dmri.tdelta[index] .= 0.0
    dmri.tsmalldel[index] .= 0.0

    return dmri
end
"""
read txt files and update dMRI fields according to file extensions
"""
function dmri_read_times!(dmri::dMRI, infiles::Tuple{Vararg{String}})
    for file in infiles
        tab, ext = dmri_read_time(file)
        if ext == "techo" || ext == "TE"
            dmri.techo = vec(tab)
        elseif ext == "tdelta"
            dmri.tdelta = vec(tab)
        elseif ext == "tsmalldel"
            dmri.tsmalldel = vec(tab)
        elseif ext == "bvals" || ext == "bval"
            dmri.nifti.bval = vec(tab)
        elseif ext == "bvecs" || ext == "bvec"
            n1, n2 = size(tab)
            if n1 >= n2
                dmri.nifti.bvec = tab
            else
                dmri.nifti.bvec = tab'
            end
        else
            error("Unindentified file extension")
        end
    end

    return dmri
end

"""
read vectors and get file extension from input file 
"""
function dmri_read_time(infile::String)
    if !isfile(infile)
        error("Could not find input file")
    end

    # find input file extention
    idot = findlast(isequal('.'), infile)
    ext = lowercase(infile[(idot + 1):end])

    # read file
    tab = readdlm(infile)

    return tab, ext
end

"""
    Protocol(
    bval::Vector{Float64}
    techo::Vector{Float64}
    tdelta::Vector{Float64}
    tsmalldel::Vector{Float64}
    nmeas::Vector{Float64}
    lmeas::Vector{Float64}
    gvec::Vector{Float64}
    bvec::Matrix{Float64}
    nmeas::Vector{Float64}
    )

Return a Protocol Type object to hold parameters in acquisition protocol relavent for modelling 
including b-values, tcho times, diffusion gradient seperation, duration, strengh, direction and the number of measurements. 
Unit convention: most text files use s/mm^2 for b-values and ms for time while they are converted to SI unit in the Protocol.
b-values (s/m^2); time (s); size (m); G (T/m) 

    Protocol(
        filename::String
    )
Return a Protocol Type object from a b-table file generated from spherical_mean function.
    
    Protocol(
        bval::Vector{Float64},
        techo::Vector{Float64},
        tdelta::Vector{Float64},
        tsmalldel::Vector{Float64},
    )
Calculate `gvec` and return a Ptotocol Type object from provided parameters; other fields are not useful

    Protocol(
        dmri::dMRI
    )
Return a Protocol Type object from a dMRI object.
"""
struct Protocol
    bval::Vector{Float64}
    techo::Vector{Float64}
    tdelta::Vector{Float64}
    tsmalldel::Vector{Float64}
    nmeas::Vector{Int}
    lmeas::Vector{Int}
    gvec::Vector{Float64}
    bvec::Matrix{Float64}
    #qvec=gmr.*tsmalldel.*gvec
end

# make protocols from acq vectors
#function Protocol(bval::SVector{<:Any,Float64}, techo::SVector{<:Any,Float64}, tdelta::SVector{<:Any,Float64}, tsmalldel::SVector{<:Any,Float64})
#    gvec = 1.0 ./ gmr ./ tsmalldel .* sqrt.(bval ./ (tdelta .- tsmalldel ./ 3.0))
#    Protocol(bval, techo, tdelta, tsmalldel, gvec)
#end

function Protocol(
    bval::Vector{Float64},
    techo::Vector{Float64},
    tdelta::Vector{Float64},
    tsmalldel::Vector{Float64},
)
    gvec = 1.0 ./ gmr ./ tsmalldel .* sqrt.(bval ./ (tdelta .- tsmalldel ./ 3.0))
    bvec = zeros(length(bval), 3)
    nmeas = Int.(ones(length(bval)))
    lmeas = Int.(zeros(length(bval)))
    return Protocol(bval, techo, tdelta, tsmalldel, nmeas, lmeas, gvec, bvec)
end

function Protocol(
    bval::Vector{Float64},
    techo::Vector{Float64},
    tdelta::Vector{Float64},
    tsmalldel::Vector{Float64},
    nmeas::Vector{Int},
    lmeas::Vector{Int},
)
    gvec = 1.0 ./ gmr ./ tsmalldel .* sqrt.(bval ./ (tdelta .- tsmalldel ./ 3.0))
    bvec = zeros(length(bval), 3)
    return Protocol(bval, techo, tdelta, tsmalldel, nmeas, lmeas, gvec, bvec)
end

"""
Make protocol from a dMRI object
"""
function Protocol(dmri::dMRI)
    protocol = Protocol(
        dmri.nifti.bval .* 1.0e6,
        dmri.techo .* 1.0e-3,
        dmri.tdelta .* 1.0e-3,
        dmri.tsmalldel .* 1.0e-3,
    )
    protocol.bvec .= dmri.nifti.bvec
    protocol.nmeas .= dmri.nmeas
    protocol.lmeas .= dmri.lmeas
    return protocol
end

# make protocol from btable file
function Protocol(infile::String)
    if !isfile(infile)
        error("Could not find btable file")
    end

    # find input file extention
    idot = findlast(isequal('.'), infile)
    ext = lowercase(infile[(idot + 1):end])

    if ext != "btable"
        error("Input is not a btable")
    end

    # read file and make protocol
    tab = readdlm(infile)
    return Protocol(
        tab[:, 1], tab[:, 2], tab[:, 3], tab[:, 4], Int.(tab[:, 5]), Int.(tab[:, 6])
    )
end

################################ spherical_fit module ###############################################
"""
    spherical_fit(
        image_file::String, 
        mask_file::String,
        Lmax::Int,
        tissue_type::String,
        save_dir::String, 
        acq_files::String...
    )

    spherical_fit(
        image_file::String, 
        mask_file::String,
        sigma_file::String,
        Lmax::Int,
        tissue_type::String,
        save_dir::String, 
        acq_files::String...
    )

This function estimates the Spherical Harmonic (SH) coefficents of the input DW images using linear least squares 
    and extract the rotional invariants at an order up to Lmax for later microstructure model fitting. 

Arguments:
- image_file: full file path of the input DWI volumes
- mask_file: full file path of the brain mask 
- sigma_file: full file path of the noise map; if provided, simple correction will be applied to the measurements to reduce Rician bias before the LLS fit of SH coefficents. 
- Lmax: The maximum order to extract rotational invariants; set Lmax = 0 for models using only the spherical mean signals, e.g. SANDI,
        and set Lmax = 2 or 4 for models of the white matter using higher order rotional invariants, e.g. SMI.
- tissue_type: "in_vivo" or "ex_vivo"; this is used along with the number of gradient directions to help decide the maximum SH order of each b-shell for the SH fitting.
- save_dir: the path where outputs will be saved
- acq_files:: full file path to the acquistion files assoiated with the DWI volumes. Provide at least the .bvals/.bvecs files for standard microstructure model fitting.
              Accepted file extensions are .bvals/.bvecs/.techo/.tdelta/.tsmalldel for b-values, gradient directions, echo times, diffusion gradient seperation and duration times. 
               
              The format of a .tdelta/.tsmalldel/.techo file is similar to a .bvals file (a vector with the length equal to the number of DWI volumes). Unit in the .tdelta/.tsmalldel/.techo file is ms. 
                .tdelta/.tsmalldel files are needed for any models that estimate size, e.g. axon diameter, soma radius.
                .techo is needed if your data is collected with multiple echo-time and you want to do combined diffusion-relaxometry modelling. 
               
Returns:
- Sl_norm: a 4D Array containing the rotional invariants Sl normalized by the b=0 invariant with minimal TE
- snr_b0: a 3D Array containing the SNR map estimated from the b=0 measurements with minimal TE
- protocol: A Protocol type object containing the relevant parameters associated with each volume in Sl_norm
- mask: An MRI type object with nifti header information and Array of brain mask

Besides returning necessary variables for model fitting, files are saved under `save_dir` for reuse (naming examples with Lmax=2):
Sl_lmax2_norm.nii.gz: the normalized measurements that are used for model fitting
snr_b0.nii.gz       : the SNR map that can be used in setting up neural network estimators trained with realistic noise distribution
Sl_lmax2.btable     : the protocol 

Slm.nii.gz          : the SH coefficents of all the shells
Sl_lmax2.nii.gz     : the rotational invariants before normalization
sigma_est.nii.gz    : the sigma maps estimated from SH fit for each b-shell
"""
function spherical_fit(
    infile_image::String,
    mask_file::String,
    sigma_file::String,
    Lmax::Int,
    tissuetype::String,
    savedir::String,
    infiles::String...,
)
    mri = mri_read(infile_image)
    dmri = dmri_read_times(mri, infiles)
    brainmask = mri_read(mask_file)

    # correct for rician bias based on given sigma map when estimating the SH coefficents
    volsize = size(mri.vol)
    sigma = mri_read(sigma_file)
    for x in 1:volsize[1]
        for y in 1:volsize[2]
            for z in 1:volsize[3]
                brainmask.vol[x, y, z] == 0 && continue
                dmri.nifti.vol[x, y, z, :] .= correct_rician_bias(
                    dmri.nifti.vol[x, y, z, :], sigma.vol[x, y, z]
                )
            end
        end
    end

    return spherical_fit(dmri, brainmask, Lmax, tissuetype, savedir)
end

function spherical_fit(
    infile_image::String,
    mask_file::String,
    Lmax::Int,
    tissuetype::String,
    savedir::String,
    infiles::String...,
)
    mri = mri_read(infile_image)
    dmri = dmri_read_times(mri, infiles)
    brainmask = mri_read(mask_file)

    return spherical_fit(dmri, brainmask, Lmax, tissuetype, savedir)
end

function spherical_fit(
    dmri::dMRI, brainmask::MRI, Lmax::Int, tissuetype::String, savedir::String
)
    Sl, prot, sigma, Slm = spherical_fit(dmri, brainmask, Lmax, tissuetype)

    # snr estimate on b=0 images with minimal TE
    mri = MRI(brainmask, 1, Float32)
    snr_b0 = Sl[:, :, :, 1] ./ sigma[:, :, :, 1]
    mri.vol = snr_b0
    mri_write(mri, joinpath(savedir, "snr_b0.nii.gz"))

    mri = MRI(brainmask, size(Sl, 4), Float32)
    mri.vol = Sl
    mri_write(mri, joinpath(savedir, "Sl_lmax" * string(Lmax) * ".nii.gz"))

    Sl_norm = Sl ./ repeat(Sl[:, :, :, 1], 1, 1, 1, size(Sl, 4))
    mri.vol = Sl_norm
    mri_write(mri, joinpath(savedir, "Sl_lmax" * string(Lmax) * "_norm.nii.gz"))

    mri = MRI(brainmask, size(Slm, 4), Float32)
    mri.vol = Slm
    mri_write(mri, joinpath(savedir, "Slm.nii.gz"))

    mri = MRI(brainmask, size(sigma, 4), Float32)
    mri.vol = sigma
    mri_write(mri, joinpath(savedir, "sigma_est.nii.gz"))

    btable = hcat(
        prot.bval,
        prot.techo,
        prot.tdelta,
        prot.tsmalldel,
        prot.nmeas,
        prot.lmeas,
        prot.gvec,
    )
    writedlm(joinpath(savedir, "Sl_lmax" * string(Lmax) * ".btable"), btable, ' ')

    return Sl_norm, snr_b0, prot, brainmask
end

function spherical_fit(dmri::dMRI, brainmask::MRI, Lmax::Int, tissuetype::String)

    # select unique combinations of bval, techo, tdelta, tsmalldel 
    sets = [dmri.nifti.bval dmri.techo dmri.tdelta dmri.tsmalldel]
    combinations = unique(sets; dims=1)

    # sortting to help check signals when bval/techo are not in assending order
    ind = sortperm(combinations[:, 1])
    combinations = combinations[ind, :]
    ind = sortperm(combinations[:, 2])
    combinations = combinations[ind, :]

    volsize = size(dmri.nifti.vol)
    nsets = size(combinations, 1)
    sigma = zeros(volsize[1:3]..., nsets)
    Slm = []
    Sl = []
    prot = []

    for i in 1:nsets

        # get indices for each combinaitons
        index = []
        for j in 1:volsize[4]
            if sets[j, :] == combinations[i, :]
                append!(index, j)
            end
        end

        print("SH fit on unique shell (b/TE/delta/smalldel):")
        println(combinations[i, :])
        #println(index)

        # extract measurements for SH fit
        meas = @view dmri.nifti.vol[:, :, :, index]
        dirs = @view dmri.nifti.bvec[index, :]

        # get the order of SH to fit
        L = get_sh_order(combinations[i, 1], length(index), tissuetype)
        Y = build_sh_basis(dirs, L)

        # estimate SH coeffs per shell; estimate an averaged sigma per shell
        n_ml = Int((L+1)*(L+2)/2)
        Slm_shell = zeros(volsize[1:3]..., n_ml)

        for x in 1:volsize[1]
            for y in 1:volsize[2]
                for z in 1:volsize[3]
                    brainmask.vol[x, y, z] == 0 && continue
                    Slm_shell[x, y, z, :] .= Y \ meas[x, y, z, :]
                    # residual sum of squares
                    sigma[x, y, z, i] = sum(
                        abs2, Y * Slm_shell[x, y, z, :] - meas[x, y, z, :]
                    )
                    # estimate of sigma
                    sigma[x, y, z, i] = sqrt(sigma[x, y, z, i] ./ (length(index) - n_ml))
                end
            end
        end

        # compute rotational invariants; normalized sqare root of spherical variance 
        L>Lmax ? L = Lmax : L = L
        n_l = Int(L/2+1)

        Sl_shell = zeros(volsize[1:3]..., n_l)
        offset = 0
        for (ind, l) in enumerate(0:2:L)
            n_coeffs_l = 2l + 1
            coeffs_l = @view Slm_shell[:, :, :, (offset + 1):(offset + n_coeffs_l)]

            norm_factor = sqrt(4π * n_coeffs_l)
            Sl_shell[:, :, :, ind] .= sqrt.(sum(abs2, coeffs_l; dims=4)) ./ norm_factor

            offset += n_coeffs_l
            push!(prot, [combinations[i, :]..., length(index), l])
        end

        push!(Sl, Sl_shell)
        push!(Slm, Slm_shell)
    end

    Sl = cat(Sl...; dims=4)
    Slm = cat(Slm...; dims=4)

    prot = hcat(prot...)
    prot = Protocol(
        prot[1, :]*1.0e6,
        prot[2, :]*1.0e-3,
        prot[3, :]*1.0e-3,
        prot[4, :]*1.0e-3,
        Int.(prot[5, :]),
        Int.(prot[6, :]),
    )

    return Sl, prot, sigma, Slm
end

"""
The real SH basis functions are used to be compatible with the SMI matlab toolbox and MRtrix3;
Definition: https://cs.dartmouth.edu/~wjarosz/publications/dissertation/appendixB.pdf
"""
function real_spherical_harmonic(l::Int, m::Int, θ::AbstractFloat, ϕ::AbstractFloat)
    abs_m = abs(m)
    norm = sqrt((2l + 1) / (4π) * factorial(l - abs_m) / factorial(l + abs_m))
    plm = assoc_legendre(l, abs_m, cos(θ))
    if m > 0
        return √2 * norm * plm * cos(m * ϕ)
    elseif m < 0
        return √2 * norm * plm * sin(-m * ϕ)
    else
        return norm * plm
    end
end

"""
Custom the unnormalized associated Legendre polynomial P_l^m(x).

This function gives equal results as AssociatedLegendrePolynomials.Plm when using Condon-Shortley phase.
Condon-Shortley phase is not used by default to be compatible with other tools, e.g. MRtrix3, SMI matlab toolbox.

Arguments:
- l::Int : order (l ≥ 0)
- m::Int : degree/phase factor (|m| ≤ l)
- x::Float64 : input, typically cos(θ) ∈ [-1, 1]
- condon_shortley::Bool : whether to include the Condon-Shortley phase (-1)^m (default: false)

Returns:
- P_l^m(x) as Float64
"""
function assoc_legendre(l::Int, m::Int, x::Float64; condon_shortley::Bool=false)
    m_abs = abs(m)
    if m_abs > l
        error("|m| cannot exceed order l")
    end

    # Compute P_m^m
    pmm = 1.0
    if m_abs > 0
        somx2 = sqrt(1.0 - x^2)
        for i in 1:m_abs
            factor = (2i - 1) * somx2
            pmm *= condon_shortley ? -factor : factor
        end
    end

    if l == m_abs
        return pmm
    end

    # Compute P_{m+1}^m
    pmmp1 = x * (2m_abs + 1) * pmm
    if l == m_abs + 1
        return pmmp1
    end

    # Recurrence for P_l^m
    for ll in (m_abs + 2):l
        pll = ((2ll - 1) * x * pmmp1 - (ll + m_abs - 1) * pmm) / (ll - m_abs)
        pmm, pmmp1 = pmmp1, pll
    end

    return pmmp1
end

"""
Convert 3D unit vector to spherical coordinates (θ, φ)
"""
function cart2sph(x, y, z)
    θ = acos(clamp(z, -1.0, 1.0))
    ϕ = atan(y, x)
    return θ, ϕ < 0 ? ϕ + 2π : ϕ
end

"""
Return SH basis matrix (only even ℓ) evaluated at a list of directions with Lmax
Size(Y) = [N, (Lmax+1)*(Lmax+2)/2]
"""
function build_sh_basis(dirs::AbstractArray{<:AbstractFloat,2}, Lmax::Int)
    N = size(dirs, 1)
    n_coeffs = sum(2l + 1 for l in 0:2:Lmax)
    Y = zeros(N, n_coeffs)

    col = 1
    for l in 0:2:Lmax
        for m in (-l):l
            for i in 1:N
                θ, ϕ = cart2sph(dirs[i, 1], dirs[i, 2], dirs[i, 3])
                Y[i, col] = real_spherical_harmonic(l, m, θ, ϕ)
            end
            col += 1
        end
    end
    return Y
end

"""
Simple correction to reduce Rician bias in DWI measurements given the noise level.

Reference
Gudbjartsson, H. and Patz, S., 1995. The Rician distribution of noisy MRI data. Magnetic resonance in medicine, 34(6), pp.910-914.
"""
function correct_rician_bias(S::Vector{<:AbstractFloat}, σ::Float64)
    return [s > σ ? sqrt(s^2 - σ^2) : 0.0 for s in S]
end

"""
This function decides the lmax order when estimating the SH coefficents from measurements of a b-shell
    it depends on both the number of measurements and the b-values (adjusted for tissue type).
"""
function get_sh_order(bval, nmeas::Int, tissue::String="in_vivo")

    # feasible lmax
    if nmeas > 45
        lmax_n = 8
    elseif nmeas > 28
        lmax_n = 6
    elseif nmeas > 15
        lmax_n = 4
    elseif nmeas > 6
        lmax_n = 2
    else
        lmax_n = 0
    end

    if tissue == "ex_vivo"
        b = bval ./ 4.0
    else
        b = bval
    end

    # consider the b-value
    if b <= 100
        lmax_b = 0
    elseif b <= 1200
        lmax_b = 2
    elseif b <= 2500
        lmax_b = 4
    elseif b <= 7000
        lmax_b = 6
    else
        lmax_b = 8
    end

    return min(lmax_n, lmax_b)
end

######################## previous spherical_mean module #########################
"""
mutating dmri structure after direction averaging 
"""
function spherical_mean!(dmri::dMRI)
    if dmri.smt == true
        error("The input contains already spherical mean signals")
    end

    # select unique combinations of bval, techo, tdelta, tsmalldel 
    sets = [dmri.nifti.bval dmri.techo dmri.tdelta dmri.tsmalldel]
    combinations = unique(sets; dims=1)

    # sortting to help check signals when bval/techo are not in assending order
    ind = sortperm(combinations[:, 1])
    combinations = combinations[ind, :]
    ind = sortperm(combinations[:, 2])
    combinations = combinations[ind, :]

    # initialize new volume
    nsets = size(combinations, 1)
    volsize = size(dmri.nifti.vol)
    vol = Array{AbstractFloat}(undef, volsize[1:3]..., nsets)

    # save estimate of SNR map per TE
    b0ind = iszero.(combinations[:, 1])
    snr = Array{AbstractFloat}(undef, volsize[1:3]..., sum(b0ind))
    start = 1

    # direction average persets
    nmeas = Int.(ones(nsets))
    for i in 1:nsets
        index = []
        for j in 1:volsize[4]
            if sets[j, :] == combinations[i, :]
                append!(index, j)
            end
        end
        vol[:, :, :, i] .= mean(dmri.nifti.vol[:, :, :, index]; dims=4)
        nmeas[i] = length(index)
        if b0ind[i] == 1
            snr[:, :, :, start] .=
                mean(dmri.nifti.vol[:, :, :, index]; dims=4) ./
                std(dmri.nifti.vol[:, :, :, index]; dims=4)
            start = start + 1
        end
    end

    # update related fields
    dmri.nifti.bval = combinations[:, 1]
    dmri.techo = combinations[:, 2]
    dmri.tdelta = combinations[:, 3]
    dmri.tsmalldel = combinations[:, 4]
    dmri.nifti.vol = vol
    dmri.nifti.bvec = Matrix{Float64}(undef, nsets, 3)
    dmri.nifti.nframes = nsets
    dmri.smt = 1
    dmri.nmeas = nmeas
    dmri.lmeas = Int.(zeros(nsets))
    return snr
end

"""
normalize signals with minimal TE and b=0 volume; 
save the first volume (all 1) for the associated acquistion parameters in the normalizing volume
"""
function normalize_smt!(dmri::dMRI)
    if dmri.nifti.bval[1] != 0
        error("First volume is not from b=0")
    end

    nvol = length(dmri.nifti.bval)
    vol_b0 = dmri.nifti.vol[:, :, :, 1]

    for i in 1:nvol
        dmri.nifti.vol[:, :, :, i] = dmri.nifti.vol[:, :, :, i] ./ vol_b0
    end

    return nothing
end

"""
    spherical_mean(
        image_file::String, 
        save::Bool=true, 
        acq_files::String...
    )

Perform direction average on input DWI images `image_file` and return an MRI object with normalized spherical mean signal and associated imaging protocol.
`image_file` is the full path of the DWI image file; `save` indicates whether to save the smt and normalized smt image volumes and protocol. If saving the files, nifti and text file (.btable) will be saved in the same path as the input data.
Finall, variable number of `acq_files` are text files that tell you acquistion parameters of each DWI in the `image_file`. 
Accepted file extensions are .bvals/.bvecs/.techo/.tdelta/.tsmalldel for b-values, gradient directions, echo times, diffusion gradient seperation and duration times.

Besides .bvals/.bvecs for conventional modelling, .tdelta/.tsmalldel files are needed for any models that estimate size, e.g. axon diameter, soma radius.
.techo is needed if your data is collected with multiple echo-time and you want to do combined-diffusion relaxometry modelling. 
The format of a .tdelta/.tsmalldel/.techo file is similar to a .bvals file (a vector with the length equal to the number of DWI volumes). 
Unit in the .tdelta/.tsmalldel/.techo file is ms. 
"""
function spherical_mean(infile_image::String, save::Bool=true, infiles::String...)
    mri = mri_read(infile_image)
    dmri = dmri_read_times(mri, infiles)

    snr = spherical_mean!(dmri)
    if save
        datapath = infile_image[1:findlast(isequal('/'), infile_image)]
        dmri_write(dmri, datapath, "diravg.nii.gz")

        new = MRI(mri)
        new.vol = snr
        new.nframes = size(snr, 4)
        empty(new.bval)
        new.bvec = Matrix{Float32}(undef, 0, 0)
        mri_write(new, joinpath(datapath, "snr_b0.nii.gz"))
    end

    # default to normalize signals
    normalize_smt!(dmri)
    if save
        datapath = infile_image[1:findlast(isequal('/'), infile_image)]
        dmri_write(dmri, datapath, "diravg_norm.nii.gz")
    end

    prot = Protocol(dmri)
    return dmri.nifti, prot, snr
end

"""
    dmri_write(dmri::dMRI, datapath::String, filename::String)

Write the nifti volume in a dMRI object to nifti file and associated protocol as b-table text files in the given `datapath` and `filename`.
"""
function dmri_write(dmri::dMRI, datapath::String, outfile::String)
    mri_write(dmri.nifti, joinpath(datapath, outfile))

    # find input file name
    idot = findfirst(isequal('.'), outfile)
    name = lowercase(outfile[1:(idot - 1)])

    prot = Protocol(dmri)
    btable = hcat(
        prot.bval,
        prot.techo,
        prot.tdelta,
        prot.tsmalldel,
        prot.nmeas,
        prot.lmeas,
        prot.gvec,
        prot.bvec,
    )
    writedlm(joinpath(datapath, name * ".btable"), btable, ' ')
    return nothing
end
