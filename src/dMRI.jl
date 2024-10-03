# I/O functions for images and protocols
# Include functions to perform direction average

using Fibers, DelimitedFiles, Statistics, StaticArrays

export dMRI,
    Protocol,
    spherical_mean,
    spherical_mean!,
    normalize_smt!,
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
    smt::Bool)

Return a dMRI Type object with MRI object `nifti`, and additional volume-wise 
experimental settings `tdelta`, `tsmalldel`, `techo`, and `smt` for identifing smt signals. 
"""
mutable struct dMRI
    nifti::MRI
    tdelta::Vector{Float64}
    tsmalldel::Vector{Float64}
    techo::Vector{Float64}
    smt::Bool
end

"""
initialize dMRI structure from MRI or Array
"""
dMRI(mri::MRI) = dMRI(
    mri,
    Vector{Float64}(zeros(mri.nframes)),
    Vector{Float64}(zeros(mri.nframes)),
    Vector{Float64}(zeros(mri.nframes)),
    false,
)

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

    # round b table; .element-wise openration
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
        if ext == "techo"
            dmri.techo = vec(tab)
        elseif ext == "tdelta"
            dmri.tdelta = vec(tab)
        elseif ext == "tsmalldel"
            dmri.tsmalldel = vec(tab)
        elseif ext == "bvals" || ext == "bval"
            dmri.nifti.bval = vec(tab)
        elseif ext == "bvecs" || ext == "bvec"
            dmri.nifti.bvec = tab
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
    ind = sortperm(combinations[:,1])
    combinations = combinations[ind,:]
    ind = sortperm(combinations[:,2])
    combinations = combinations[ind,:]

    # initialize new volume
    nsets = size(combinations, 1)
    volsize = size(dmri.nifti.vol)
    vol = Array{AbstractFloat}(undef, volsize[1:3]..., nsets)

    # save estimate of SNR map per TE
    b0ind = iszero.(combinations[:,1])
    snr = Array{AbstractFloat}(undef, volsize[1:3]..., sum(b0ind))
    start = 1

    # direction average persets
    for i in 1:nsets
        index = []
        for j in 1:volsize[4]
            if sets[j, :] == combinations[i, :]
                append!(index, j)
            end
        end
        vol[:, :, :, i] .= mean(dmri.nifti.vol[:, :, :, index]; dims=4)
        if b0ind[i] == 1
            snr[:,:,:,start] .= 
                mean(dmri.nifti.vol[:, :, :, index]; dims=4)./
                std(dmri.nifti.vol[:, :, :, index]; dims=4)
            start = start +1
        end
    end

    # update related fields
    dmri.nifti.bval = combinations[:, 1]
    dmri.techo = combinations[:, 2]
    dmri.tdelta = combinations[:, 3]
    dmri.tsmalldel = combinations[:, 4]
    dmri.nifti.vol = vol
    dmri.nifti.bvec = Matrix{Float32}(undef,0,0)
    dmri.nifti.nframes = nsets
    dmri.smt = 1
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
Apply universal scaling to smt signals
"""
function scale_smt!(dmri::dMRI)

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
function spherical_mean(
    infile_image::String, save::Bool=true, infiles::String...
)
    mri = mri_read(infile_image)
    dmri = dmri_read_times(mri, infiles)
    
    snr = spherical_mean!(dmri)
    if save
        datapath = infile_image[1:findlast(isequal('/'), infile_image)]
        dmri_write(dmri, datapath, "diravg.nii.gz")
        
        new = MRI(mri)
        new.vol = snr
        new.nframes = size(snr,4)
        empty(new.bval)
        new.bvec = Matrix{Float32}(undef,0,0)
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
    btable = hcat(prot.bval, prot.techo, prot.tdelta, prot.tsmalldel, prot.gvec)
    writedlm(joinpath(datapath, name * ".btable"), btable, ' ')
    return nothing
end

"""
    Protocol(
    bval::Vector{Float64}
    techo::Vector{Float64}
    tdelta::Vector{Float64}
    tsmalldel::Vector{Float64}
    gvec::Vector{Float64}
    )

Return a Protocol Type object to hold parameters in acquisition protocol relavent for modelling 
including b-values, tcho times, diffusion gradient seperation, duration and strengh. 
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
Calculate `gvec` and return a Ptotocol Type object from provided parameters.
"""
struct Protocol
    bval::Vector{Float64}
    techo::Vector{Float64}
    tdelta::Vector{Float64}
    tsmalldel::Vector{Float64}
    gvec::Vector{Float64}
    #bvec::AbstractMatrix{Float64}
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
    return Protocol(bval, techo, tdelta, tsmalldel, gvec)
end

"""
Make protocol from a dMRI object
"""
function Protocol(dmri::dMRI)
    return Protocol(
        dmri.nifti.bval .* 1.0e6,
        dmri.techo .* 1.0e-3,
        dmri.tdelta .* 1.0e-3,
        dmri.tsmalldel .* 1.0e-3,
    )
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
    return Protocol(tab[:, 1], tab[:, 2], tab[:, 3], tab[:, 4], tab[:, 5])
end
