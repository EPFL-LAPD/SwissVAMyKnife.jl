module SwissVAMyKnife

using WaveOpticsPropagation
using RadonKA
using DiffImageRotation
using IndexFunArrays
using ChainRulesCore
using Optim
using Zygote
using CUDA
using Plots
using FFTW
using HDF5
using FileIO
using ImageShow

include("ray_optics.jl")
include("wave_optics.jl")
include("optimization.jl")


export plot_intensity 
export save_patterns



function plot_intensity(target, object_printed, thresholds)
    # :stephist vs :barhist
    plot_font = "Computer Modern"
    default(fontfamily=plot_font,
	    linewidth=2, framestyle=:box, label=nothing, grid=false)
	plot(object_printed[target .== 0], seriestype=:barhist, bins=(0.0:0.01:1), xlim=(0.0, 1.0), label="dose distribution void", ylabel="voxel count", xlabel="normalized intensity",  ylim=(10, 10000000),  linewidth=1, legend=:topleft, yscale=:log10, size=(500, 350))
	plot!(object_printed[target .== 1], seriestype=:barhist, bins=(0.0:0.01:1), xlim=(0.0, 1.0), label="dose distribution object", ylabel="voxel count", xlabel="normalized intensity",  ylim=(10, 10000000),  linewidth=1, legend=:topleft, yscale=:log10, size=(500, 350))
	plot!([thresholds[1], thresholds[1]], [1, 10000_000], label="lower threshold", linewidth=3)
	plot!([thresholds[2], thresholds[2]], [1, 10000_000], label="upper threshold", linewidth=3)
	#plot!([chosen_threshold, chosen_threshold], [1, 30000000], label="chosen threshold", linewidth=3)
end

"""
    save_patterns(fpath, patterns, printed, angles, target; overwrite=true)

Save all arguments into the path `fpath`.
`fpath` should be a path.

`overwrite=false` per default and does not overwrite existing files!

"""
function save_patterns(fpath, patterns, printed, angles, target; overwrite=true)
    @assert size(angles, 1) == size(patterns, 2) "Size mismatch between angles and patterns"
    #@assert size(patterns, 1) == size(target, 1) "Size mismatch between target and patterns"
    #@assert size(patterns, 3) == size(target, 3) "Size mismatch between target and patterns"

    # might be on GPU, move to CPU
    patterns = Array(patterns)
    patterns ./ maximum(patterns)
    printed = Array(printed)
    printed ./ maximum(printed)
    angles = Array(angles)
    target = Array(target)


    # check if path exists, otherwise create
    isdir(fpath) || mkpath(fpath)
    file = joinpath(fpath, "data.hdf5")
    if isfile(file) && overwrite==false
        throw(ArgumentError("HDF5 file exists already"))
    end

    fid = h5open(file, "w")
    fid["patterns"] = patterns
    fid["printed"] = printed
    fid["angle"] = angles
    close(fid)
    
    # create folder for png files
    fpath_images = joinpath(fpath, "patterns_png")
    isdir(fpath) || mkpath(fpath_images) 


    # convert to proper Grayscale image
    patterns = simshow(patterns)
    for i in 1:size(angles, 1)
        number = lpad(string(i), 5, "0")
        fname = joinpath(fpath_images, number * ".png")
        if isfile(fname) && overwrite==false
            throw(ArgumentError("png file exists already"))
        end
        save(fname, view(patterns, :, i, :))
    end

    return 0
end

end # module SwissVAMyKnife
