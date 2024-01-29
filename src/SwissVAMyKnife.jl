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
using Parameters
using NNlib

include("ray_optics.jl")
include("wave_optics.jl")
include("optimization.jl")


export plot_histogram
export save_patterns



function plot_histogram(target, object_printed, thresholds; yscale=:log10)
    # :stephist vs :barhist
    
    plot_font = "Computer Modern"
    default(fontfamily=plot_font,
	    linewidth=2, framestyle=:box, label=nothing, grid=false)
    m = maximum(object_printed)
	plot(object_printed[target .== 0], seriestype=:stephist, bins=(0.0:0.01:m), xlim=(0.0, m), label="dose distribution void", ylabel="voxel count", xlabel="normalized intensity",  ylim=(1, 10000000),  linewidth=1, legend=:topleft, yscale=yscale, size=(500, 350))
	plot!(object_printed[target .== 1], seriestype=:stephist, bins=(0.0:0.01:m), xlim=(0.0, m), label="dose distribution object", ylabel="voxel count", xlabel="normalized intensity",  ylim=(1, 10000000),  linewidth=1, legend=:topleft, yscale=yscale, size=(500, 350))
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
    patterns = simshow(patterns, cmap=:turbo)
    for i in 1:size(angles, 1)
        number = lpad(string(i), 5, "0")
        fname = joinpath(fpath_images, number * ".png")
        if isfile(fname) && overwrite==false
            throw(ArgumentError("png file exists already"))
        end
        save(fname, view(patterns, :, i, :))
    end
    


    fpath_images = joinpath(fpath, "printed_png")
    isdir(fpath) || mkpath(fpath_images) 
    # convert to proper Grayscale image
    printed = simshow(printed, cmap=:turbo)
    for i in 1:size(printed, 1)
        number = lpad(string(i), 5, "0")
        fname = joinpath(fpath_images, number * ".png")
        if isfile(fname) && overwrite==false
            throw(ArgumentError("png file exists already"))
        end
        save(fname, view(printed, :, :, i))
    end



    return 0
end

end # module SwissVAMyKnife
