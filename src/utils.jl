export printing_errors, plot_intensity_histogram, save_patterns, calculate_IoU
export load_example_target, load_image_stack

"""
    interpolate_patterns(patterns, N_angles)

"""
function interpolate_patterns(patterns; N_angles, Nx, Ny)


end

"""
    load_example_target(name)

Load the example target `name` from the artifacts as voxelized array.

Possible targets are:
- `"3DBenchy_180"`
- `"3DBenchy_550"`
"""
function load_example_target(name)
    artifacts_toml = abspath(joinpath(@__DIR__, "..", "Artifacts.toml"))
    ensure_artifact_installed(name, artifacts_toml)
    image_dir = artifact_path(artifact_hash(name, artifacts_toml))

    if name == "3DBenchy_180"
        return JLD2.load(joinpath(Pkg.Artifacts.@artifact_str(name), "benchy_100.jld2"), "target")
    elseif name == "3DBenchy_550"
        return JLD2.load(joinpath(Pkg.Artifacts.@artifact_str(name), "3D_benchy_550.jld2"), "target")
    else
        throw(ArgumentError("Unknown example target $name"))
    end
end


"""
    load_image_stack(sz, path, prefix="boat_", pad=2)

Load the image stack from the path `path` and resize it to `sz`.
The files should be binary images with 1s and 0s.

Example:
This will load all images from path which have file names `boat_00.png`, `boat_01.png`, ...
The output array will have size (128, 128, 100). `sz` shouldm be larger than the size of the images.

```julia
julia> load_image_stack((128, 128, 100), "path/to/images", prefix="boat_", pad=2)

```

"""
function load_image_stack(sz, path; prefix="boat", pad=2)
    one_img = load(joinpath(path, string(prefix, string(0, pad=pad) ,".png")))
    if any(size(one_img) .> sz[1:2]) || length(readdir(path)) > sz[3]
        throw(ArgumentError("Size of object is larger than provided $(sz)"))
    end

	target = zeros(Float32, sz)

    Threads.@threads for i in 0:length(readdir(path)) - 1
		target[:, :, i+1] .= select_region(Gray.(load(joinpath(path, string(prefix, string(i, pad=pad) ,".png")))) .> 0, new_size=(sz))
	end

	return target
end


"""
    calculate_IoU(target, printed)

Calculate the Intersection over Union (IoU) of the `printed` object compared with the `target` object.

"""
function calculate_IoU(target, printed)
    intersection = sum((target .> 0) .& (printed .> 0))
    union = sum((target .> 0) .| (printed .> 0))
    return intersection / union
end



"""
    printing_errors(target, printed, thresholds)

Calculate the printing errors of the `printed` object compare with the `target` object.
`thresholds` should indicate the used threshold values for polymerization.
"""
function printing_errors(target, printed, thresholds)
    isobject = target .≈ 1
    notobject = target .≈ 0
	mid_thresh = (thresholds[2] + thresholds[1]) / 2
	W_not = sum(printed[notobject] .> thresholds[1])
	W_not_is = sum(printed[notobject] .> mid_thresh)
	W_is = sum(printed[isobject] .< thresholds[2])
	
	N_not = sum(notobject)
	N_is = sum(isobject)
	
	voxels_object_wrong_printed = sum(abs.((printed .> mid_thresh)[isobject] .- target[isobject]))
	voxels_void_wrong_printed = sum(abs.((printed .> mid_thresh)[notobject] .- target[notobject]))

	#voxels_object_wrong_printed / N_is, W_not_is / N_not, W_not / N_not, W_is / N_is

	@info "Object pixels not printed $(round(voxels_object_wrong_printed / N_is * 100, digits=4))%"
	@info "Void pixels falsely printed $(round(voxels_void_wrong_printed / N_not * 100, digits=4))%"
end



"""
    plot_intensity_histogram(target, object_printed, thresholds)

Plot an intensity histogram of the `printed_object`.
`target` is the original binary target object.
`thresholds` should indicate the threshold values for polymerization.
"""
function plot_intensity_histogram(target, object_printed, thresholds; yscale=:log10, xlim=(0.0, 1.1))
    # :stephist vs :barhist
    target = Array(target)
    object_printed = Array(object_printed)
    plot_font = "Computer Modern"
    default(fontfamily=plot_font,
	    linewidth=2, framestyle=:box, label=nothing, grid=false)
    m = maximum(object_printed)
	plot(object_printed[target .== 0], seriestype=:stephist, bins=(0.0:0.01:m), xlim=xlim, label="dose distribution void", ylabel="voxel count", xlabel="normalized intensity",  ylim=(1, 10000000),  linewidth=1, legend=:topleft, yscale=yscale, size=(400, 300))
	plot!(object_printed[target .== 1], seriestype=:stephist, bins=(0.0:0.01:m),  label="dose distribution object", ylabel="voxel count", xlabel="normalized intensity",  ylim=(1, 10000000),  linewidth=1, legend=:topleft, yscale=yscale, size=(400, 300))
	plot!([thresholds[1], thresholds[1]], [1, 10000_000], label="lower threshold", linewidth=1)
	plot!([thresholds[2], thresholds[2]], [1, 10000_000], label="upper threshold", linewidth=1)
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
    patterns = simshow(patterns, cmap=:gray)
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
    for i in 1:size(printed, 3)
        number = lpad(string(i), 5, "0")
        fname = joinpath(fpath_images, number * ".png")
        if isfile(fname) && overwrite==false
            throw(ArgumentError("png file exists already"))
        end
        save(fname, view(printed, :, :, i))
    end



    return 0
end


const start_time = Ref(zeros(1))
const last_time = Ref(zeros(1))
function log_time(x)
    if x[end].iteration == 0
        start_time[] .= time()
        last_time[] .= time()
    end
    @info "Iteration:\t$(x[end].iteration),\t total time:\t$(round(time()-start_time[][1], digits=2))s,\ttime since last iteration:\t$(round(time()-last_time[][1], digits=2))s, loss value: $(round(x[end].value, digits=4))"

    last_time[] .= time()
    return false
end
