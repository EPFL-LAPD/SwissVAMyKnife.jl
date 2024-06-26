export printing_errors, plot_intensity_histogram, save_patterns, calculate_IoU
export load_example_target, load_image_stack

"""
    interpolate_patterns(patterns, N_angles)

"""
function interpolate_patterns(patterns; N_angles, Nx, Ny)


end


"""
    load_images_into_array(path)

Load all images from the path `path` into an array.
The images are loaded according to the file names and sorted by the file names.
"""
function load_images_into_array(path)
    fnames = sort(readdir(path))
    sz = size(Gray.(load(joinpath(path, fnames[1]))))

    images = zeros(Float32, sz[1], length(fnames), sz[2])

    Threads.@threads for i in 1:length(fnames)
        images[:,  i, :] .= Gray.(load(joinpath(path, fnames[i])))
    end

    return images
end


"""
    correct_rotation_axis_wobbling(patterns, angles, f_wobble=φ -> 0)

This function corrects the wobbling of the rotation axis along the first dimension of the `patterns`.
The shifting is integer pixel shifts
The function `f_wobble` should return the wobbling distance from the center of the object in first dimension.
Units of `angles` and `f_wobble` should be the same.
"""
function correct_rotation_axis_wobbling(patterns, angles, f_wobble=ϕ -> 0)
    @assert size(patterns, 2) == size(angles, 1) "Size mismatch between angles and patterns"
    patterns_out = copy(patterns)
    Threads.@threads for i in 1:size(patterns, 2)
        φ = angles[i]
        patterns_out[:, i, :] .= @views circshift(patterns[:, i, :], (round(Int, f_wobble(φ)), 0))
    end
    
    return patterns_out
end



"""
    load_example_target(name)

Load the example target `name` from the artifacts as voxelized array.

Those two targets are equal size in all dimensions:
- `"3DBenchy_180"`
- `"3DBenchy_550"`

Those two are the original 3D Benchy objects with correct aspect ratio:
- `"3DBenchy_120_aspect_ratio"`
- `"3DBenchy_450_aspect_ratio"`
"""
function load_example_target(name)
    artifacts_toml = abspath(joinpath(@__DIR__, "..", "Artifacts.toml"))
    ensure_artifact_installed(name, artifacts_toml)
    image_dir = artifact_path(artifact_hash(name, artifacts_toml))

    if name == "3DBenchy_180"
        return Float32.(JLD2.load(joinpath(Pkg.Artifacts.@artifact_str(name), "benchy_100.jld2"), "target"))
    elseif name == "3DBenchy_550"
        return Float32.(JLD2.load(joinpath(Pkg.Artifacts.@artifact_str(name), "3D_benchy_550.jld2"), "target"))
    elseif name == "3DBenchy_120_aspect_ratio"
        return Float32.(JLD2.load(joinpath(Pkg.Artifacts.@artifact_str(name), "3D_benchy_120_aspect_ratio.jld2"), "target"))
    elseif name ==  "3DBenchy_450_aspect_ratio"
        return Float32.(JLD2.load(joinpath(Pkg.Artifacts.@artifact_str(name), "3D_benchy_450_aspect_ratio.jld2"), "target"))
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
function load_image_stack(sz, path; prefix="boat_", pad=2)
    one_img = load(joinpath(path, string(prefix, string(0, pad=pad) ,".png")))
    Nz = length(readdir(path))
    sz_object = (size(one_img, 1), size(one_img, 2), Nz)
    if any(size(one_img) .> sz[1:2]) || Nz > sz[3]
        throw(ArgumentError("Size of object $(sz_object) is larger than provided $(sz)"))
    end

	target = zeros(Float32, sz)

    Threads.@threads for i in 0:Nz - 1
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
    plot_intensity_histogram(target, object_printed, thresholds; yscale=:log10, xlim=(0.0, 1.1)

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

The `patterns` are saved as HDF5 file with the name `data.hdf5` if `skip_hdf5=false`.
`overwrite=false` per default and does not overwrite existing files!
"""
function save_patterns(fpath, patterns, printed, angles, target; overwrite=true, skip_hdf5=true)
    @assert size(angles, 1) == size(patterns, 2) "Size mismatch between angles and patterns"
    #@assert size(patterns, 1) == size(target, 1) "Size mismatch between target and patterns"
    #@assert size(patterns, 3) == size(target, 3) "Size mismatch between target and patterns"

    # might be on GPU, move to CPU
    patterns = patterns isa Array ? patterns : Array(patterns)
    patterns = patterns ./ maximum(patterns)
    printed = printed isa Array ? printed : Array(printed)
    printed = printed ./ maximum(printed)
    angles = Array(angles)
    target = target isa Array ? target : Array(target)


    # check if path exists, otherwise create
    isdir(fpath) || mkpath(fpath)
    file = joinpath(fpath, "data.hdf5")
    if isfile(file) && overwrite==false
        throw(ArgumentError("HDF5 file exists already"))
    end
    
    if skip_hdf5 == false
        fid = h5open(file, "w")
        fid["patterns"] = patterns
        fid["printed"] = printed
        fid["angle"] = angles
        close(fid)
    end 
    # create folder for png files
    fpath_images = joinpath(fpath, "patterns_png")
    isdir(fpath) || mkpath(fpath_images) 


    # convert to proper Grayscale image
    # simshow normalizes the whole stack to [0, 1]
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
        number = lpad(string(i - 1), 5, "0")
        fname = joinpath(fpath_images, number * ".png")
        if isfile(fname) && overwrite==false
            throw(ArgumentError("png file exists already"))
        end
        save(fname, view(printed, :, :, i))
    end

    @info "Saved patterns to $fpath"

    return 0
end


const start_time = Ref(zeros(1))
const last_time = Ref(zeros(1))
"""
    log_time(x)

Used as callback within Optim.jl to print the time of each iteration.

"""
function log_time(x)
    if x[end].iteration == 0
        start_time[] .= time()
        last_time[] .= time()
    end
    @info "Iteration:\t$(x[end].iteration),\t total time:\t$(round(time()-start_time[][1], digits=2))s,\ttime since last iteration:\t$(round(time()-last_time[][1], digits=2))s, loss value: $(round(x[end].value, digits=4))"

    last_time[] .= time()
    return false
end
