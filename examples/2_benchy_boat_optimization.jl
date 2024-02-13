### A Pluto.jl notebook ###
# v0.19.38

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 022dde50-b50c-4198-b5d2-50cb95562a3f
begin
	using Pkg
	Pkg.activate(".")
	Pkg.instantiate()
	using Revise
end

# ╔═╡ d98ea233-c3dc-4e9e-b12d-15a7b687e8a8
# this is our main package
using SwissVAMyKnife

# ╔═╡ 7de2dce2-0faa-4b5a-a23d-3929f03413cd
using ImageShow, ImageIO, PlutoUI, IndexFunArrays, Optim, RadonKA, FileIO, Colors, Plots

# ╔═╡ b77c01d2-4d8f-4c83-b933-f2744e48350b
using CUDA

# ╔═╡ 0be4ad80-2433-4f91-866a-b8cbb2a7458a
using NDTools

# ╔═╡ 29d65ea9-fb36-414a-a70c-e26738dea956
using OpenEXR

# ╔═╡ 1f8a2f2b-e006-43e9-8fc8-2df2c72d8bc6
md"# 0. Load packages
On the first run, Julia is going to install some packages automatically. So start this notebook and give it some minutes (5-10min) to install all packages. 
No worries, any future runs will be much faster to start!
"

# ╔═╡ d8163acd-be6c-4176-89a1-a4a7643441f6
TableOfContents()

# ╔═╡ cf59efa4-8f0c-4796-a435-b89e3526b568
use_CUDA = Ref(true && CUDA.functional())

# ╔═╡ a5d61a5f-54c9-4560-9a68-3e2d70a62f14
md" ## CUDA
CUDA accelerates the pattern generation easily by 5-20 times!
Otherwise most of the code will be multithreaded on your CPU but we strongly recommended the usage of CUDA for large scale 3D pattern generation.

Your CUDA is functional: **$(use_CUDA[])**
"

# ╔═╡ c9c2b7b7-96bb-4756-b191-418e5adca43c
var"@mytime" = use_CUDA[] ? CUDA.var"@time" : var"@time"

# ╔═╡ 2794d1bf-dea1-494a-917d-7d675d8bc8a3
togoc(x) = use_CUDA[] ? CuArray(x) : x

# ╔═╡ c1240545-a674-4e3c-8386-0b6cb4591fc6


# ╔═╡ 3665c1e2-ae84-47a9-89db-7f5dc6657e4c
function load_benchy(sz, sz_file, path)
	target = zeros(Float32, sz)
	#@show size(load(joinpath(path, string("slice_", string(1, pad=3) ,".png"))))
	
	for i in 0:sz_file[1]-1
		target[i + 1, :, :] .= select_region(Gray.(load(joinpath(path, string("slice_", string(i, pad=3) ,".png")))), new_size=(sz_file))
	end

	#target2 = zeros(Float32, sz)
	#WaveOpticsPropagation.set_center!(target2, target
	target2 = select_region(target, new_size=sz)
	return togoc(target2)
end

# ╔═╡ 246b8900-a4dd-4caf-8f34-9d868d6127b2
target = permutedims(load_benchy((255,255,255), (255,255,255), "/home/felix/boat_slices/"), (1,3,2))

# ╔═╡ 49be1b07-dc49-4515-9afc-d66414fd7f59
simshow(Array(target[:,:,150]))

# ╔═╡ 93cebf3f-389b-4fde-8d13-af3e1165ad9c
geometry = ParallelRayOptics(π / 2 .+ range(0, 2π, 401)[begin:end-1], nothing)

# ╔═╡ 4e876bff-ac18-4d85-bd1b-1adf0d72f659
# ╠═╡ disabled = true
#=╠═╡
begin
	LL = 9
	target = box(Float32, (LL, LL), (3,3), offset=(6,6)) .-  box(Float32, (LL, LL), (1,1), offset=(6,6))
end;
  ╠═╡ =#

# ╔═╡ 971f7c2f-415f-4e73-aaea-acda355e2f5a
simshow(target)

# ╔═╡ d1fb09d5-cee0-4027-acaa-24ccabaf9cf0
loss = LossThreshold(thresholds=(0.65, 0.75))

# ╔═╡ 792e4c29-7f0c-4d9b-833b-8341702e472e
@mytime patterns, printed_intensity, optim_res = optimize_patterns(togoc(target), geometry, 
								GradientBased(optimizer=Optim.LBFGS(), options=Optim.Options(iterations=20, store_trace=true))					
								, loss)

# ╔═╡ da8a4c5f-7439-498b-b4c2-678987b45f2c
@bind thresh4 PlutoUI.Slider(0:0.01:1, show_value=true, default=0.5)

# ╔═╡ 0dac21c5-5576-499c-8cb8-e3f1e7c7f39c
md"z slider value $(@bind slice2 PlutoUI.Slider(axes(target, 3), show_value=true, default=0.5))"

# ╔═╡ b1690362-a23d-4047-a9c4-18fcc625ef3e
[simshow(Array(printed_intensity[:, :, slice2]), set_one=false) simshow(ones((size(target, 1), 5))) simshow(thresh4 .< Array(printed_intensity[:, :, slice2])) simshow(ones((size(target, 1), 5))) simshow(Array(target[:, :, slice2]))]

# ╔═╡ d7de379b-601f-4bd5-b26d-45d424811aa2
plot_intensity_histogram(target, printed_intensity, (0.65, 0.75))

# ╔═╡ 21a99ca0-6b97-4a83-9780-bb120f190afd
md"angle $(@bind angle PlutoUI.Slider(axes(patterns, 2), show_value=true, default=0.5))"

# ╔═╡ e14faca6-62d4-4551-aadc-5342cb6d2aed
simshow(Array(patterns[:,angle,:])[end:-1:begin, :]', cmap=:turbo, set_one=true)

# ╔═╡ 0c3ff4cc-09ab-479c-a5a8-2194c1db9891
OpenEXR.save("/tmp/pattern.exr", simshow(simshow(Array(patterns[:,angle,:])[end:-1:begin, :]', cmap=:gray, set_one=true) .* 0.007))

# ╔═╡ Cell order:
# ╠═1f8a2f2b-e006-43e9-8fc8-2df2c72d8bc6
# ╠═022dde50-b50c-4198-b5d2-50cb95562a3f
# ╠═d98ea233-c3dc-4e9e-b12d-15a7b687e8a8
# ╠═7de2dce2-0faa-4b5a-a23d-3929f03413cd
# ╠═d8163acd-be6c-4176-89a1-a4a7643441f6
# ╟─a5d61a5f-54c9-4560-9a68-3e2d70a62f14
# ╠═b77c01d2-4d8f-4c83-b933-f2744e48350b
# ╠═cf59efa4-8f0c-4796-a435-b89e3526b568
# ╠═c9c2b7b7-96bb-4756-b191-418e5adca43c
# ╠═2794d1bf-dea1-494a-917d-7d675d8bc8a3
# ╠═c1240545-a674-4e3c-8386-0b6cb4591fc6
# ╠═3665c1e2-ae84-47a9-89db-7f5dc6657e4c
# ╠═0be4ad80-2433-4f91-866a-b8cbb2a7458a
# ╠═246b8900-a4dd-4caf-8f34-9d868d6127b2
# ╠═49be1b07-dc49-4515-9afc-d66414fd7f59
# ╠═93cebf3f-389b-4fde-8d13-af3e1165ad9c
# ╠═4e876bff-ac18-4d85-bd1b-1adf0d72f659
# ╠═971f7c2f-415f-4e73-aaea-acda355e2f5a
# ╠═d1fb09d5-cee0-4027-acaa-24ccabaf9cf0
# ╠═792e4c29-7f0c-4d9b-833b-8341702e472e
# ╠═da8a4c5f-7439-498b-b4c2-678987b45f2c
# ╠═0dac21c5-5576-499c-8cb8-e3f1e7c7f39c
# ╠═b1690362-a23d-4047-a9c4-18fcc625ef3e
# ╠═d7de379b-601f-4bd5-b26d-45d424811aa2
# ╟─21a99ca0-6b97-4a83-9780-bb120f190afd
# ╠═e14faca6-62d4-4551-aadc-5342cb6d2aed
# ╠═0c3ff4cc-09ab-479c-a5a8-2194c1db9891
# ╠═29d65ea9-fb36-414a-a70c-e26738dea956
