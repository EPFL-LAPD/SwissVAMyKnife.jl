### A Pluto.jl notebook ###
# v0.19.42

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

# ╔═╡ 32d9b7b4-bacd-48c5-8ac5-a7d16bc26f3f
begin
	using Pkg
	Pkg.activate(".")
	Pkg.instantiate()
	using Revise
end

# ╔═╡ fb00f77e-3a43-43c6-83f5-ab9ffa690585
# this is our main package
using SwissVAMyKnife

# ╔═╡ 2a2c5c57-3b51-463b-90b3-bf4733df1c71
using ImageShow, ImageIO, PlutoUI, IndexFunArrays, Optim, RadonKA, FileIO, Colors, Plots

# ╔═╡ efc9b013-0452-482f-9112-d444d2197ba9
using CUDA

# ╔═╡ 5e2ffa51-c23b-4223-89f8-dfa138cfaca5
using NDTools

# ╔═╡ c973ccfd-11b7-4e0a-9d1a-32b0d449bfd1
md"# 0. Load packages
On the first run, Julia is going to install some packages automatically. So start this notebook and give it some minutes (5-10min) to install all packages. 
No worries, any future runs will be much faster to start!
"

# ╔═╡ 9122b9a9-43f4-48be-88f6-6d02a110f900
TableOfContents()

# ╔═╡ 54e9b00d-2589-430e-9ac9-ffdb48a54c0e
use_CUDA = Ref(true && CUDA.functional())

# ╔═╡ 5c3dda08-010e-4d76-bf35-1c3385a2a0c0
md" ## CUDA
CUDA accelerates the pattern generation easily by 5-20 times!
Otherwise most of the code will be multithreaded on your CPU but we strongly recommended the usage of CUDA for large scale 3D pattern generation.

Your CUDA is functional: **$(use_CUDA[])**
"

# ╔═╡ 93591474-679b-4944-a673-616cb5f354e0
var"@mytime" = use_CUDA[] ? CUDA.var"@time" : var"@time"

# ╔═╡ 0a7cc390-892a-4973-a725-f9765f7aaaff
togoc(x) = use_CUDA[] ? CuArray(x) : x

# ╔═╡ aebb7657-4b6c-4f22-b6be-c7ba054d7aa3
md"# 1. Load Benchy"

# ╔═╡ 9aa5e2c5-f0d0-43dc-9c3a-b4cebc5bb100
target = select_region(load_example_target("3DBenchy_120_aspect_ratio")[:, :, :], new_size=(240, 240, 150));

# ╔═╡ 6869714a-040b-4694-b099-8962445daa33
size(target)

# ╔═╡ e7715af7-de78-4668-9733-487f482ac3ef
md"z slide value $(@bind slice PlutoUI.Slider(axes(target, 3), show_value=true, default=77))"

# ╔═╡ 018120f1-8859-4606-ac62-507fbcaf4097
simshow(Array(target[:, :, slice]))

# ╔═╡ f1cbc5fd-8454-430c-9267-73c67eca623b
md"# 2. Specify Optimization Parameters"

# ╔═╡ 3f00e0e5-2d87-406c-9e4a-42b76c3bf2ec
loss = LossThresholdSparsity(thresholds=(0.85, 0.97), λ=2f-6)

# ╔═╡ e5bec833-47c5-4f65-8667-f28d4ccb1fbc
angles = range(0, 2π, 501)[begin:end-1]

# ╔═╡ e45eeeed-a86e-40e1-b4a2-e4c25cc6a368
optimizer = GradientBased(optimizer=Optim.LBFGS(), options=Optim.Options(iterations=5, store_trace=true))

# ╔═╡ f618cc32-6851-42de-b2c2-0ddc70b4668d
geometry_vial = VialRayOptics(
	angles=angles,
	μ=195.7197329044939,
	R_outer=(12.95e-3) / 2,
	R_inner=(11.36e-3) / 2,
	DMD_diameter=12.95e-3,
	n_vial=1.58,
	n_resin=1.4849
)

# ╔═╡ 29bbf20e-da89-42af-b7a0-a921c1c5fe3f
md"
In this case the DMD is smaller than the glass vial.
That means, our DMD area is smaller than the simulated target volume.
The resulting pattern size is: $(floor(Int, (10.0e-3 / 12.95e-3) * size(target, 1))) pixels

"

# ╔═╡ 3dfccb9a-8f21-4695-965c-c1d97077ba52
md"# 3. Optimize"

# ╔═╡ 2350dd2f-11dd-4b81-93b0-a05e88dd0fa7
@mytime patterns_vial, printed_intensity_vial, optim_res_vial = optimize_patterns(togoc(target), geometry_vial, 
								GradientBased(optimizer=Optim.LBFGS(), options=Optim.Options(iterations=30, store_trace=true,
								callback=SwissVAMyKnife.log_time)), loss)

# ╔═╡ 819bb337-ef70-4088-82da-bda208264b30
md"# 4. Inspect"

# ╔═╡ dc3457c8-e0df-4288-84eb-68ee844ee729
md"The intersection over union is: $(round(calculate_IoU(togoc(target), printed_intensity_vial .> (sum(loss.thresholds)/2)), digits=3))"

# ╔═╡ a59aa782-d446-431c-b6a9-65546989c9b8
md"Choose threshold for image: $(@bind thresh4 PlutoUI.Slider(0:0.01:1, show_value=true, default=0.7))"

# ╔═╡ dc987041-b76f-492c-9610-1da2d796fc80
md"z slice $(@bind slice2 PlutoUI.Slider(axes(target, 3), show_value=true, default=77))

Intensity distribution ----------- after threshold -------------------- target -------------------------------difference
"

# ╔═╡ fa85a991-7b26-4590-8b4c-f1fedd07ef41
[simshow(Array(printed_intensity_vial[:, :, slice2]), set_one=false, cmap=:turbo) simshow(ones((size(target, 1), 5))) simshow(thresh4 .< Array(printed_intensity_vial[:, :, slice2])) simshow(ones((size(target, 1), 5))) simshow(Array(target[:, :, slice2]))  simshow(ones((size(target, 1), 5))) simshow(Array(togoc(target)[:, :, slice2] .!= (thresh4 .< (printed_intensity_vial[:, :, slice2]))))]

# ╔═╡ adfb312f-f9ec-41e5-9670-4b1df5cee62f
maxx = maximum(printed_intensity_vial)

# ╔═╡ 9d07fad2-2d63-4f56-94c2-0988b59e5a90
heatmap(Array(printed_intensity_vial[:, :, slice2] ./ maxx), clim=(0, 1.05))

# ╔═╡ dad8320a-211a-48e8-b39e-adfd7ef1e219
md"## Pattern efficiency
The efficiency of the patterns is:
$(round(0.0 + 100 * sum(patterns_vial ./ maximum(patterns_vial)) / length(patterns_vial), digits=2))%
"	

# ╔═╡ ea432930-9c08-4882-be85-fb667b8d1354
plot_intensity_histogram(target, printed_intensity_vial, loss.thresholds)

# ╔═╡ 4031ba71-9a81-4ee8-a97f-0cd0f494c7bf
md"Different projection patterns: $(@bind angle PlutoUI.Slider(axes(patterns_vial, 2), show_value=true, default=0.5))"

# ╔═╡ c7b5b0bd-15a0-4127-a4c9-bfefdafccd50
simshow(Array(patterns_vial[:,angle,:] ./ maximum(patterns_vial))[end:-1:begin, :]', cmap=:turbo, set_one=false)

# ╔═╡ fe7dd475-11be-4ee6-a06c-bd4d2358e51f
md"# 5. Export patterns
The input size of the target is $(size(target)).
The output size of the patterns is $(size(patterns_vial)).

It means that there is $(length(angles)) angles. The height of the DMD patterns is $(size(patterns_vial, 3)) and the width $(size(patterns_vial, 1)).

The reason is, that the DMD size is smaller than the vial diameter.
In this case the pattern size is: 
$240 \cdot \frac{10e-3}{11.36e-3} \approx 185.3$


To change the patterns to the right DMD size, you can either interpolate or simulate a larger width and height of the target.

Click on the right of the bottom cell to *activate* it and save the patterns to the disk.
"

# ╔═╡ 81b3a243-7878-4c86-89af-bfe35def3b55
240 * 10e-3 / 12.95e-3

# ╔═╡ fbcd99fa-b084-4a0f-860d-15a67d50ca5c
# ╠═╡ disabled = true
#=╠═╡
save_patterns("/tmp/boat_patterns", select_region(patterns_vial, new_size=(768, length(angles), 1024)), printed_intensity_vial, angles, target)
  ╠═╡ =#

# ╔═╡ Cell order:
# ╟─c973ccfd-11b7-4e0a-9d1a-32b0d449bfd1
# ╠═32d9b7b4-bacd-48c5-8ac5-a7d16bc26f3f
# ╠═fb00f77e-3a43-43c6-83f5-ab9ffa690585
# ╠═2a2c5c57-3b51-463b-90b3-bf4733df1c71
# ╠═9122b9a9-43f4-48be-88f6-6d02a110f900
# ╟─5c3dda08-010e-4d76-bf35-1c3385a2a0c0
# ╠═efc9b013-0452-482f-9112-d444d2197ba9
# ╠═54e9b00d-2589-430e-9ac9-ffdb48a54c0e
# ╠═93591474-679b-4944-a673-616cb5f354e0
# ╠═0a7cc390-892a-4973-a725-f9765f7aaaff
# ╟─aebb7657-4b6c-4f22-b6be-c7ba054d7aa3
# ╠═5e2ffa51-c23b-4223-89f8-dfa138cfaca5
# ╠═9aa5e2c5-f0d0-43dc-9c3a-b4cebc5bb100
# ╠═6869714a-040b-4694-b099-8962445daa33
# ╟─e7715af7-de78-4668-9733-487f482ac3ef
# ╠═018120f1-8859-4606-ac62-507fbcaf4097
# ╟─f1cbc5fd-8454-430c-9267-73c67eca623b
# ╠═3f00e0e5-2d87-406c-9e4a-42b76c3bf2ec
# ╠═e5bec833-47c5-4f65-8667-f28d4ccb1fbc
# ╠═e45eeeed-a86e-40e1-b4a2-e4c25cc6a368
# ╠═f618cc32-6851-42de-b2c2-0ddc70b4668d
# ╟─29bbf20e-da89-42af-b7a0-a921c1c5fe3f
# ╟─3dfccb9a-8f21-4695-965c-c1d97077ba52
# ╠═2350dd2f-11dd-4b81-93b0-a05e88dd0fa7
# ╟─819bb337-ef70-4088-82da-bda208264b30
# ╟─dc3457c8-e0df-4288-84eb-68ee844ee729
# ╟─a59aa782-d446-431c-b6a9-65546989c9b8
# ╟─dc987041-b76f-492c-9610-1da2d796fc80
# ╠═fa85a991-7b26-4590-8b4c-f1fedd07ef41
# ╠═adfb312f-f9ec-41e5-9670-4b1df5cee62f
# ╠═9d07fad2-2d63-4f56-94c2-0988b59e5a90
# ╟─dad8320a-211a-48e8-b39e-adfd7ef1e219
# ╠═ea432930-9c08-4882-be85-fb667b8d1354
# ╟─4031ba71-9a81-4ee8-a97f-0cd0f494c7bf
# ╠═c7b5b0bd-15a0-4127-a4c9-bfefdafccd50
# ╟─fe7dd475-11be-4ee6-a06c-bd4d2358e51f
# ╠═81b3a243-7878-4c86-89af-bfe35def3b55
# ╠═fbcd99fa-b084-4a0f-860d-15a67d50ca5c
