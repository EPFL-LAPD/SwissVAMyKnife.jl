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

# ╔═╡ fb4bb1e2-c5e3-11ee-2f22-0bf5f621e215
begin
	using Pkg
	Pkg.activate(".")
	Pkg.instantiate()
	using Revise
end

# ╔═╡ 2487fdcc-f8c0-426d-9eff-ca9c8ce98100
# this is our main package
using SwissVAMyKnife

# ╔═╡ 53c94ae4-3a86-4f3b-8e35-832859c5b465
using ImageShow, ImageIO, PlutoUI, IndexFunArrays, Optim, RadonKA, FileIO, Colors, Plots

# ╔═╡ 2dc23db0-87b8-4044-9408-b2ffcde33ee8
using CUDA

# ╔═╡ 9c478480-163a-4fe0-ad6b-32a337a616b9
md"# 0. Load packages
On the first run, Julia is going to install some packages automatically. So start this notebook and give it some minutes (5-10min) to install all packages. 
No worries, any future runs will be much faster to start!
"

# ╔═╡ daf8043e-293f-495c-9b0d-b4e2b50de4a7
TableOfContents()

# ╔═╡ 9ec75d19-98be-4e04-990e-fd82b967525c
use_CUDA = Ref(true && CUDA.functional())

# ╔═╡ de126e87-55c7-44ef-935d-b4bd96591914
md" ## CUDA
CUDA accelerates the pattern generation easily by 5-20 times!
Otherwise most of the code will be multithreaded on your CPU but we strongly recommended the usage of CUDA for large scale 3D pattern generation.

Your CUDA is functional: **$(use_CUDA[])**
"

# ╔═╡ bd516459-0be9-4ccb-a95b-32d07c15b410
var"@mytime" = use_CUDA[] ? CUDA.var"@time" : var"@time"

# ╔═╡ 9d1ffd4d-1fe3-4f5c-bdc3-817fd0f23c7b
togoc(x) = use_CUDA[] ? CuArray(x) : x

# ╔═╡ 465e830f-ca33-427a-ad7b-e18f3bbaa3c4
md"# 1. Let's start with a simple 2D pattern
However, we add a trailing 1 z dimension.
"

# ╔═╡ 4475e0c5-4ae6-403f-bb2b-c5bc8a25d18a
sz = (256, 256, 1)

# ╔═╡ b2dae3ea-5a40-498f-a134-5a9a98e58de0
# ╠═╡ disabled = true
#=╠═╡
target = box(Float32, sz, (150, 110, 1), offset=(90, 160, 1)) .-  box(Float32, sz, (80, 50, 1), offset=(90, 160, 1));
  ╠═╡ =#

# ╔═╡ 827f6f0a-12d9-4a72-8f8b-c3869a037590
target = box(Float32, sz, (150, 110, 1)) .-  box(Float32, sz, (80, 50, 1));

# ╔═╡ 047023bf-96dd-4427-aa13-3fb2340af90e
simshow(target[:, :, 1])

# ╔═╡ 1d2bdd44-95d2-45ef-8e24-c8ca3df7485a
md"# 2. Simple optimization


## Loss Function
One simple optimization algorithm tries to distribute the intensity as following.
Object pixels should receive enough intensity to polymerize.
Void pixels should stay below this intensity threshold and should not polymerize.
Also, object pixels should not strongly over polymerize

Let's use the `LossThreshold` to achieve this:
"

# ╔═╡ ae3fdae9-7b3c-4391-8fff-530f7758942c
loss = LossThreshold(thresholds=(0.9, 0.98))

# ╔═╡ b58c9e1f-7ca6-4171-8004-2b84d33cf335
md"## Specify Geometry

First we try with a parallel geometry. Meaning that the rays are not refracted upon entry into the glass vial.
This corresponds to the situation with an index matching bath!

For this optimization we use 400 angles in the range from 0 to 2π
As absorption we set $μ=nothing$, so we ignore absorption first!
"

# ╔═╡ 0dd200a7-a0f0-416e-ad69-7343e210a36e
angles = range(0, 2π, 401)[begin:end-1]

# ╔═╡ 303a363f-f941-45b9-9194-62a2c411964a
μ = nothing

# ╔═╡ 6d130809-8b78-485c-bba1-fc958bc9d873
geometry = ParallelRayOptics(angles=angles, μ=μ, DMD_diameter=10e-3)

# ╔═╡ cb8b9ccc-4085-4abe-bd71-5e70cc8decfa
md" As optimizer we use a gradient descent based variant"

# ╔═╡ e5f0660d-2adb-4c82-8945-e8164ec500f6
optimizer = GradientBased(optimizer=Optim.LBFGS(), options=Optim.Options(iterations=20, store_trace=true))

# ╔═╡ b767afa8-784c-453b-ad0d-a1daf4d5ba5c
md"
Let's try to run the optimization: `togoc` moves the target onto the CUDA device, if possible"

# ╔═╡ 07e97af6-f30b-4100-999c-c61b52cec76e
@mytime patterns, printed_intensity, optim_res = optimize_patterns(togoc(target), geometry, optimizer, loss)


# ╔═╡ c13abe2d-ff27-42e2-9534-cb1936e68236
optim_res

# ╔═╡ 9887da10-96fa-4509-9b91-a6c21354f4ba
md"# 3. Look at the results"

# ╔═╡ f79bdf14-fefa-449f-b13a-49e6fb9153bf
md"We can inspect the histogram of the optimization.

We see, void pixels receive less intensity than the threshold.
Object pixels stay well above it
"

# ╔═╡ 66560fd0-3cee-41ca-bb13-dee2a78bbb9b
plot_intensity_histogram(target, printed_intensity, loss.thresholds)

# ╔═╡ dc538ea6-54e3-4a0f-a5ef-427802239cf2
md"threshold value $(@bind thresh PlutoUI.Slider(0:0.01:1.05, show_value=true, default=0.5))"

# ╔═╡ 71333d50-fa28-4bdf-bad4-d787682e043a
md"

Printed intensity ----------------------------- Printed intensity after threshold ------------------- target 
"

# ╔═╡ 5801f61a-79fa-4326-82dd-3901dd4d493a
[simshow(Array(printed_intensity[:, :, 1])) simshow(ones((sz[1], 5))) simshow(thresh .< Array(printed_intensity[:, :, 1])) simshow(ones((sz[1], 5))) simshow(target[:, :, 1])]

# ╔═╡ 605e1421-23dd-465d-b94d-3a649154f6f8
simshow(360 .< Array(backproject(patterns, angles; μ)[:, :, 1]))

# ╔═╡ b08e897d-c260-4a42-8065-989abbbcab4e
simshow(Array(backproject(reverse(patterns, dims=Tuple(())), .-angles; μ)[:, :, 1]))

# ╔═╡ 44ea2ea2-1e4f-44d3-aa02-68e96d6c7b39
size(patterns)

# ╔═╡ 92f5b987-13c7-463c-b150-71c51ec4aa72
simshow(Array(patterns)[:, :, 1], cmap=:turbo)

# ╔═╡ 6654e278-3d71-48a3-9382-7b8e486f4d72
sum(patterns ./ maximum(patterns)) / length(patterns)

# ╔═╡ 11af6da4-5fa2-4e48-a1c8-32bc686b8239
md"# 4. Include Absorption of the Photo initiator
Let's add some absorption!
$\mu$ expects units of inverse meters.
Empirically, a factor of 3/10mm is possible to print. 
This is equivalent that only $\exp(-3)$ of the light is left after the propagation through the vial (if the vial has a diameter of 10mm).
A higher $\mu$ means actually shorter printing times. 
"

# ╔═╡ 2f51ec68-09c6-4475-81b9-ec252fba5df7
μ2 = 3 / (10e-3)

# ╔═╡ 2e76c088-e61f-4deb-ac69-b73d6e3f361c
geometry_μ = ParallelRayOptics(angles=angles, DMD_diameter=10e-3, μ=μ2)

# ╔═╡ 0e4da85b-99bd-47e0-a8fe-3c69c3a05708
@mytime patterns_μ, printed_intensity_μ, optim_res_μ = optimize_patterns(togoc(target), geometry_μ, 
								GradientBased(optimizer=Optim.LBFGS(), options=Optim.Options(iterations=100, store_trace=true))					
								, loss)

# ╔═╡ a04d12c2-f718-44e0-a755-da1132c0330e
plot_intensity_histogram(target, printed_intensity_μ,  loss.thresholds)

# ╔═╡ 5ade07d4-e7b3-4e78-94fa-454924a4066c
@bind thresh2 PlutoUI.Slider(0:0.01:1, show_value=true, default=0.5)

# ╔═╡ ccd0ad5e-5c01-4ec0-afc4-0dcce85dc9ef
[simshow(Array(printed_intensity_μ[:, :, 1])) simshow(ones((sz[1], 5))) simshow(thresh2 .< Array(printed_intensity_μ[:, :, 1])) simshow(ones((sz[1], 5))) simshow(target[:, :, 1])]

# ╔═╡ 17621d91-1f93-4fcf-abf1-5e97463a0bdf
simshow(Array(patterns_μ[:,:,1]), cmap=:turbo)

# ╔═╡ 05c6a38e-a87d-46da-acb6-33b393369f4c
sum(patterns_μ) / (maximum(patterns_μ) * length(patterns))

# ╔═╡ f940c2a6-ebc2-4dca-986c-fe25bfa9e4f0
md"""# 5. Include refraction of the glass vial 

Printing without index matching bath is more challenging but still possible.
The glass vial changes the direction.
Because of our flexible [RadonKA.jl](https://github.com/roflmaostc/RadonKA.jl) package, we can simulate those scenarios.

Analytically, we can derive the ray direction after propagation through a glass interface of the vial and then entering the resin.

Luckily, you just have to specify the following parameters for the geometry:
 
 - `angles` is a range or `Vector` (or `CuVector`) storing the illumination angles.
 - `μ` is the absorption coefficient of the resin in units of pixels.
 - So `μ=0.1` means that after ten pixels of propagation the intensity is `I(10) = I_0 * exp(-10 * 0.1)`.
 - `R_outer` is the outer radius of the glass vial.
 - `R_inner` is the inner radius of the glass vial.
 - `DMD_diameter` is the diameter of the DMD along the vial radius. So this is not the height along the rotation axis!
 - `n_vial` is the refractive index of the glass vial.
 - `n_resin` is the refractive index of the resin.
"""

# ╔═╡ f9a82207-6841-44c9-9aec-e8da3de8e0b0
geometry_vial = VialRayOptics(
	angles=angles,
	μ=0/256,
	R_outer=(16.60e-3) / 2,
	R_inner=(15.2e-3) / 2,
	DMD_diameter=16.60e-3,
	n_vial=1.47,
	n_resin=1.4849
)

# ╔═╡ 675764cb-721a-4b89-92c4-8a1ab7f5867f
@mytime patterns_vial, printed_intensity_vial, optim_res_vial = optimize_patterns(togoc(target), geometry_vial, 
								GradientBased(optimizer=Optim.LBFGS(), options=Optim.Options(iterations=100, store_trace=true))					
								, loss)

# ╔═╡ 71b479a0-29fc-4411-a02d-096427f2a631
size(patterns_vial)

# ╔═╡ 9ddd098d-2d78-4de8-a322-40a2463adcda
plot_intensity_histogram(target, printed_intensity_vial, loss.thresholds)

# ╔═╡ d56eb3ab-4698-4ee4-aae1-461b964b778c
@bind thresh3 PlutoUI.Slider(0:0.01:1, show_value=true, default=0.5)

# ╔═╡ b6a6237f-b03f-402d-9693-f00044e3539e
[simshow(Array(printed_intensity_vial[:, :, 1])) simshow(ones((sz[1], 5))) simshow(thresh3 .< Array(printed_intensity_vial[:, :, 1])) simshow(ones((sz[1], 5))) simshow(target[:, :, 1])]

# ╔═╡ eda29eed-507c-4f55-b1e6-590c9616db8e
md"
The sinogram looks very similar to the one with an index matching bath.
Some people do post-optimization interpolation.
However, this is not exactly if absorption takes place.
"

# ╔═╡ fc239b37-dc88-48cd-9479-99074d96ece5
simshow(Array(patterns_vial[:,:,1]), cmap=:turbo)

# ╔═╡ 65d23a80-ad8b-4b36-90a0-d6427c19ff88
md"
The efficiency can be defined as the mean of the pixel values divided by the maximum.
Because in experiment the real absolute maximum is fixed by the light source intensity. Hence we would like to maximize the real efficiency of the DMD
"

# ╔═╡ 7de5d398-20d6-44e2-b2eb-3007111e146f
sum(patterns_vial) / (maximum(patterns_vial) * length(patterns_vial))

# ╔═╡ 79bea0b4-d990-4269-8e4f-6abe19023f87
md"# 6. Reduce Sparsity"

# ╔═╡ 699a0dbd-df85-41b9-b751-cc936d6c03c5
loss_sparse = LossThresholdSparsity(thresholds=(0.9f0, 0.97f0), λ=500f-6)

# ╔═╡ 006aa5f6-8008-455a-a2a6-54eeb0c097bf
@mytime patterns_vial_s, printed_intensity_vial_s, optim_res_vial_s = optimize_patterns(togoc(target), geometry_vial, 
								GradientBased(optimizer=Optim.LBFGS(), options=Optim.Options(iterations=50, store_trace=true))					
								, loss_sparse)

# ╔═╡ 9f7d62be-378f-4b3c-bf38-239c9daa8705
plot_intensity_histogram(target, printed_intensity_vial_s, loss_sparse.thresholds)

# ╔═╡ 93f5bffa-f916-4d4b-90f5-751613b48efb
calculate_IoU(target, Array(printed_intensity_vial_s .> 0.93))

# ╔═╡ b4c7ed0d-e31e-417b-abaf-2193980c9378
@bind thresh5 PlutoUI.Slider(0:0.01:1, show_value=true, default=0.5)

# ╔═╡ afa94e1a-d426-4a29-b0fd-73bcdc3e914f
[simshow(Array(printed_intensity_vial_s[:, :, 1])) simshow(ones((sz[1], 5))) simshow(thresh5 .< Array(printed_intensity_vial_s[:, :, 1])) simshow(ones((sz[1], 5))) simshow(target[:, :, 1])]

# ╔═╡ 0fc8abe6-2a22-477e-b2a4-6ccba1e16991
simshow(Array(patterns_vial_s[:,:,1]), cmap=:turbo)

# ╔═╡ dc0c21f2-980a-4c97-9a9d-b3d824121bbe
md"
The efficiency can be defined as the mean of the pixel values divided by the maximum.
Because in experiment the real absolute maximum is fixed by the light source intensity. Hence we would like to maximize the real efficiency of the DMD.
In this case the efficiency is $\eta$ = $(sum(patterns_vial_s) / (maximum(patterns_vial_s) * length(patterns_vial)))
"

# ╔═╡ a0f85250-7c40-4026-b2b2-d267ec58de6f
histogram(xlabel="pixel intensity", ylabel="Occurence", Array(patterns_vial)[:], yscale=:log10, title="Without sparsity term")

# ╔═╡ 0a805204-d2ad-4672-88ab-4acdff243083
histogram(xlabel="pixel intensity", ylabel="Occurence", Array(patterns_vial_s)[:], yscale=:log10, title="With sparsity term")

# ╔═╡ a7cafba6-849a-4eb8-9709-a76cb98e9879
md"# 7. Let's do a bigger 3D object!
Be a little patient, this might take 10 seconds on a GPU.
On a CPU much longer accordingly, some minutes.

We deactive the cell by default. However of the (...) next to the cell and click *Enable Cell*

"

# ╔═╡ 8824fb56-0fbf-4cba-9aea-6449627923f2
geometry_vial2 = VialRayOptics(
	angles=range(0, 2π, 1000)[begin:end-1],
	μ=nothing,
	R_outer=(16.60e-3) / 2,
	R_inner=(15.2e-3) / 2,
	n_vial=1.47,
	n_resin=1.4849,
	DMD_diameter=10e-3
)

# ╔═╡ 0a655d51-e3b6-413b-83de-9781974242a2
begin
	KK = 400
	target_3D = box(Float32, (KK, KK, KK), (80, 80, 100)) .- 
	box(Float32, (KK, KK, KK), (60, 50, 80));
	target_3D = Float32.(Bool.(target_3D) .|| (rr2(size(target_3D)) .< 30^2))
end;

# ╔═╡ 877b1484-969c-45d7-a3e7-4f0301a81a4b
md"z slide value $(@bind slice PlutoUI.Slider(axes(target_3D, 3), show_value=true, default=0.5))"

# ╔═╡ a05d8dd4-40c3-41da-9dbd-f58ab161b2d4
simshow(target_3D[:, :, slice])

# ╔═╡ 1bdd07f2-35df-4d5e-b296-7159ae774aa3
geometry3 = ParallelRayOptics(angles=angles[1:1], μ=1/16.6e-3, DMD_diameter=16.6e-3)

# ╔═╡ d6a59254-bffe-4116-9335-8884eb44556f
@mytime patterns_3D, printed_intensity_3D, optim_res_3D = optimize_patterns(togoc(target_3D), geometry_vial2, 
								GradientBased(optimizer=Optim.LBFGS(m=3), options=Optim.Options(iterations=20, store_trace=true))					
								, loss)

# ╔═╡ 6009b601-6988-4be8-a519-7c59660c73ab
optim_res_3D

# ╔═╡ fa45b520-18a8-4465-b642-a1c08de48e20
md"threshold value=$(@bind thresh4 PlutoUI.Slider(0:0.01:1, show_value=true, default=0.5))"

# ╔═╡ 10cc0e79-5d5b-43cf-afb8-ad21875d6d97
md"z slider value $(@bind slice2 PlutoUI.Slider(axes(target_3D, 3), show_value=true, default=0.5))"

# ╔═╡ 2a879d1f-c430-48dd-9264-4c1d6b5bbf18
exp(-0.99)

# ╔═╡ 0835d1b0-a562-4edf-939e-3044f94e53c7
[simshow(Array(printed_intensity_3D[:, :, slice2]), set_one=false) simshow(ones((size(target_3D, 1), 5))) simshow(thresh4 .< Array(printed_intensity_3D[:, :, slice2])) simshow(ones((size(target_3D, 1), 5))) simshow(target_3D[:, :, slice2])]

# ╔═╡ 83793ffe-38de-4a76-b80f-8dee1d18c5a5
plot_intensity_histogram(target_3D, printed_intensity_3D, loss.thresholds, xlim=(0.7, 1.1))

# ╔═╡ 262b96e8-a78c-441c-9b46-2d87636286d7
md"angle $(@bind angle PlutoUI.Slider(axes(patterns_3D, 2), show_value=true, default=0.5))"

# ╔═╡ c8774a2b-47e0-4be6-b213-bdef6d7b0726
simshow(Array(patterns_3D[:,angle,:]), cmap=:turbo, set_one=true)

# ╔═╡ Cell order:
# ╟─9c478480-163a-4fe0-ad6b-32a337a616b9
# ╠═fb4bb1e2-c5e3-11ee-2f22-0bf5f621e215
# ╠═2487fdcc-f8c0-426d-9eff-ca9c8ce98100
# ╠═53c94ae4-3a86-4f3b-8e35-832859c5b465
# ╟─daf8043e-293f-495c-9b0d-b4e2b50de4a7
# ╟─de126e87-55c7-44ef-935d-b4bd96591914
# ╠═2dc23db0-87b8-4044-9408-b2ffcde33ee8
# ╠═9ec75d19-98be-4e04-990e-fd82b967525c
# ╠═bd516459-0be9-4ccb-a95b-32d07c15b410
# ╠═9d1ffd4d-1fe3-4f5c-bdc3-817fd0f23c7b
# ╟─465e830f-ca33-427a-ad7b-e18f3bbaa3c4
# ╠═4475e0c5-4ae6-403f-bb2b-c5bc8a25d18a
# ╠═b2dae3ea-5a40-498f-a134-5a9a98e58de0
# ╠═827f6f0a-12d9-4a72-8f8b-c3869a037590
# ╠═047023bf-96dd-4427-aa13-3fb2340af90e
# ╟─1d2bdd44-95d2-45ef-8e24-c8ca3df7485a
# ╠═ae3fdae9-7b3c-4391-8fff-530f7758942c
# ╟─b58c9e1f-7ca6-4171-8004-2b84d33cf335
# ╠═0dd200a7-a0f0-416e-ad69-7343e210a36e
# ╠═303a363f-f941-45b9-9194-62a2c411964a
# ╠═6d130809-8b78-485c-bba1-fc958bc9d873
# ╟─cb8b9ccc-4085-4abe-bd71-5e70cc8decfa
# ╠═e5f0660d-2adb-4c82-8945-e8164ec500f6
# ╟─b767afa8-784c-453b-ad0d-a1daf4d5ba5c
# ╠═07e97af6-f30b-4100-999c-c61b52cec76e
# ╠═c13abe2d-ff27-42e2-9534-cb1936e68236
# ╟─9887da10-96fa-4509-9b91-a6c21354f4ba
# ╟─f79bdf14-fefa-449f-b13a-49e6fb9153bf
# ╠═66560fd0-3cee-41ca-bb13-dee2a78bbb9b
# ╟─dc538ea6-54e3-4a0f-a5ef-427802239cf2
# ╟─71333d50-fa28-4bdf-bad4-d787682e043a
# ╟─5801f61a-79fa-4326-82dd-3901dd4d493a
# ╠═605e1421-23dd-465d-b94d-3a649154f6f8
# ╠═b08e897d-c260-4a42-8065-989abbbcab4e
# ╠═44ea2ea2-1e4f-44d3-aa02-68e96d6c7b39
# ╠═92f5b987-13c7-463c-b150-71c51ec4aa72
# ╠═6654e278-3d71-48a3-9382-7b8e486f4d72
# ╟─11af6da4-5fa2-4e48-a1c8-32bc686b8239
# ╠═2f51ec68-09c6-4475-81b9-ec252fba5df7
# ╠═2e76c088-e61f-4deb-ac69-b73d6e3f361c
# ╠═0e4da85b-99bd-47e0-a8fe-3c69c3a05708
# ╠═a04d12c2-f718-44e0-a755-da1132c0330e
# ╟─5ade07d4-e7b3-4e78-94fa-454924a4066c
# ╟─ccd0ad5e-5c01-4ec0-afc4-0dcce85dc9ef
# ╠═17621d91-1f93-4fcf-abf1-5e97463a0bdf
# ╠═05c6a38e-a87d-46da-acb6-33b393369f4c
# ╟─f940c2a6-ebc2-4dca-986c-fe25bfa9e4f0
# ╠═f9a82207-6841-44c9-9aec-e8da3de8e0b0
# ╠═71b479a0-29fc-4411-a02d-096427f2a631
# ╠═675764cb-721a-4b89-92c4-8a1ab7f5867f
# ╠═9ddd098d-2d78-4de8-a322-40a2463adcda
# ╟─d56eb3ab-4698-4ee4-aae1-461b964b778c
# ╟─b6a6237f-b03f-402d-9693-f00044e3539e
# ╟─eda29eed-507c-4f55-b1e6-590c9616db8e
# ╠═fc239b37-dc88-48cd-9479-99074d96ece5
# ╟─65d23a80-ad8b-4b36-90a0-d6427c19ff88
# ╟─7de5d398-20d6-44e2-b2eb-3007111e146f
# ╟─79bea0b4-d990-4269-8e4f-6abe19023f87
# ╠═699a0dbd-df85-41b9-b751-cc936d6c03c5
# ╠═006aa5f6-8008-455a-a2a6-54eeb0c097bf
# ╠═9f7d62be-378f-4b3c-bf38-239c9daa8705
# ╠═93f5bffa-f916-4d4b-90f5-751613b48efb
# ╟─b4c7ed0d-e31e-417b-abaf-2193980c9378
# ╟─afa94e1a-d426-4a29-b0fd-73bcdc3e914f
# ╠═0fc8abe6-2a22-477e-b2a4-6ccba1e16991
# ╟─dc0c21f2-980a-4c97-9a9d-b3d824121bbe
# ╟─a0f85250-7c40-4026-b2b2-d267ec58de6f
# ╟─0a805204-d2ad-4672-88ab-4acdff243083
# ╟─a7cafba6-849a-4eb8-9709-a76cb98e9879
# ╠═8824fb56-0fbf-4cba-9aea-6449627923f2
# ╠═0a655d51-e3b6-413b-83de-9781974242a2
# ╟─877b1484-969c-45d7-a3e7-4f0301a81a4b
# ╠═a05d8dd4-40c3-41da-9dbd-f58ab161b2d4
# ╠═1bdd07f2-35df-4d5e-b296-7159ae774aa3
# ╠═d6a59254-bffe-4116-9335-8884eb44556f
# ╠═6009b601-6988-4be8-a519-7c59660c73ab
# ╟─fa45b520-18a8-4465-b642-a1c08de48e20
# ╟─10cc0e79-5d5b-43cf-afb8-ad21875d6d97
# ╠═2a879d1f-c430-48dd-9264-4c1d6b5bbf18
# ╟─0835d1b0-a562-4edf-939e-3044f94e53c7
# ╠═83793ffe-38de-4a76-b80f-8dee1d18c5a5
# ╟─262b96e8-a78c-441c-9b46-2d87636286d7
# ╠═c8774a2b-47e0-4be6-b213-bdef6d7b0726
