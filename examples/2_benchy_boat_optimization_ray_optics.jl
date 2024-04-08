### A Pluto.jl notebook ###
# v0.19.40

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

# ╔═╡ 5fe8e968-7017-4fde-97cc-6459e951811c
using WaveOpticsPropagation

# ╔═╡ 7279f1d1-6423-44ae-b071-7c2847026ca9
md"""# Supplemental simulations

Note, in the publication we optimize a volume of 550x550x550 voxels which only runs on a large GPU (NVIDIA A100 with 80GB). Hence, here a smaller version which produces similar but not identical results because of a different discretization of the boat.

```bibtex
@misc{wechsler2024wave,
      title={Wave optical model for tomographic volumetric additive manufacturing}, 
      author={Felix Wechsler and Carlo Gigli and Jorge Madrid-Wolff and Christophe Moser},
      year={2024},
      eprint={2402.06283},
      archivePrefix={arXiv},
      primaryClass={physics.optics}
}
```
"""

# ╔═╡ 1f8a2f2b-e006-43e9-8fc8-2df2c72d8bc6
md"# 0. Load packages
On the first run, Julia is going to install some packages automatically. So start this notebook and give it some minutes (5-10min) to install all packages. 
No worries, any future runs will be much faster to start!
"

# ╔═╡ 1f595779-173a-4892-bbfc-0435dcb46434
Pkg.resolve()

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
md"# 1. Load Benchy"

# ╔═╡ 246b8900-a4dd-4caf-8f34-9d868d6127b2
target = load_example_target("3DBenchy_180");

# ╔═╡ 02c93620-2cf6-4446-a280-d4a7934f0ecd
md"z slide value $(@bind slice PlutoUI.Slider(axes(target, 3), show_value=true, default=77))"

# ╔═╡ 971f7c2f-415f-4e73-aaea-acda355e2f5a
simshow(Array(target[:, :, slice]))

# ╔═╡ cebf24f8-6efe-407c-b123-5289dbb60790
md"# 2. Specify Optimization Parameters"

# ╔═╡ d1fb09d5-cee0-4027-acaa-24ccabaf9cf0
loss = LossThreshold(thresholds=(0.7, 0.8))

# ╔═╡ 8328d500-2760-46db-9733-479763d5c08f
angles = range(0, 2π, 201)[begin:end-1]

# ╔═╡ 93cebf3f-389b-4fde-8d13-af3e1165ad9c
geometry = ParallelRayOptics(angles, nothing)

# ╔═╡ 2ffbef8c-066d-49b6-8a29-1ef710099029
optimizer = GradientBased(optimizer=Optim.LBFGS(), options=Optim.Options(iterations=40, store_trace=true))

# ╔═╡ c0529850-c323-4c72-bab1-ec414d3ad425
md"# 3. Optimize"

# ╔═╡ 792e4c29-7f0c-4d9b-833b-8341702e472e
@mytime patterns, printed_intensity, optim_res = optimize_patterns(togoc(target),
	geometry, optimizer, loss)

# ╔═╡ 7c9f675d-3698-459d-93fa-d8e297a7ffa5
optim_res

# ╔═╡ e94fb1d2-06c3-4f45-91ed-411fe8bd033f
md"# 4. Inspect"

# ╔═╡ 49952e37-7f07-4719-b77a-971a521e155a
md"The intersection over union is: $(round(calculate_IoU(togoc(target), printed_intensity .> 0.7), digits=3))"

# ╔═╡ da8a4c5f-7439-498b-b4c2-678987b45f2c
md"Choose threshold for image: $(@bind thresh4 PlutoUI.Slider(0:0.01:1, show_value=true, default=0.7))"

# ╔═╡ 0dac21c5-5576-499c-8cb8-e3f1e7c7f39c
md"z slice $(@bind slice2 PlutoUI.Slider(axes(target, 3), show_value=true, default=77))

Intensity distribution ----------- after threshold -------------------- target -------------------------------difference
"

# ╔═╡ b1690362-a23d-4047-a9c4-18fcc625ef3e
[simshow(Array(printed_intensity[:, :, slice2]), set_one=false, cmap=:turbo) simshow(ones((size(target, 1), 5))) simshow(thresh4 .< Array(printed_intensity[:, :, slice2])) simshow(ones((size(target, 1), 5))) simshow(Array(target[:, :, slice2]))  simshow(ones((size(target, 1), 5))) simshow(Array(togoc(target)[:, :, slice2] .!= (thresh4 .< (printed_intensity[:, :, slice2]))))]

# ╔═╡ d7de379b-601f-4bd5-b26d-45d424811aa2
plot_intensity_histogram(target, printed_intensity, loss.thresholds)

# ╔═╡ 21a99ca0-6b97-4a83-9780-bb120f190afd
md"Different projection patterns: $(@bind angle PlutoUI.Slider(axes(patterns, 2), show_value=true, default=0.5))"

# ╔═╡ e14faca6-62d4-4551-aadc-5342cb6d2aed
simshow(Array(patterns[:,angle,:])[end:-1:begin, :]', cmap=:turbo, set_one=true)

# ╔═╡ 751eb357-4a3e-46f7-9fd8-69b2df1d1839
md"# 5. Propagate with Wave Optical Forward Mechanism
Here some boiler plate to use the ray optical patterns as input for the wave optical simulation.
"

# ╔═╡ 1350e0e8-7674-4653-85a4-43fa96e983a9
L_x = 100f-6

# ╔═╡ 73a104ac-5d6b-4d3b-81c8-3d9088fd77c1
λ = 405f-9 / 1.5f0

# ╔═╡ de8017f8-eb5a-423a-b8e5-f282519317e3
angle

# ╔═╡ 9e0c837d-bfdc-41bf-91da-ad9ab0766042
z = togoc(range(-L_x/2, L_x/2, size(target, 1)));

# ╔═╡ 618f4447-5bf7-45f1-a2f7-bbd535c7a637
patterns_reshape = select_region(permutedims(patterns, (3,1,2)), new_size=(size(patterns, 1) + 1, size(patterns,3), size(angles,1)));

# ╔═╡ d0a78a5d-d56d-46ba-8255-4f62826862e3
begin
	 AS = AngularSpectrum(copy(patterns_reshape[:, :, 1]) .+ 0im, z, λ, L_x, padding=false)
	 AS_abs2 = let target=target, AS=AS, langles=length(angles)
		function AS_abs2(x)
			abs2.(AS(x .+ 0im)) ./ langles
		end
	 end

	 fwd2 = let AS_abs2=AS_abs2, angles=angles
		 function fwd2(x)
			 SwissVAMyKnife.fwd_wave(x, AS_abs2, angles)
		end
	 end
end


# ╔═╡ a745d025-8315-417a-948c-1a0b05d9e19b
size(patterns_reshape)

# ╔═╡ e93b8a10-f38f-4b73-89b0-5990c981e883
# sqrt because Radon works in intensity space whereas Wave optical works with field strength
out_wave = fwd2(sqrt.(patterns_reshape));

# ╔═╡ d4f528b2-d007-40dd-ab16-7a6f3369a3cf
begin
	CUDA.@sync out_wave
	
	iou = round(calculate_IoU(permutedims(togoc(target), (3,1,2)), permutedims(out_wave, (1,3,2)) .> 0.75), digits=3)
end

# ╔═╡ f89de548-d445-4452-a80b-a714cf30ea42
md"The intersection over union is: $iou"

# ╔═╡ 670e51cb-2d2f-46f4-b8e7-69d6cb81ace5
@bind depth2 PlutoUI.Slider(1:size(patterns, 3), show_value=true, default=77)

# ╔═╡ 62cda1a6-75ba-4fd1-90ca-bba4362a297e
[simshow(Array(out_wave[depth2, :, :]'), cmap=:acton) simshow(Array(out_wave[depth2, :, :]') .> 0.75) simshow(Array(target[:, :, depth2])) simshow(Array(target[:, :, depth2]) .!= Array(out_wave[depth2, :, :]' .> 0.70))]

# ╔═╡ Cell order:
# ╟─7279f1d1-6423-44ae-b071-7c2847026ca9
# ╟─1f8a2f2b-e006-43e9-8fc8-2df2c72d8bc6
# ╠═022dde50-b50c-4198-b5d2-50cb95562a3f
# ╠═1f595779-173a-4892-bbfc-0435dcb46434
# ╠═d98ea233-c3dc-4e9e-b12d-15a7b687e8a8
# ╠═7de2dce2-0faa-4b5a-a23d-3929f03413cd
# ╠═d8163acd-be6c-4176-89a1-a4a7643441f6
# ╟─a5d61a5f-54c9-4560-9a68-3e2d70a62f14
# ╠═b77c01d2-4d8f-4c83-b933-f2744e48350b
# ╠═cf59efa4-8f0c-4796-a435-b89e3526b568
# ╠═c9c2b7b7-96bb-4756-b191-418e5adca43c
# ╠═2794d1bf-dea1-494a-917d-7d675d8bc8a3
# ╟─c1240545-a674-4e3c-8386-0b6cb4591fc6
# ╠═0be4ad80-2433-4f91-866a-b8cbb2a7458a
# ╠═246b8900-a4dd-4caf-8f34-9d868d6127b2
# ╟─02c93620-2cf6-4446-a280-d4a7934f0ecd
# ╠═971f7c2f-415f-4e73-aaea-acda355e2f5a
# ╟─cebf24f8-6efe-407c-b123-5289dbb60790
# ╠═d1fb09d5-cee0-4027-acaa-24ccabaf9cf0
# ╠═8328d500-2760-46db-9733-479763d5c08f
# ╠═93cebf3f-389b-4fde-8d13-af3e1165ad9c
# ╠═2ffbef8c-066d-49b6-8a29-1ef710099029
# ╟─c0529850-c323-4c72-bab1-ec414d3ad425
# ╠═792e4c29-7f0c-4d9b-833b-8341702e472e
# ╠═7c9f675d-3698-459d-93fa-d8e297a7ffa5
# ╟─e94fb1d2-06c3-4f45-91ed-411fe8bd033f
# ╟─49952e37-7f07-4719-b77a-971a521e155a
# ╟─da8a4c5f-7439-498b-b4c2-678987b45f2c
# ╟─0dac21c5-5576-499c-8cb8-e3f1e7c7f39c
# ╟─b1690362-a23d-4047-a9c4-18fcc625ef3e
# ╟─d7de379b-601f-4bd5-b26d-45d424811aa2
# ╟─21a99ca0-6b97-4a83-9780-bb120f190afd
# ╟─e14faca6-62d4-4551-aadc-5342cb6d2aed
# ╟─751eb357-4a3e-46f7-9fd8-69b2df1d1839
# ╠═5fe8e968-7017-4fde-97cc-6459e951811c
# ╠═1350e0e8-7674-4653-85a4-43fa96e983a9
# ╠═73a104ac-5d6b-4d3b-81c8-3d9088fd77c1
# ╠═de8017f8-eb5a-423a-b8e5-f282519317e3
# ╠═9e0c837d-bfdc-41bf-91da-ad9ab0766042
# ╠═618f4447-5bf7-45f1-a2f7-bbd535c7a637
# ╠═d0a78a5d-d56d-46ba-8255-4f62826862e3
# ╠═a745d025-8315-417a-948c-1a0b05d9e19b
# ╠═e93b8a10-f38f-4b73-89b0-5990c981e883
# ╟─d4f528b2-d007-40dd-ab16-7a6f3369a3cf
# ╟─f89de548-d445-4452-a80b-a714cf30ea42
# ╟─670e51cb-2d2f-46f4-b8e7-69d6cb81ace5
# ╟─62cda1a6-75ba-4fd1-90ca-bba4362a297e
