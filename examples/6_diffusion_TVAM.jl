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

# ╔═╡ c982861a-5a10-4a82-8be8-fa5418d729da
begin
	using Pkg
	Pkg.activate(".")
	Pkg.instantiate()
	using Revise
end

# ╔═╡ 23a890a3-ae1b-4566-b050-677604a0c5e0
# this is our main package
using SwissVAMyKnife

# ╔═╡ ce68dbde-20c0-4bb6-bbca-9914986d9f63
using ImageShow, ImageIO, PlutoUI, IndexFunArrays, Optim, RadonKA, FileIO, Colors, Plots

# ╔═╡ b49b0a49-1317-4820-a1a3-51f3bc4cdef3
using CUDA

# ╔═╡ c4dfa825-b276-4b8c-811e-3ad3d6a9b129
using NDTools

# ╔═╡ 8d4ea5a5-3f98-4481-ac16-3fda7ccec3bb
md"""# Supplemental simulations

Note, in the publication we optimize a volume of 550x550x550 voxels which only runs on a large GPU (NVIDIA A100 with 80GB). Hence, here a smaller version which produces similar but not identical results because of a different discretization of the boat.

```bibtex
@article{Wechsler:24,
author = {Felix Wechsler and Carlo Gigli and Jorge Madrid-Wolff and Christophe Moser},
journal = {Opt. Express},
keywords = {3D printing; Computed tomography; Liquid crystal displays; Material properties; Ray tracing; Refractive index},
number = {8},
pages = {14705--14712},
publisher = {Optica Publishing Group},
title = {Wave optical model for tomographic volumetric additive manufacturing},
volume = {32},
month = {Apr},
year = {2024},
url = {https://opg.optica.org/oe/abstract.cfm?URI=oe-32-8-14705},
doi = {10.1364/OE.521322},
}
```
"""

# ╔═╡ e4d6ebe2-6f2d-43ff-9b6b-109808935aff
md"# 0. Load packages
On the first run, Julia is going to install some packages automatically. So start this notebook and give it some minutes (5-10min) to install all packages. 
No worries, any future runs will be much faster to start!
"

# ╔═╡ 1bac4230-2837-4146-a49b-248097fd0cf3
Pkg.resolve()

# ╔═╡ 015d4d1b-8921-41c6-9bc3-a4f43f3586aa
TableOfContents()

# ╔═╡ be1d52e3-c2a2-4cd0-8d10-ab088473bec3
use_CUDA = Ref(true && CUDA.functional())

# ╔═╡ a3077f1e-403b-421b-bf14-ab2307c2369b
md" ## CUDA
CUDA accelerates the pattern generation easily by 5-20 times!
Otherwise most of the code will be multithreaded on your CPU but we strongly recommended the usage of CUDA for large scale 3D pattern generation.

Your CUDA is functional: **$(use_CUDA[])**
"

# ╔═╡ d4395dd0-c784-45db-b80b-516840127f17
var"@mytime" = use_CUDA[] ? CUDA.var"@time" : var"@time"

# ╔═╡ 9960dd8c-c57f-4e1b-898e-110b3b5b5363
togoc(x) = use_CUDA[] ? CuArray(x) : x

# ╔═╡ fc67829c-53ed-4b92-a46d-c1216116e14d
md"# 1. Load Benchy"

# ╔═╡ 5b85bcc4-6bdf-4f09-969e-5c56cc236f81
target = load_example_target("3DBenchy_180");

# ╔═╡ 58963b11-5534-40b9-8101-977e6d21a0a4
md"z slide value $(@bind slice PlutoUI.Slider(axes(target, 3), show_value=true, default=77))"

# ╔═╡ bcd3affd-a880-4ba3-bcd2-52b45de97b1e
simshow(Array(target[:, :, slice]))

# ╔═╡ b52d634d-37dc-47c7-8f6c-5e5684e066a2
md"# 2. Specify Optimization Parameters"

# ╔═╡ 542d400f-4f63-45f7-bccf-a9370ee5663a
loss = LossThreshold(thresholds=(0.90, 0.97))

# ╔═╡ 2a3d1e2e-7603-4ee6-93e6-80faddfafc07
angles = range(0, 2π, 151)[begin:end-1]

# ╔═╡ 5e6ae68e-5dd7-4e26-84ad-f5ef7fd4b970
geometry = ParallelRayOptics(angles, nothing)

# ╔═╡ aeadc55a-6601-403a-b9cd-a3c6639cb15b
diffusion = Diffusion(25f-6, 1f-10, 40f0, 3, 5)

# ╔═╡ 43136316-d95d-47e5-bfbb-45443ae615cd
optimizer = GradientBased(optimizer=Optim.LBFGS(), options=Optim.Options(iterations=20, store_trace=true))

# ╔═╡ a7d4db4f-94b8-444a-846a-6e2ebfa5f368
md"# 3. Optimize with Diffusion and without"

# ╔═╡ 00004b83-427a-4704-bf13-c240ce4870fe
@mytime patterns, printed_intensity, optim_res = optimize_patterns(togoc(target),
	geometry, optimizer, loss)

# ╔═╡ e5a8c428-1dca-42df-87f7-3c429f29b61a
optim_res

# ╔═╡ 985bf921-4783-4f0f-99c8-c08cb772e2f8
@mytime patterns2, printed_intensity2, optim_res2 = optimize_patterns(togoc(target),
	geometry, diffusion, optimizer, loss)

# ╔═╡ 5d1b390d-175a-47f5-a8e9-ca8212eca646
optim_res2

# ╔═╡ 0cc85f47-068f-49f9-9701-0fcd9334f178
md"# 4. Inspect"

# ╔═╡ b48ff258-5163-42b5-9168-0ccab5ab10dc
md"Choose threshold for image: $(@bind thresh4 PlutoUI.Slider(0:0.01:1, show_value=true, default=0.7))"

# ╔═╡ a5c3f411-42f8-401b-b0b2-380dc9b57bfa
md"z slice $(@bind slice2 PlutoUI.Slider(axes(target, 3), show_value=true, default=77))

Intensity distribution ----------- after threshold -------------------- target -------------------------------difference
"

# ╔═╡ 0d665c2b-7e6c-4af4-afeb-cee2db582a02
[simshow(Array(printed_intensity[:, :, slice2]), set_one=false, cmap=:turbo) simshow(ones((size(target, 1), 5))) simshow(thresh4 .< Array(printed_intensity[:, :, slice2])) simshow(ones((size(target, 1), 5))) simshow(Array(target[:, :, slice2]))  simshow(ones((size(target, 1), 5))) simshow(Array(togoc(target)[:, :, slice2] .!= (thresh4 .< (printed_intensity[:, :, slice2]))))]

# ╔═╡ 5f8ae887-5d69-41bc-9601-8894f8557ffd
[simshow(Array(printed_intensity2[:, :, slice2]), set_one=false, cmap=:turbo) simshow(ones((size(target, 1), 5))) simshow(thresh4 .< Array(printed_intensity2[:, :, slice2])) simshow(ones((size(target, 1), 5))) simshow(Array(target[:, :, slice2]))  simshow(ones((size(target, 1), 5))) simshow(Array(togoc(target)[:, :, slice2] .!= (thresh4 .< (printed_intensity2[:, :, slice2]))))]

# ╔═╡ 44b67acf-b31f-4cd1-b675-475a81e8ed85
plot_intensity_histogram(target, printed_intensity, loss.thresholds)

# ╔═╡ b9636f03-e822-4b7f-8639-ae8ec3bd1d78
plot_intensity_histogram(target, printed_intensity2, loss.thresholds)

# ╔═╡ d1ee51c2-e30e-4a4e-976d-7322a879d35e
md"Different projection patterns: $(@bind angle PlutoUI.Slider(axes(patterns, 2), show_value=true, default=0.5))"

# ╔═╡ 9c05b82e-787e-4909-9fcf-4257bd8c9ddc
simshow(Array(patterns[:,angle,:])[end:-1:begin, :]', cmap=:turbo, set_one=true)

# ╔═╡ 58187e08-9608-415d-ba8a-c4a952bfa2a7
simshow(Array(patterns2[:,angle,:])[end:-1:begin, :]', cmap=:turbo, set_one=true)

# ╔═╡ 9a9051e1-03bf-4e76-85dc-8b07d35c6ca1
md"# 5. Compare to non Diffusion
Propagate the non diffusion patterns with the diffusion model
"

# ╔═╡ c1bcdafe-e03a-4403-9911-b6084c4cd539
fwd_diffusion, _ = SwissVAMyKnife._prepare_ray_forward(togoc(target), geometry, diffusion)

# ╔═╡ 48fb1a90-d410-40e2-8d5f-9b5714b611f0
printend_intensity_without_diff = fwd_diffusion(patterns ./ diffusion.N_rotations);

# ╔═╡ 37320779-2cfd-44e5-a258-6cec54ba1a38
# ╠═╡ disabled = true
#=╠═╡
printend_intensity_without_diff = fwd_diffusion(patterns2);
  ╠═╡ =#

# ╔═╡ 2fe8b777-1867-44b3-905f-6f9079d38f1a
md"Choose threshold for image: $(@bind thresh5 PlutoUI.Slider(0:0.01:1, show_value=true, default=0.7))"

# ╔═╡ 966b4aa1-ddc2-4fd4-98d0-f571ca67c8d1
md"z slice $(@bind slice3 PlutoUI.Slider(axes(target, 3), show_value=true, default=77))

Intensity distribution ----------- after threshold -------------------- target -------------------------------difference
"

# ╔═╡ 6da2ab24-8817-4873-8006-bee30ddf4b2e
[simshow(Array(printend_intensity_without_diff[:, :, slice2]), set_one=false, cmap=:turbo) simshow(ones((size(target, 1), 5))) simshow(thresh5 .< Array(printend_intensity_without_diff[:, :, slice3])) simshow(ones((size(target, 1), 5))) simshow(Array(target[:, :, slice3]))  simshow(ones((size(target, 1), 5))) simshow(Array(togoc(target)[:, :, slice3] .!= (thresh5 .< (printend_intensity_without_diff[:, :, slice3]))))]

# ╔═╡ ba5ff89a-f569-4a70-9205-7dbde326df4b
plot_intensity_histogram(target, printend_intensity_without_diff, loss.thresholds)

# ╔═╡ db4c7a5c-1395-4566-912d-5150503ce1a7
Δt = 40 / 3 / 3

# ╔═╡ Cell order:
# ╟─8d4ea5a5-3f98-4481-ac16-3fda7ccec3bb
# ╟─e4d6ebe2-6f2d-43ff-9b6b-109808935aff
# ╠═c982861a-5a10-4a82-8be8-fa5418d729da
# ╠═1bac4230-2837-4146-a49b-248097fd0cf3
# ╠═23a890a3-ae1b-4566-b050-677604a0c5e0
# ╠═ce68dbde-20c0-4bb6-bbca-9914986d9f63
# ╠═015d4d1b-8921-41c6-9bc3-a4f43f3586aa
# ╟─a3077f1e-403b-421b-bf14-ab2307c2369b
# ╠═b49b0a49-1317-4820-a1a3-51f3bc4cdef3
# ╠═be1d52e3-c2a2-4cd0-8d10-ab088473bec3
# ╠═d4395dd0-c784-45db-b80b-516840127f17
# ╠═9960dd8c-c57f-4e1b-898e-110b3b5b5363
# ╟─fc67829c-53ed-4b92-a46d-c1216116e14d
# ╠═c4dfa825-b276-4b8c-811e-3ad3d6a9b129
# ╠═5b85bcc4-6bdf-4f09-969e-5c56cc236f81
# ╟─58963b11-5534-40b9-8101-977e6d21a0a4
# ╠═bcd3affd-a880-4ba3-bcd2-52b45de97b1e
# ╟─b52d634d-37dc-47c7-8f6c-5e5684e066a2
# ╠═542d400f-4f63-45f7-bccf-a9370ee5663a
# ╠═2a3d1e2e-7603-4ee6-93e6-80faddfafc07
# ╠═5e6ae68e-5dd7-4e26-84ad-f5ef7fd4b970
# ╠═aeadc55a-6601-403a-b9cd-a3c6639cb15b
# ╠═43136316-d95d-47e5-bfbb-45443ae615cd
# ╟─a7d4db4f-94b8-444a-846a-6e2ebfa5f368
# ╠═00004b83-427a-4704-bf13-c240ce4870fe
# ╠═e5a8c428-1dca-42df-87f7-3c429f29b61a
# ╠═985bf921-4783-4f0f-99c8-c08cb772e2f8
# ╠═5d1b390d-175a-47f5-a8e9-ca8212eca646
# ╠═0cc85f47-068f-49f9-9701-0fcd9334f178
# ╟─b48ff258-5163-42b5-9168-0ccab5ab10dc
# ╟─a5c3f411-42f8-401b-b0b2-380dc9b57bfa
# ╠═0d665c2b-7e6c-4af4-afeb-cee2db582a02
# ╟─5f8ae887-5d69-41bc-9601-8894f8557ffd
# ╠═44b67acf-b31f-4cd1-b675-475a81e8ed85
# ╠═b9636f03-e822-4b7f-8639-ae8ec3bd1d78
# ╠═d1ee51c2-e30e-4a4e-976d-7322a879d35e
# ╠═9c05b82e-787e-4909-9fcf-4257bd8c9ddc
# ╠═58187e08-9608-415d-ba8a-c4a952bfa2a7
# ╟─9a9051e1-03bf-4e76-85dc-8b07d35c6ca1
# ╠═c1bcdafe-e03a-4403-9911-b6084c4cd539
# ╠═48fb1a90-d410-40e2-8d5f-9b5714b611f0
# ╠═37320779-2cfd-44e5-a258-6cec54ba1a38
# ╟─2fe8b777-1867-44b3-905f-6f9079d38f1a
# ╟─966b4aa1-ddc2-4fd4-98d0-f571ca67c8d1
# ╟─6da2ab24-8817-4873-8006-bee30ddf4b2e
# ╠═ba5ff89a-f569-4a70-9205-7dbde326df4b
# ╠═db4c7a5c-1395-4566-912d-5150503ce1a7
