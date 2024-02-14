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

# ╔═╡ 874ffaea-c7ae-11ee-11c6-a7ddbc5bf852
begin
	using Pkg
	Pkg.activate(".")
	Pkg.instantiate()
	using Revise
end

# ╔═╡ 5d5d3113-e8d6-40a9-82d6-1d4f52e10d47
# this is our main package
using SwissVAMyKnife

# ╔═╡ 20587240-34ff-4e44-b481-5c631db90f9a
using ImageShow, ImageIO, PlutoUI, IndexFunArrays, Optim, RadonKA, FileIO, Colors, Plots

# ╔═╡ 50bb6567-4fea-4794-8a5d-2c519e065296
using CUDA

# ╔═╡ 86b8c061-2f0e-4e4d-9d5e-7694e152813a
using NDTools

# ╔═╡ bf0c719f-4fa7-4f58-b031-e535813acedf
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

# ╔═╡ ba186bc1-2d3d-4422-92b9-98e2657e2d47
md"# 0. Load packages
On the first run, Julia is going to install some packages automatically. So start this notebook and give it some minutes (5-10min) to install all packages. 
No worries, any future runs will be much faster to start!
"

# ╔═╡ 6d54655e-eacd-490c-adbf-57efd2172edc
TableOfContents()

# ╔═╡ 2480db2e-699e-41b8-ab4e-4fece6b16945
use_CUDA = Ref(true && CUDA.functional())

# ╔═╡ 8b8006df-bd5d-45b3-93a8-95739ab8284b
md" ## CUDA
CUDA accelerates the pattern generation easily by 5-20 times!
Otherwise most of the code will be multithreaded on your CPU but we strongly recommended the usage of CUDA for large scale 3D pattern generation.

Your CUDA is functional: **$(use_CUDA[])**
"

# ╔═╡ aad77bcc-9593-4a48-8518-107fc2832d75
var"@mytime" = use_CUDA[] ? CUDA.var"@time" : var"@time"

# ╔═╡ dfaeb34d-b86f-485d-a216-87f58d7175c7
togoc(x) = use_CUDA[] ? CuArray(x) : x

# ╔═╡ 8a5d2951-293c-42a1-91b6-a63ba4ef8a95


# ╔═╡ 0cae8621-9b65-4506-b604-72719293d64d
md"# 2. Simple optimization


## Loss Function
One simple optimization algorithm tries to distribute the intensity as following.
Object pixels should receive enough intensity to polymerize.
Void pixels should stay below this intensity threshold and should not polymerize.
Also, object pixels should not strongly over polymerize

Let's use the `LossThreshold` to achieve this:
"

# ╔═╡ ed6b2393-0ebb-4a24-b872-7424723e1f98
loss = LossThreshold(thresholds=(0.65, 0.75))

# ╔═╡ 7f812aa1-7fae-40fd-8601-a5d8c19a5e49
md"# 2. Target"

# ╔═╡ 1931aa0f-966e-4ca0-b6d6-2ff6995ff8cb
md"# 1. Load Benchy"

# ╔═╡ d708c3a1-6e59-4001-a93a-69442466a9d5
function load_benchy(sz, sz_file, path)
	target = zeros(Float32, sz)
	#@show size(load(joinpath(path, string("slice_", string(1, pad=3) ,".png"))))
	
	for i in 0:sz_file[1]-1
		target[:, :, 40 + i+1] .= select_region(Gray.(load(joinpath(path, string("boat_", string(i, pad=3) ,".png")))), new_size=(sz))
	end

	#target2 = zeros(Float32, sz)
	#WaveOpticsPropagation.set_center!(target2, target
	target2 = select_region(target, new_size=sz)
	target2 = permutedims(target2, (3,1,2))[end:-1:begin, :, :]
	return togoc(target2)
end

# ╔═╡ 70375fb2-a3f9-482d-b637-7b5867b1e2d0
target = permutedims(load_benchy((180, 180, 180), (100, 100, 100), "/home/felix/Downloads/benchy/files/output_100/"), (1,2,3));

# ╔═╡ 0594a1b3-cac0-4829-a801-78e2420022eb
md"z slide value $(@bind slice PlutoUI.Slider(axes(target, 3), show_value=true, default=0.5))"

# ╔═╡ cc95eb12-e948-44ba-8165-edf6886efb37
simshow(Array(target[:, :, slice]))

# ╔═╡ ceb4233f-efaa-4e78-922d-92acf6625ff3
md"# 3. Specify Wave optical properties"

# ╔═╡ 396e84a8-ce0c-4e84-b9ad-e79227f51bd4
n_resin = 1.5f0

# ╔═╡ 265171b5-6bf2-4466-9a16-c6a601cf99a7
angles = range(0, 1f0*π, 200)[1:end-1]

# ╔═╡ 8c368676-453d-4939-bf23-0880285c5817
L = 100f-6

# ╔═╡ a3539fe2-b282-4875-88dd-948c36d38bac
waveoptics = WaveOptics(
	z=togoc(range(-L/2, L/2, size(target,1))), 
	L=L, 
	λ=405f-9 / n_resin, 
	μ=nothing, 
	angles=angles,
	)

# ╔═╡ 24fbd950-59ba-457d-9451-094a051d1268
md"# Optimize
This takes around ~30s on a GPU to optimize

"

# ╔═╡ 0292b640-4a4f-4122-9fff-c3254f233485
optimizer = GradientBased(optimizer=Optim.LBFGS(), 		options=Optim.Options(iterations=20, store_trace=true))

# ╔═╡ ee025d44-4b1d-4aff-8b69-6b30b9bbcc2a
@mytime patterns, printed, res = optimize_patterns(togoc(target), waveoptics, optimizer, loss)

# ╔═╡ 01bb81c2-ab87-47c3-9e0c-6196f6c770f6
res

# ╔═╡ b83de72c-8e0d-453f-8324-9745cf768d6c
md"Threshold value=$(@bind thresh4 PlutoUI.Slider(0:0.01:1, show_value=true, default=0.7))"

# ╔═╡ 605ba331-a480-492f-98aa-402155d33ebf
md"z slider value $(@bind slice2 PlutoUI.Slider(axes(target, 3), show_value=true, default=0.5))"

# ╔═╡ 6e67afcc-597d-4c18-a351-e24c5b732f2d
[simshow(Array(printed[:, :, slice2]), set_one=false) simshow(ones((size(target, 1), 5))) simshow(thresh4 .< Array(printed[:, :, slice2])) simshow(ones((size(target, 1), 5))) simshow(target[:, :, slice2])]

# ╔═╡ 03270673-3fea-49d1-a551-2c08883d76f8
plot_intensity_histogram(target, printed, (0.65, 0.75))

# ╔═╡ c56e0332-7abf-4041-8360-ba4bd09f2082
md"angle $(@bind angle PlutoUI.Slider(axes(patterns, 2), show_value=true, default=0.5))"

# ╔═╡ 082dcba1-b2b9-433e-941f-1c96ab63a7a9
simshow(Array(patterns[:,angle,:]), cmap=:turbo, set_one=true)

# ╔═╡ Cell order:
# ╟─bf0c719f-4fa7-4f58-b031-e535813acedf
# ╠═ba186bc1-2d3d-4422-92b9-98e2657e2d47
# ╠═874ffaea-c7ae-11ee-11c6-a7ddbc5bf852
# ╠═5d5d3113-e8d6-40a9-82d6-1d4f52e10d47
# ╠═20587240-34ff-4e44-b481-5c631db90f9a
# ╟─6d54655e-eacd-490c-adbf-57efd2172edc
# ╟─8b8006df-bd5d-45b3-93a8-95739ab8284b
# ╠═50bb6567-4fea-4794-8a5d-2c519e065296
# ╠═2480db2e-699e-41b8-ab4e-4fece6b16945
# ╠═aad77bcc-9593-4a48-8518-107fc2832d75
# ╠═dfaeb34d-b86f-485d-a216-87f58d7175c7
# ╠═8a5d2951-293c-42a1-91b6-a63ba4ef8a95
# ╟─0cae8621-9b65-4506-b604-72719293d64d
# ╠═ed6b2393-0ebb-4a24-b872-7424723e1f98
# ╟─7f812aa1-7fae-40fd-8601-a5d8c19a5e49
# ╟─1931aa0f-966e-4ca0-b6d6-2ff6995ff8cb
# ╠═86b8c061-2f0e-4e4d-9d5e-7694e152813a
# ╠═d708c3a1-6e59-4001-a93a-69442466a9d5
# ╠═70375fb2-a3f9-482d-b637-7b5867b1e2d0
# ╟─0594a1b3-cac0-4829-a801-78e2420022eb
# ╠═cc95eb12-e948-44ba-8165-edf6886efb37
# ╟─ceb4233f-efaa-4e78-922d-92acf6625ff3
# ╠═a3539fe2-b282-4875-88dd-948c36d38bac
# ╠═396e84a8-ce0c-4e84-b9ad-e79227f51bd4
# ╠═265171b5-6bf2-4466-9a16-c6a601cf99a7
# ╠═8c368676-453d-4939-bf23-0880285c5817
# ╟─24fbd950-59ba-457d-9451-094a051d1268
# ╠═0292b640-4a4f-4122-9fff-c3254f233485
# ╠═ee025d44-4b1d-4aff-8b69-6b30b9bbcc2a
# ╠═01bb81c2-ab87-47c3-9e0c-6196f6c770f6
# ╟─b83de72c-8e0d-453f-8324-9745cf768d6c
# ╟─605ba331-a480-492f-98aa-402155d33ebf
# ╠═6e67afcc-597d-4c18-a351-e24c5b732f2d
# ╠═03270673-3fea-49d1-a551-2c08883d76f8
# ╠═c56e0332-7abf-4041-8360-ba4bd09f2082
# ╠═082dcba1-b2b9-433e-941f-1c96ab63a7a9
