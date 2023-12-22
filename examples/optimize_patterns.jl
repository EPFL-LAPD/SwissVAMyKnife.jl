### A Pluto.jl notebook ###
# v0.19.36

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

# ╔═╡ 30aab62e-a0eb-11ee-06db-11eb60e1a051
begin
	using Pkg
	Pkg.activate(".")
	using Revise
end

# ╔═╡ 6f8d4329-9b45-410b-98a2-17cb22475060
using SwissVAMyKnife

# ╔═╡ f4772bab-4700-443f-9e86-7fb35b616551
using ImageShow, ImageIO, PlutoUI, IndexFunArrays, Optim

# ╔═╡ 2a21b5cd-9a73-4da8-b5b6-2a59aeefaf03
using Plots

# ╔═╡ c53419ef-e17c-4513-b621-dcf7a2996ea0
using CUDA

# ╔═╡ d1880cdd-22b0-41d2-8887-d78f82081259
md"# Check if your CUDA is functional"

# ╔═╡ ac4e92e0-732f-4f81-954b-5944bd23ed76
use_CUDA = Ref(true && CUDA.functional())

# ╔═╡ 23a00638-9061-4587-8f2d-1d65ba4a7492
togoc(x) = use_CUDA[] ? CuArray(x) : x

# ╔═╡ e5175713-ad1c-4ae0-8f67-2af2644637fa
md"Your CUDA card is **$(use_CUDA[] ? \"functional\" : \"not functional\")**.
Hence all computations will be performed multithreaded on your CPU only. 
CUDA GPUs offer much more performance often leading to >10x speedup
"

# ╔═╡ 990a9f1f-401c-4e4d-9c02-cf847f394e52
TableOfContents()

# ╔═╡ 9fb9061f-169f-48d0-9acf-120f533d235e
md"# Load Target Pixels for Print"

# ╔═╡ 22c4be0e-555b-4ed6-b4a0-1062e539f498
target = zeros(Float32, (200, 200, 1));

# ╔═╡ a6fdcf6f-24dc-44cd-996f-8eb8f5b651af
begin
	target .= box(target, (100, 100, 1))
	target .-= box(target, (70, 70, 1))
end;

# ╔═╡ f90cf4dd-6844-4515-a877-19b0d6a07f3f
simshow(target[:, :, 1])

# ╔═╡ 89882c40-2e93-4a15-a3c6-09ae9ade39eb
md"# Specify angles and thresholds for Optimization"

# ╔═╡ fd7a91df-5edd-479d-a810-3f30e38a2a21
angles = range(0f0, 2f0 * π, 100)	

# ╔═╡ b865677b-e54e-4468-80a2-8426f8eb0be3
md"Thresholds such as the default ones are often okayish.
If the distance is too large between the value, the optimization does not provide good results.
Also if the values are too large or too small, the optimization fails."

# ╔═╡ f4d93076-8728-46f3-bc8d-da42b9024e3c
thresholds = (0.65f0, 0.75f0)

# ╔═╡ ce3dae02-da47-45b1-bb00-4c4f0079499a
md"# Optimize patterns"

# ╔═╡ f5d4ab19-55fb-4408-9f97-a9e98278a222
@time patterns, printed_i, res = optimize_patterns(target, angles, iterations=20,
											 optimizer=LBFGS(),
											thresholds=thresholds,
											μ=1/100f0)

# ╔═╡ 1ecb3d90-4b50-4107-92c7-f2474675bbb8
res

# ╔═╡ 463b7a89-4df0-4442-8d6e-2b6b251fbc1b
plot([a.value for a in res.trace], title="Loss over iterations", xlabel="Iterations", ylabel="Loss", yscale=:log10, ylim=(1f-2, 1f4))

# ╔═╡ c443703d-cd2a-4e2a-bfd0-742b4fcb001f
SwissVAMyKnife.plot_intensity(target, printed_i, thresholds)

# ╔═╡ 0ac5bb39-f621-4c94-8e5b-5b148f7221a2
md"Threshold=$(@bind threshold Slider(0.0:0.01:1, show_value=true))"

# ╔═╡ 302e5892-dd11-4691-a4a8-2a51624c7aaf
md"depth in z=$(@bind z_i Slider(1:1:size(target, 3), show_value=true))"

# ╔═╡ 347a0bc9-e387-4dab-a74c-3ff5d5f1a33a
[simshow(printed_i[:,:, z_i], cmap=:turbo) simshow(printed_i[:, :, z_i] .> threshold) simshow(target[:, :, z_i])]

# ╔═╡ 2e8caf8a-d2d2-4326-b089-70ef18d74630
simshow(patterns[:, :, 1], cmap=:turbo)

# ╔═╡ fc5c2bf9-9aa1-4391-82f0-bf7c538d6985
sum(patterns) / length(patterns)

# ╔═╡ Cell order:
# ╠═30aab62e-a0eb-11ee-06db-11eb60e1a051
# ╠═6f8d4329-9b45-410b-98a2-17cb22475060
# ╠═f4772bab-4700-443f-9e86-7fb35b616551
# ╠═2a21b5cd-9a73-4da8-b5b6-2a59aeefaf03
# ╟─d1880cdd-22b0-41d2-8887-d78f82081259
# ╠═c53419ef-e17c-4513-b621-dcf7a2996ea0
# ╟─ac4e92e0-732f-4f81-954b-5944bd23ed76
# ╠═23a00638-9061-4587-8f2d-1d65ba4a7492
# ╟─e5175713-ad1c-4ae0-8f67-2af2644637fa
# ╟─990a9f1f-401c-4e4d-9c02-cf847f394e52
# ╟─9fb9061f-169f-48d0-9acf-120f533d235e
# ╠═22c4be0e-555b-4ed6-b4a0-1062e539f498
# ╠═a6fdcf6f-24dc-44cd-996f-8eb8f5b651af
# ╠═f90cf4dd-6844-4515-a877-19b0d6a07f3f
# ╟─89882c40-2e93-4a15-a3c6-09ae9ade39eb
# ╠═fd7a91df-5edd-479d-a810-3f30e38a2a21
# ╟─b865677b-e54e-4468-80a2-8426f8eb0be3
# ╠═f4d93076-8728-46f3-bc8d-da42b9024e3c
# ╟─ce3dae02-da47-45b1-bb00-4c4f0079499a
# ╠═f5d4ab19-55fb-4408-9f97-a9e98278a222
# ╠═1ecb3d90-4b50-4107-92c7-f2474675bbb8
# ╟─463b7a89-4df0-4442-8d6e-2b6b251fbc1b
# ╠═c443703d-cd2a-4e2a-bfd0-742b4fcb001f
# ╟─0ac5bb39-f621-4c94-8e5b-5b148f7221a2
# ╟─302e5892-dd11-4691-a4a8-2a51624c7aaf
# ╟─347a0bc9-e387-4dab-a74c-3ff5d5f1a33a
# ╠═2e8caf8a-d2d2-4326-b089-70ef18d74630
# ╠═fc5c2bf9-9aa1-4391-82f0-bf7c538d6985
