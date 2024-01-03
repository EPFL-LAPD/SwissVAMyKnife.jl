### A Pluto.jl notebook ###
# v0.19.27

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

# ╔═╡ ebd218dd-6e34-4d49-87bf-bb552c89db97
using RadonKA

# ╔═╡ 2a21b5cd-9a73-4da8-b5b6-2a59aeefaf03
using Plots

# ╔═╡ c53419ef-e17c-4513-b621-dcf7a2996ea0
using CUDA

# ╔═╡ 535ab140-e779-448a-a362-2268dd6062cb
using Zygote

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

# ╔═╡ a6fdcf6f-24dc-44cd-996f-8eb8f5b651af
begin
	target = zeros(Float32, (50, 50, 50));
	target .= box(target, (25, 25, 25))
	#target .-= box(target, (50, 70, 5))
	target = togoc(target)
end;

# ╔═╡ f90cf4dd-6844-4515-a877-19b0d6a07f3f
simshow(Array(target[:, :, 25]))

# ╔═╡ 89882c40-2e93-4a15-a3c6-09ae9ade39eb
md"# Specify angles and thresholds for Optimization"

# ╔═╡ 117da88b-4981-4be5-b79e-93a2938ff2e1
200 / 2 * 2π

# ╔═╡ fd7a91df-5edd-479d-a810-3f30e38a2a21
angles = range(0f0, 2f0 * π, 120)

# ╔═╡ b865677b-e54e-4468-80a2-8426f8eb0be3
md"Thresholds such as the default ones are often okayish.
If the distance is too large between the value, the optimization does not provide good results.
Also if the values are too large or too small, the optimization fails."

# ╔═╡ f4d93076-8728-46f3-bc8d-da42b9024e3c
thresholds = (0.65f0, 0.75f0)

# ╔═╡ ce3dae02-da47-45b1-bb00-4c4f0079499a
md"# Optimize patterns"

# ╔═╡ cc45b0d2-efd2-467f-a6f7-22f8ef901735
@time patterns, printed_i, res = optimize_patterns(target, angles, iterations=100,
											method=:radon_iterative,
											thresholds=thresholds,
											μ=nothing)

# ╔═╡ 513abaa9-bcf9-4e6c-8fec-96091755d2c5


# ╔═╡ 5d110682-623d-46cc-9657-17ecc47c79bf
L = 400f-6

# ╔═╡ 0d0b3478-fc0d-4f1b-b956-ff220bd6539a
λ = 405f-9 ./ 1.5

# ╔═╡ ad64cd2d-158f-41c1-af8d-b1b713a8b502
z = togoc(range(0, L, size(target, 1)));

# ╔═╡ 08d04781-b4c7-41da-8e2f-107d5719239a
@time patterns_wave, printed_i_wave, res_wave = optimize_patterns(target, angles; iterations=20, method=:wave, thresholds=thresholds, μ=nothing, L, λ, z)

# ╔═╡ 52972b48-7693-4525-abae-231ae450bc2d


# ╔═╡ adfb4187-6b45-47f4-a591-0d4952a2d96a
Zygote.refresh()

# ╔═╡ 92dbd8f4-cff0-46b6-bb3a-9f5c26c79277
Revise.retry()

# ╔═╡ f4ea9f0e-7e05-44bf-b9b3-9eecb7f0212f
res

# ╔═╡ 1f4af66f-fee2-4c0c-8259-645016080379
Revise.retry()

# ╔═╡ 1ecb3d90-4b50-4107-92c7-f2474675bbb8
res

# ╔═╡ 463b7a89-4df0-4442-8d6e-2b6b251fbc1b
# ╠═╡ disabled = true
#=╠═╡
plot([a.value for a in res.trace], title="Loss over iterations", xlabel="Iterations", ylabel="Loss", yscale=:log10)
  ╠═╡ =#

# ╔═╡ 19c2d9dd-ec68-420e-9d06-2c66ddbe0125
plot([a for a in res], title="Loss over iterations", xlabel="Iterations", ylabel="Loss", yscale=:log10)

# ╔═╡ c443703d-cd2a-4e2a-bfd0-742b4fcb001f
SwissVAMyKnife.plot_intensity(Array(target), Array(printed_i), thresholds)

# ╔═╡ 0ac5bb39-f621-4c94-8e5b-5b148f7221a2
md"Threshold=$(@bind threshold Slider(0.0:0.01:1, show_value=true))"

# ╔═╡ 302e5892-dd11-4691-a4a8-2a51624c7aaf
md"depth in z=$(@bind z_i Slider(1:1:size(target, 3), show_value=true))"

# ╔═╡ 347a0bc9-e387-4dab-a74c-3ff5d5f1a33a
[simshow(Array(printed_i[:,:, z_i]), cmap=:turbo) simshow(Array(printed_i[:, :, z_i]) .> threshold) simshow(Array(target[:, :, z_i]))]

# ╔═╡ b86d65cd-b52f-405d-874a-7a8f6fce732e
simshow(Array(printed_i[:, :, 4]))

# ╔═╡ 85aa66e4-fc61-4e81-9d1c-8bce66ff51c6
extrema(printed_i)

# ╔═╡ 2e8caf8a-d2d2-4326-b089-70ef18d74630
simshow(Array(patterns[:, :, 1]), cmap=:turbo)

# ╔═╡ fc5c2bf9-9aa1-4391-82f0-bf7c538d6985
sum(patterns) / length(patterns)

# ╔═╡ 95f21e79-4c88-4641-9b79-fb0568b72374
simshow(max.(0, radon(RadonKA.filtered_backprojection(radon(target, angles), angles), angles)[:, :, 10]) |> Array, cmap=:turbo)

# ╔═╡ 618a4ad6-e626-4c67-8d32-098c69e0a0e7
simshow(max.(0, radon(RadonKA.filtered_backprojection(radon(target, angles), angles), angles)[:, :, 10]) |> Array, cmap=:turbo)

# ╔═╡ 03fd16cd-ec92-4426-8727-181dd6deb6bd


# ╔═╡ e4d45c59-a999-4c56-ab85-620538648fb5
size(target)

# ╔═╡ 5ae33322-1f38-4f9d-aa86-cfa8c2aad891
patterns_filtered = max.(0, radon(RadonKA.filtered_backprojection(radon(target, angles), angles), angles)) ./ 7.62f0;

# ╔═╡ be9dcd32-995a-445b-9dbe-30e44a22f06d
iradon(patterns_filtered, angles) |> extrema

# ╔═╡ 1b6ba316-9718-4c6f-8ed9-deaae701d187
extrema(patterns_filtered)

# ╔═╡ c097fb21-7971-41c6-b8d6-a730f5ddc1fb
size(angles)

# ╔═╡ Cell order:
# ╠═30aab62e-a0eb-11ee-06db-11eb60e1a051
# ╠═6f8d4329-9b45-410b-98a2-17cb22475060
# ╠═f4772bab-4700-443f-9e86-7fb35b616551
# ╠═ebd218dd-6e34-4d49-87bf-bb552c89db97
# ╠═2a21b5cd-9a73-4da8-b5b6-2a59aeefaf03
# ╟─d1880cdd-22b0-41d2-8887-d78f82081259
# ╠═c53419ef-e17c-4513-b621-dcf7a2996ea0
# ╠═ac4e92e0-732f-4f81-954b-5944bd23ed76
# ╠═23a00638-9061-4587-8f2d-1d65ba4a7492
# ╟─e5175713-ad1c-4ae0-8f67-2af2644637fa
# ╟─990a9f1f-401c-4e4d-9c02-cf847f394e52
# ╟─9fb9061f-169f-48d0-9acf-120f533d235e
# ╠═a6fdcf6f-24dc-44cd-996f-8eb8f5b651af
# ╟─f90cf4dd-6844-4515-a877-19b0d6a07f3f
# ╟─89882c40-2e93-4a15-a3c6-09ae9ade39eb
# ╠═117da88b-4981-4be5-b79e-93a2938ff2e1
# ╠═fd7a91df-5edd-479d-a810-3f30e38a2a21
# ╟─b865677b-e54e-4468-80a2-8426f8eb0be3
# ╠═f4d93076-8728-46f3-bc8d-da42b9024e3c
# ╟─ce3dae02-da47-45b1-bb00-4c4f0079499a
# ╠═cc45b0d2-efd2-467f-a6f7-22f8ef901735
# ╠═513abaa9-bcf9-4e6c-8fec-96091755d2c5
# ╠═5d110682-623d-46cc-9657-17ecc47c79bf
# ╠═0d0b3478-fc0d-4f1b-b956-ff220bd6539a
# ╠═ad64cd2d-158f-41c1-af8d-b1b713a8b502
# ╠═08d04781-b4c7-41da-8e2f-107d5719239a
# ╠═52972b48-7693-4525-abae-231ae450bc2d
# ╠═adfb4187-6b45-47f4-a591-0d4952a2d96a
# ╠═535ab140-e779-448a-a362-2268dd6062cb
# ╠═92dbd8f4-cff0-46b6-bb3a-9f5c26c79277
# ╠═f4ea9f0e-7e05-44bf-b9b3-9eecb7f0212f
# ╠═1f4af66f-fee2-4c0c-8259-645016080379
# ╠═1ecb3d90-4b50-4107-92c7-f2474675bbb8
# ╟─463b7a89-4df0-4442-8d6e-2b6b251fbc1b
# ╠═19c2d9dd-ec68-420e-9d06-2c66ddbe0125
# ╠═c443703d-cd2a-4e2a-bfd0-742b4fcb001f
# ╟─0ac5bb39-f621-4c94-8e5b-5b148f7221a2
# ╟─302e5892-dd11-4691-a4a8-2a51624c7aaf
# ╟─347a0bc9-e387-4dab-a74c-3ff5d5f1a33a
# ╠═b86d65cd-b52f-405d-874a-7a8f6fce732e
# ╠═85aa66e4-fc61-4e81-9d1c-8bce66ff51c6
# ╠═2e8caf8a-d2d2-4326-b089-70ef18d74630
# ╠═fc5c2bf9-9aa1-4391-82f0-bf7c538d6985
# ╠═95f21e79-4c88-4641-9b79-fb0568b72374
# ╠═618a4ad6-e626-4c67-8d32-098c69e0a0e7
# ╠═03fd16cd-ec92-4426-8727-181dd6deb6bd
# ╠═e4d45c59-a999-4c56-ab85-620538648fb5
# ╠═be9dcd32-995a-445b-9dbe-30e44a22f06d
# ╠═5ae33322-1f38-4f9d-aa86-cfa8c2aad891
# ╠═1b6ba316-9718-4c6f-8ed9-deaae701d187
# ╠═c097fb21-7971-41c6-b8d6-a730f5ddc1fb
