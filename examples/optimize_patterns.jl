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
using SwissVAMyKnife, WaveOpticsPropagation

# ╔═╡ f4772bab-4700-443f-9e86-7fb35b616551
using ImageShow, ImageIO, PlutoUI, IndexFunArrays, Optim

# ╔═╡ ebd218dd-6e34-4d49-87bf-bb552c89db97
using RadonKA, FileIO

# ╔═╡ 2a21b5cd-9a73-4da8-b5b6-2a59aeefaf03
using Plots, NDTools

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

# ╔═╡ a6fdcf6f-24dc-44cd-996f-8eb8f5b651af
# ╠═╡ disabled = true
#=╠═╡
begin
	target = zeros(Float32, (50, 50, 50));
	target .= box(target, (25, 25, 25))
	#target .-= box(target, (50, 70, 5))
	target = togoc(target)
end;
  ╠═╡ =#

# ╔═╡ 2a605106-9eb3-426e-91fa-7900c24776fe
function load_benchy(sz, sz_file, path)
	target = zeros(Float32, sz_file)

	for i in 1:sz_file[1]
		target[i, :, :] .= Gray.(load(joinpath(path, string("boat_", string(i, pad=3) ,".png"))))
	end

	target2 = zeros(Float32, sz)
	WaveOpticsPropagation.set_center!(target2, target)
	return togoc(target2)
end

# ╔═╡ edd4580e-4cf2-4623-a127-007547b88a03
target2 = load_benchy((252, 252, 252), (200, 200, 200), "/home/felix/Downloads/benchy/files/output/");

# ╔═╡ f90cf4dd-6844-4515-a877-19b0d6a07f3f
simshow(Array(target[:, :, 25]))

# ╔═╡ 29461391-9128-4411-b33e-97f3ce6625a7
@bind izzz Slider(1:size(target, 3), show_value=true)

# ╔═╡ 5b45efde-da5a-45af-968b-208d04e43517
simshow(Array(target[:, :, izzz]))

# ╔═╡ 8f12a113-6772-4567-95bc-d43f72d303ca


# ╔═╡ 89882c40-2e93-4a15-a3c6-09ae9ade39eb
md"# Specify angles and thresholds for Optimization"

# ╔═╡ 117da88b-4981-4be5-b79e-93a2938ff2e1
200 / 2 * 2π

# ╔═╡ fd7a91df-5edd-479d-a810-3f30e38a2a21
angles = range(0f0, 2f0 * π, 50)

# ╔═╡ b865677b-e54e-4468-80a2-8426f8eb0be3
md"Thresholds such as the default ones are often okayish.
If the distance is too large between the value, the optimization does not provide good results.
Also if the values are too large or too small, the optimization fails."

# ╔═╡ f4d93076-8728-46f3-bc8d-da42b9024e3c
thresholds = (0.65f0, 0.75f0)

# ╔═╡ ce3dae02-da47-45b1-bb00-4c4f0079499a
md"# Optimize patterns"

# ╔═╡ cc45b0d2-efd2-467f-a6f7-22f8ef901735
# ╠═╡ show_logs = false
@time patterns, printed_i, res = optimize_patterns(target, angles, iterations=10,
											method=:radon_iterative,
											thresholds=thresholds,
											μ=nothing)

# ╔═╡ 6c3a476a-087f-4988-93d3-65ebfdadc26b
plot(res, title="Loss over iterations", xlabel="Iterations", ylabel="Loss", yscale=:log10)

# ╔═╡ 2f7d32c6-912d-43fd-bfb8-75713f0dd0ef
md"Threshold=$(@bind threshold Slider(0.0:0.01:1, show_value=true))"

# ╔═╡ c4865085-0b1f-42be-ad88-9b8d694278e5
md"depth in z=$(@bind z_i Slider(1:1:size(target, 3), show_value=true))"

# ╔═╡ 6d78294c-4c92-457a-91de-4bd8a881de71
[simshow(Array(printed_i[:,:, z_i]), cmap=:turbo) simshow(Array(printed_i[:, :, z_i]) .> threshold) simshow(Array(target[:, :, z_i]))]

# ╔═╡ b5cdb60e-6294-4caf-b8b3-9bef8db0d137
md"angle=$(@bind i_angle Slider(1:1:size(patterns, 2), show_value=true))"

# ╔═╡ 9a81a25c-bf91-4b1e-9713-ae65c7db2d16
simshow(Array(patterns[:, i_angle, :])')

# ╔═╡ 960c8db1-3f96-48c1-a1cc-f6f75aa98c15


# ╔═╡ 238931ba-9ce9-496a-b424-a9f1682b6d99
md"# Wave Optics"

# ╔═╡ 5d110682-623d-46cc-9657-17ecc47c79bf
L = 400f-6

# ╔═╡ 0d0b3478-fc0d-4f1b-b956-ff220bd6539a
λ = 405f-9 ./ 1.5

# ╔═╡ ad64cd2d-158f-41c1-af8d-b1b713a8b502
z = togoc(range(0, L, size(target, 1)));

# ╔═╡ 08d04781-b4c7-41da-8e2f-107d5719239a
@time patterns_wave, printed_i_wave, res_wave = optimize_patterns(target, angles; iterations=50, method=:wave, optimizer=LBFGS(), thresholds=thresholds, μ=nothing, L, λ, z)

# ╔═╡ 53c2c266-e0ae-4fd8-9f90-e1cf700cdba8
res_wave

# ╔═╡ 575ba3c2-1f80-4193-98ad-fdf0e73341d8
md"angle=$(@bind i_angle2 Slider(1:1:size(patterns, 2), show_value=true))"

# ╔═╡ 8ce5a252-9fac-47b3-a84e-351ba723c243
simshow(Array(patterns_wave[:, i_angle2, :]), γ=1)

# ╔═╡ 105db6be-382f-4521-837c-bc7d39a3dce9
md"Threshold=$(@bind threshold3 Slider(0.0:0.01:1, show_value=true))"

# ╔═╡ afddd44a-3b23-48d1-aeb6-a7b9a05e303b
md"depth in z=$(@bind z_i3 Slider(1:1:size(target, 3), show_value=true))"

# ╔═╡ 71247ebd-9be4-4328-a1e6-a2fa82d3ae2a
[simshow(Array(printed_i_wave[:,:, z_i3]), cmap=:turbo, set_one=false) simshow(Array(printed_i_wave[:, :, z_i3]) .> threshold3) simshow(Array(target[:, :, z_i3]))]

# ╔═╡ 463b7a89-4df0-4442-8d6e-2b6b251fbc1b
# ╠═╡ disabled = true
#=╠═╡
plot([a.value for a in res.trace], title="Loss over iterations", xlabel="Iterations", ylabel="Loss", yscale=:log10)
  ╠═╡ =#

# ╔═╡ 19c2d9dd-ec68-420e-9d06-2c66ddbe0125
plot([a.value for a in res_wave.trace], title="Loss over iterations", xlabel="Iterations", ylabel="Loss", yscale=:log10)

# ╔═╡ c443703d-cd2a-4e2a-bfd0-742b4fcb001f
#=╠═╡
SwissVAMyKnife.plot_intensity(Array(target), Array(printed_i), thresholds)
  ╠═╡ =#

# ╔═╡ 347a0bc9-e387-4dab-a74c-3ff5d5f1a33a
#=╠═╡
[simshow(Array(printed_i[:,:, z_i]), cmap=:turbo) simshow(Array(printed_i[:, :, z_i]) .> threshold) simshow(Array(target[:, :, z_i]))]
  ╠═╡ =#

# ╔═╡ b86d65cd-b52f-405d-874a-7a8f6fce732e
#=╠═╡
simshow(Array(printed_i[:, :, 4]))
  ╠═╡ =#

# ╔═╡ 85aa66e4-fc61-4e81-9d1c-8bce66ff51c6
#=╠═╡
extrema(printed_i)
  ╠═╡ =#

# ╔═╡ 2e8caf8a-d2d2-4326-b089-70ef18d74630
#=╠═╡
simshow(Array(patterns[:, 4, :]), cmap=:turbo)
  ╠═╡ =#

# ╔═╡ fc5c2bf9-9aa1-4391-82f0-bf7c538d6985
#=╠═╡
sum(patterns) / length(patterns)
  ╠═╡ =#

# ╔═╡ 4a9d17be-f8f2-4bab-aa66-ee71077acaec
begin
	         AS, _ = Angular_Spectrum(patterns_wave[:, :, 1] .+ 0im, z, λ, L, padding=false)    
	         AS_abs2 = let target=target, AS=AS
	                 function AS_abs2(x)
	                     abs2.(AS(abs2.(x) .+ 0im)[1])
	                 end 
	         end 
	 
	         fwd2 = let AS_abs2=AS_abs2, angles=angles
	             function fwd2(x)
	                 SwissVAMyKnife.fwd_wave(x, AS_abs2, angles)
	             end 
	         end
end

# ╔═╡ ddd1f894-dde5-46ab-8559-ec5aee176bb6
sum(abs2, AS(sqrt.(patterns_wave[:, :, 10]))[1])

# ╔═╡ 6eb8d6c6-029a-46cd-9ab3-bfd9a3133c69
sum(abs2, patterns_wave[:, :, 10])

# ╔═╡ e0a09d91-6316-4137-aa1d-300ab5b8b764
fwd2(sqrt.(patterns_wave)) |> sum

# ╔═╡ 33e9f7ac-0a1b-424d-999b-31538c40b77e
size(patterns_wave)

# ╔═╡ 571a12c5-cea8-44d6-b5d0-25404f55d3c9
sum(abs2, patterns_wave)

# ╔═╡ db29a6a5-2635-4fbe-8111-8c369fafa6e1
begin
	target = togoc(zeros(Float32, (100, 100, 100)))
	
	for i in 1:80
		target[begin+10+i÷2:end-10-i÷2,begin+10+i÷3:end-10-i÷3, 10 + i] .= 1
	end
end

# ╔═╡ 66e90a47-981e-4d19-81aa-90e262f64158
# ╠═╡ disabled = true
#=╠═╡
target = select_region(target2, new_size=(100, 100, 100))
  ╠═╡ =#

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
# ╠═f90cf4dd-6844-4515-a877-19b0d6a07f3f
# ╠═2a605106-9eb3-426e-91fa-7900c24776fe
# ╠═edd4580e-4cf2-4623-a127-007547b88a03
# ╠═66e90a47-981e-4d19-81aa-90e262f64158
# ╟─29461391-9128-4411-b33e-97f3ce6625a7
# ╟─5b45efde-da5a-45af-968b-208d04e43517
# ╠═db29a6a5-2635-4fbe-8111-8c369fafa6e1
# ╠═8f12a113-6772-4567-95bc-d43f72d303ca
# ╟─89882c40-2e93-4a15-a3c6-09ae9ade39eb
# ╠═117da88b-4981-4be5-b79e-93a2938ff2e1
# ╠═fd7a91df-5edd-479d-a810-3f30e38a2a21
# ╟─b865677b-e54e-4468-80a2-8426f8eb0be3
# ╠═f4d93076-8728-46f3-bc8d-da42b9024e3c
# ╟─ce3dae02-da47-45b1-bb00-4c4f0079499a
# ╠═cc45b0d2-efd2-467f-a6f7-22f8ef901735
# ╠═6c3a476a-087f-4988-93d3-65ebfdadc26b
# ╠═2f7d32c6-912d-43fd-bfb8-75713f0dd0ef
# ╠═c4865085-0b1f-42be-ad88-9b8d694278e5
# ╠═6d78294c-4c92-457a-91de-4bd8a881de71
# ╠═b5cdb60e-6294-4caf-b8b3-9bef8db0d137
# ╠═9a81a25c-bf91-4b1e-9713-ae65c7db2d16
# ╠═960c8db1-3f96-48c1-a1cc-f6f75aa98c15
# ╟─238931ba-9ce9-496a-b424-a9f1682b6d99
# ╠═5d110682-623d-46cc-9657-17ecc47c79bf
# ╠═0d0b3478-fc0d-4f1b-b956-ff220bd6539a
# ╠═ad64cd2d-158f-41c1-af8d-b1b713a8b502
# ╠═08d04781-b4c7-41da-8e2f-107d5719239a
# ╠═53c2c266-e0ae-4fd8-9f90-e1cf700cdba8
# ╟─575ba3c2-1f80-4193-98ad-fdf0e73341d8
# ╠═8ce5a252-9fac-47b3-a84e-351ba723c243
# ╟─105db6be-382f-4521-837c-bc7d39a3dce9
# ╟─afddd44a-3b23-48d1-aeb6-a7b9a05e303b
# ╟─71247ebd-9be4-4328-a1e6-a2fa82d3ae2a
# ╟─463b7a89-4df0-4442-8d6e-2b6b251fbc1b
# ╠═19c2d9dd-ec68-420e-9d06-2c66ddbe0125
# ╠═c443703d-cd2a-4e2a-bfd0-742b4fcb001f
# ╟─347a0bc9-e387-4dab-a74c-3ff5d5f1a33a
# ╠═b86d65cd-b52f-405d-874a-7a8f6fce732e
# ╠═85aa66e4-fc61-4e81-9d1c-8bce66ff51c6
# ╠═2e8caf8a-d2d2-4326-b089-70ef18d74630
# ╠═fc5c2bf9-9aa1-4391-82f0-bf7c538d6985
# ╠═4a9d17be-f8f2-4bab-aa66-ee71077acaec
# ╠═ddd1f894-dde5-46ab-8559-ec5aee176bb6
# ╠═6eb8d6c6-029a-46cd-9ab3-bfd9a3133c69
# ╠═e0a09d91-6316-4137-aa1d-300ab5b8b764
# ╠═33e9f7ac-0a1b-424d-999b-31538c40b77e
# ╠═571a12c5-cea8-44d6-b5d0-25404f55d3c9
