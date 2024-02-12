### A Pluto.jl notebook ###
# v0.19.35

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

# ╔═╡ f90cf4dd-6844-4515-a877-19b0d6a07f3f
#=╠═╡
simshow(Array(target[:, :, 25]))
  ╠═╡ =#

# ╔═╡ 29461391-9128-4411-b33e-97f3ce6625a7
#=╠═╡
@bind izzz Slider(1:size(target, 3), show_value=true)
  ╠═╡ =#

# ╔═╡ 5b45efde-da5a-45af-968b-208d04e43517
#=╠═╡
simshow(Array(target[:, :, izzz]))
  ╠═╡ =#

# ╔═╡ 8f12a113-6772-4567-95bc-d43f72d303ca


# ╔═╡ 89882c40-2e93-4a15-a3c6-09ae9ade39eb
md"# Specify angles and thresholds for Optimization"

# ╔═╡ 117da88b-4981-4be5-b79e-93a2938ff2e1
200 / 2 * 2π

# ╔═╡ fd7a91df-5edd-479d-a810-3f30e38a2a21
angles = range(0f0, 2f0 * π, 200)

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
@time patterns, printed_i, res = optimize_patterns(target, angles, iterations=100,
											method=:radon_iterative,
											thresholds=thresholds,
											μ=nothing)

# ╔═╡ 6c3a476a-087f-4988-93d3-65ebfdadc26b
plot(res, title="Loss over iterations", xlabel="Iterations", ylabel="Loss", yscale=:log10)

# ╔═╡ 2f7d32c6-912d-43fd-bfb8-75713f0dd0ef
md"Threshold=$(@bind threshold Slider(0.0:0.01:1, show_value=true))"

# ╔═╡ c4865085-0b1f-42be-ad88-9b8d694278e5
#=╠═╡
md"depth in z=$(@bind z_i Slider(1:1:size(target, 3), show_value=true))"
  ╠═╡ =#

# ╔═╡ 6d78294c-4c92-457a-91de-4bd8a881de71
#=╠═╡
[simshow(Array(printed_i[:,:, z_i]), cmap=:turbo) simshow(Array(printed_i[:, :, z_i]) .> threshold) simshow(Array(target[:, :, z_i]))]
  ╠═╡ =#

# ╔═╡ b5cdb60e-6294-4caf-b8b3-9bef8db0d137
md"angle=$(@bind i_angle Slider(1:1:size(patterns, 2), show_value=true))"

# ╔═╡ 9a81a25c-bf91-4b1e-9713-ae65c7db2d16
simshow(Array(patterns[:, i_angle, :])', cmap=:turbo)

# ╔═╡ b953013d-aae8-4b90-89ed-d76eb190bba4
#=╠═╡
p1 = plot_intensity(Array(target[begin:1:end]), Array(printed_i[begin:1:end]), (0.65, 0.75))
  ╠═╡ =#

# ╔═╡ 0e3585a4-8079-4272-b0d1-bd4949dd872a
#=╠═╡
savefig(p1, "/home/felix/Documents/data/candidacy/histogram_radon_boat.pdf")
  ╠═╡ =#

# ╔═╡ 0e167ace-96ce-443a-adcc-ae06229a2d5f
#=╠═╡
size(target)
  ╠═╡ =#

# ╔═╡ 238931ba-9ce9-496a-b424-a9f1682b6d99
md"# Wave Optics"

# ╔═╡ 5d110682-623d-46cc-9657-17ecc47c79bf
L = 400f-6

# ╔═╡ 0d0b3478-fc0d-4f1b-b956-ff220bd6539a
λ = 405f-9 ./ 1.5

# ╔═╡ ad64cd2d-158f-41c1-af8d-b1b713a8b502
#=╠═╡
z = togoc(range(0, L, size(target, 1)));
  ╠═╡ =#

# ╔═╡ 41f01110-9962-44d1-ae97-2fa711a0c521
SwissVAMyKnife.leaky_relu.(CUDA.rand(2,2))

# ╔═╡ 08d04781-b4c7-41da-8e2f-107d5719239a
@time patterns_wave, printed_i_wave, res_wave = optimize_patterns(target, angles; iterations=150, method=:wave, loss=:object_space, optimizer=LBFGS(), thresholds=thresholds, μ=nothing, L, λ, z)

# ╔═╡ 29ac74c2-ab2a-45ec-835b-1c6f6c2886f9
Revise.retry()

# ╔═╡ 53c2c266-e0ae-4fd8-9f90-e1cf700cdba8
res_wave

# ╔═╡ b0a3f7bc-c435-450b-9429-60b3c9cac929
sum(patterns_wave) / length(patterns_wave)

# ╔═╡ 902720fb-962d-4d4f-9eca-e0812d031871
SwissVAMyKnife.leaky_relu(-10.0)

# ╔═╡ 575ba3c2-1f80-4193-98ad-fdf0e73341d8
md"angle=$(@bind i_angle2 Slider(1:1:size(patterns, 2), show_value=true))"

# ╔═╡ 8ce5a252-9fac-47b3-a84e-351ba723c243
simshow(Array((patterns_wave[:, i_angle2, :])), γ=1, cmap=:turbo)

# ╔═╡ 34d86aad-4a33-4e54-9c29-e0b61e044574
extrema(patterns_wave)

# ╔═╡ 60c856a5-ff82-4de0-86b0-8c3ce8d03049
#=╠═╡
p2 = plot_intensity(Array(target), Array(printed_i_wave), (0.65, 0.75))
  ╠═╡ =#

# ╔═╡ f6f98113-d480-4a0e-9acd-60043c70b4cb
#=╠═╡
savefig(p2, "/home/felix/Documents/data/candidacy/histogram_wave_boat.pdf")
  ╠═╡ =#

# ╔═╡ 105db6be-382f-4521-837c-bc7d39a3dce9
md"Threshold=$(@bind threshold3 Slider(0.0:0.01:1, show_value=true))"

# ╔═╡ afddd44a-3b23-48d1-aeb6-a7b9a05e303b
#=╠═╡
md"depth in z=$(@bind z_i3 Slider(1:1:size(target, 3), show_value=true))"
  ╠═╡ =#

# ╔═╡ 71247ebd-9be4-4328-a1e6-a2fa82d3ae2a
#=╠═╡
[simshow(Array(printed_i_wave[:,:, z_i3]), cmap=:turbo, set_one=false) simshow(Array(printed_i_wave[:, :, z_i3]) .> threshold3) simshow(Array(target[:, :, z_i3]))]
  ╠═╡ =#

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

# ╔═╡ d6624c7d-9266-4a41-843e-ddd54bcda7ec
#=╠═╡
printing_errors(printed_i_wave, target, thresholds)
  ╠═╡ =#

# ╔═╡ 6c106f9c-0690-4170-a383-5143376093e1
#=╠═╡
printing_errors(printed_i, target, thresholds)
  ╠═╡ =#

# ╔═╡ 347a0bc9-e387-4dab-a74c-3ff5d5f1a33a
#=╠═╡
[simshow(Array(printed_i[:,:, z_i]), cmap=:turbo) simshow(Array(printed_i[:, :, z_i]) .> threshold) simshow(Array(target[:, :, z_i]))]
  ╠═╡ =#

# ╔═╡ b86d65cd-b52f-405d-874a-7a8f6fce732e
simshow(Array(printed_i[:, 20, :]))

# ╔═╡ 85aa66e4-fc61-4e81-9d1c-8bce66ff51c6
extrema(printed_i)

# ╔═╡ 2e8caf8a-d2d2-4326-b089-70ef18d74630
simshow(Array(patterns[:, 4, :]), cmap=:turbo)

# ╔═╡ fc5c2bf9-9aa1-4391-82f0-bf7c538d6985
sum(patterns) / length(patterns)

# ╔═╡ 4a9d17be-f8f2-4bab-aa66-ee71077acaec
#=╠═╡
begin
	         AS, _ = AngularSpectrum(permutedims(patterns_wave, (1, 3,2))[:, :, 1] .+ 0im, z, λ, L, padding=false)    
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
  ╠═╡ =#

# ╔═╡ fd4e50a4-1a94-498d-9fd5-0274424f97fd
#=╠═╡
simshow(permutedims(Array(fwd2(sqrt.(permutedims(patterns_wave, (1,3,2))))), (3,2,1)))[:, :, 20]
  ╠═╡ =#

# ╔═╡ 73e14d8f-a76a-4ba9-a368-aa743da47160
#=╠═╡
extrema((permutedims(Array(fwd2(sqrt.(permutedims(0 .* patterns_wave, (1,3,2))))), (3,2,1)))[:, :, 20])
  ╠═╡ =#

# ╔═╡ 8c389472-7bcc-4a3d-af1d-de4ad969a077
size(patterns)

# ╔═╡ 1ffd0a6b-c413-46c0-a5c0-181b36f13c96
size(patterns_wave)

# ╔═╡ 8e68bbcb-f648-47fe-81ef-305d4d6e8ec2
#=╠═╡
begin
	intensity_radon = permutedims(Array(fwd2(sqrt.(permutedims(select_region(permutedims(patterns, (3,2,1)), new_size=(size(patterns,1)+1, size(patterns)[2:3]...)), (1,3,2))).^0.5f0)), (3,2,1))
	intensity_radon ./= maximum(intensity_radon)
end;
  ╠═╡ =#

# ╔═╡ e0792c95-72ba-4af2-add3-fe50cea3cc80
#=╠═╡
begin
	intensity_wave = permutedims(Array(fwd2(sqrt.(permutedims(patterns_wave, (1,3,2))))), (3,2,1))
	intensity_wave ./= maximum(intensity_wave)
end;
  ╠═╡ =#

# ╔═╡ 62753472-6c6a-4cc4-9473-f5e32b1a9735
md"Threshold=$(@bind threshold4 Slider(0.0:0.01:1, show_value=true))"

# ╔═╡ e0fd1a8a-aa2b-4c5a-afe0-6c346b340589
md"Threshold=$(@bind threshold5 Slider(0.0:0.01:1, show_value=true))"

# ╔═╡ 70115b27-c18d-4e5e-8be7-87af641e751b
#=╠═╡
md"depth in z=$(@bind z_i4 Slider(1:1:size(target, 3), show_value=true))"
  ╠═╡ =#

# ╔═╡ 4eb9b1c1-3969-483c-bc8e-d57530f6ce8e
#=╠═╡
[simshow(intensity_radon[:, :, z_i4] .> threshold5) simshow(intensity_radon[:, :, z_i4], set_one=false) simshow(intensity_wave[:, :, z_i4] .> threshold4) simshow(simshow(intensity_wave[:, :, z_i4], set_one=false)) simshow(Array(target)[:, :, z_i4])]
  ╠═╡ =#

# ╔═╡ c272d6e0-80f5-4cb4-9d7f-f6ccc3857412
#=╠═╡
wave_printed =  simshow(intensity_wave[:, :, z_i4] .> threshold4)
  ╠═╡ =#

# ╔═╡ 9e6cbeef-08b7-4b63-98a1-729875ef05e4
size(patterns_wave)

# ╔═╡ 5c452db0-f96d-470b-84ec-2ba0751838c3
#=╠═╡
save("/home/felix/Documents/data/candidacy/wave_simulation.png", wave_printed)
  ╠═╡ =#

# ╔═╡ c43d8d6f-a4da-4704-b142-6eda10aa8833
Revise.retry()

# ╔═╡ 390391d4-4ae6-4f46-8d51-ed29b93c5209
extrema(printed_i_wave)

# ╔═╡ b3bc4f92-2b45-42f4-942f-685726fcd201
#=╠═╡
radon_wave_printed =  simshow(intensity_radon[:, :, z_i4] .> threshold5)
  ╠═╡ =#

# ╔═╡ e3ea1e0d-4834-46d6-8417-7d10851df5f3
begin
	pattern_wave_example = simshow(Array(patterns_wave[:, 51, :]), cmap=:turbo)
	save("/home/felix/Documents/data/candidacy/pattern_wave_example.png", pattern_wave_example)
end

# ╔═╡ 7a41d265-ab8f-496f-b36a-d0072131e91b
begin
	pattern_radon_example = simshow(Array(patterns[:, 51, :]'), cmap=:turbo)
	save("/home/felix/Documents/data/candidacy/pattern_radon_example.png", pattern_radon_example)
end

# ╔═╡ 92dbd9d4-314b-407c-9420-a5b54287aef7
simshow(Array(patterns_wave[:, 61, :]))

# ╔═╡ 9b119ca1-06ef-4781-9049-04a34f328068
extrema(patterns_wave)

# ╔═╡ d187a388-06b2-4f48-983b-925f20350923
#=╠═╡
save_patterns("/home/felix/Documents/data/candidacy/boat_wave", patterns_wave, printed_i_wave, angles, target, overwrite=true)
  ╠═╡ =#

# ╔═╡ ff8970ab-0e04-4805-bb31-25e782a0f2ba
#=╠═╡
save_patterns("/home/felix/Documents/data/candidacy/boat_radon", permutedims(patterns, (3, 2,1)), printed_i, angles, target, overwrite=true)
  ╠═╡ =#

# ╔═╡ 5a16bdce-418a-4ea7-bfde-e2461603f0df
Revise.retry()

# ╔═╡ 1e07c210-3e27-44e5-8b8f-07811d8ef917
#=╠═╡
target_printed =  simshow(Array(target)[:, :, z_i4])
  ╠═╡ =#

# ╔═╡ 6dd224fc-ba01-49a4-97ab-67d2712f810b
#=╠═╡
save("/home/felix/Documents/data/candidacy/target.png", target_printed)
  ╠═╡ =#

# ╔═╡ 524a7c04-3b61-4132-bfdf-a16c1d64453e
#=╠═╡
save("/home/felix/Documents/data/candidacy/wave_radon_simulation.png", radon_wave_printed)
  ╠═╡ =#

# ╔═╡ e5181ea2-12be-40eb-acae-5d4b9cc9b3c6
#=╠═╡
plot_intensity(Array(target), Array(intensity_radon), (0.65, 0.75))
  ╠═╡ =#

# ╔═╡ 82039c35-f97c-476c-9efd-6a468bffe7f8


# ╔═╡ ddd1f894-dde5-46ab-8559-ec5aee176bb6
#=╠═╡
sum(abs2, AS(sqrt.(patterns_wave[:, :, 10]))[1])
  ╠═╡ =#

# ╔═╡ 6eb8d6c6-029a-46cd-9ab3-bfd9a3133c69
sum(abs2, patterns_wave[:, :, 10])

# ╔═╡ 33e9f7ac-0a1b-424d-999b-31538c40b77e
size(patterns_wave)

# ╔═╡ 571a12c5-cea8-44d6-b5d0-25404f55d3c9
sum(abs2, patterns_wave)

# ╔═╡ db29a6a5-2635-4fbe-8111-8c369fafa6e1
# ╠═╡ disabled = true
#=╠═╡
begin
	target = togoc(zeros(Float32, (100, 100, 100)))
	
	for i in 1:80
		target[begin+10+i÷2:end-10-i÷2,begin+10+i÷3:end-10-i÷3, 10 + i] .= 1
	end
end
  ╠═╡ =#

# ╔═╡ 66e90a47-981e-4d19-81aa-90e262f64158
# ╠═╡ disabled = true
#=╠═╡
target = select_region(target2, new_size=(100, 100, 100))
  ╠═╡ =#

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

# ╔═╡ edd4580e-4cf2-4623-a127-007547b88a03
#=╠═╡
target = load_benchy((252, 252, 252), (200, 200, 200), "/home/felix/Downloads/benchy/files/output/");
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═30aab62e-a0eb-11ee-06db-11eb60e1a051
# ╠═6f8d4329-9b45-410b-98a2-17cb22475060
# ╠═f4772bab-4700-443f-9e86-7fb35b616551
# ╠═ebd218dd-6e34-4d49-87bf-bb552c89db97
# ╠═2a21b5cd-9a73-4da8-b5b6-2a59aeefaf03
# ╠═d1880cdd-22b0-41d2-8887-d78f82081259
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
# ╠═b953013d-aae8-4b90-89ed-d76eb190bba4
# ╠═0e3585a4-8079-4272-b0d1-bd4949dd872a
# ╠═0e167ace-96ce-443a-adcc-ae06229a2d5f
# ╟─238931ba-9ce9-496a-b424-a9f1682b6d99
# ╠═5d110682-623d-46cc-9657-17ecc47c79bf
# ╠═0d0b3478-fc0d-4f1b-b956-ff220bd6539a
# ╠═ad64cd2d-158f-41c1-af8d-b1b713a8b502
# ╠═41f01110-9962-44d1-ae97-2fa711a0c521
# ╠═08d04781-b4c7-41da-8e2f-107d5719239a
# ╠═29ac74c2-ab2a-45ec-835b-1c6f6c2886f9
# ╠═53c2c266-e0ae-4fd8-9f90-e1cf700cdba8
# ╠═b0a3f7bc-c435-450b-9429-60b3c9cac929
# ╠═902720fb-962d-4d4f-9eca-e0812d031871
# ╟─575ba3c2-1f80-4193-98ad-fdf0e73341d8
# ╠═8ce5a252-9fac-47b3-a84e-351ba723c243
# ╠═34d86aad-4a33-4e54-9c29-e0b61e044574
# ╠═60c856a5-ff82-4de0-86b0-8c3ce8d03049
# ╠═f6f98113-d480-4a0e-9acd-60043c70b4cb
# ╟─105db6be-382f-4521-837c-bc7d39a3dce9
# ╟─afddd44a-3b23-48d1-aeb6-a7b9a05e303b
# ╠═71247ebd-9be4-4328-a1e6-a2fa82d3ae2a
# ╟─463b7a89-4df0-4442-8d6e-2b6b251fbc1b
# ╠═19c2d9dd-ec68-420e-9d06-2c66ddbe0125
# ╠═c443703d-cd2a-4e2a-bfd0-742b4fcb001f
# ╠═d6624c7d-9266-4a41-843e-ddd54bcda7ec
# ╠═6c106f9c-0690-4170-a383-5143376093e1
# ╟─347a0bc9-e387-4dab-a74c-3ff5d5f1a33a
# ╠═b86d65cd-b52f-405d-874a-7a8f6fce732e
# ╠═85aa66e4-fc61-4e81-9d1c-8bce66ff51c6
# ╠═2e8caf8a-d2d2-4326-b089-70ef18d74630
# ╠═fc5c2bf9-9aa1-4391-82f0-bf7c538d6985
# ╠═4a9d17be-f8f2-4bab-aa66-ee71077acaec
# ╠═fd4e50a4-1a94-498d-9fd5-0274424f97fd
# ╠═73e14d8f-a76a-4ba9-a368-aa743da47160
# ╠═8c389472-7bcc-4a3d-af1d-de4ad969a077
# ╠═1ffd0a6b-c413-46c0-a5c0-181b36f13c96
# ╠═8e68bbcb-f648-47fe-81ef-305d4d6e8ec2
# ╠═e0792c95-72ba-4af2-add3-fe50cea3cc80
# ╠═4eb9b1c1-3969-483c-bc8e-d57530f6ce8e
# ╟─62753472-6c6a-4cc4-9473-f5e32b1a9735
# ╟─e0fd1a8a-aa2b-4c5a-afe0-6c346b340589
# ╟─70115b27-c18d-4e5e-8be7-87af641e751b
# ╠═c272d6e0-80f5-4cb4-9d7f-f6ccc3857412
# ╠═9e6cbeef-08b7-4b63-98a1-729875ef05e4
# ╠═5c452db0-f96d-470b-84ec-2ba0751838c3
# ╠═c43d8d6f-a4da-4704-b142-6eda10aa8833
# ╠═390391d4-4ae6-4f46-8d51-ed29b93c5209
# ╠═b3bc4f92-2b45-42f4-942f-685726fcd201
# ╠═e3ea1e0d-4834-46d6-8417-7d10851df5f3
# ╠═7a41d265-ab8f-496f-b36a-d0072131e91b
# ╠═92dbd9d4-314b-407c-9420-a5b54287aef7
# ╠═9b119ca1-06ef-4781-9049-04a34f328068
# ╠═d187a388-06b2-4f48-983b-925f20350923
# ╠═ff8970ab-0e04-4805-bb31-25e782a0f2ba
# ╠═5a16bdce-418a-4ea7-bfde-e2461603f0df
# ╠═1e07c210-3e27-44e5-8b8f-07811d8ef917
# ╠═6dd224fc-ba01-49a4-97ab-67d2712f810b
# ╠═524a7c04-3b61-4132-bfdf-a16c1d64453e
# ╠═e5181ea2-12be-40eb-acae-5d4b9cc9b3c6
# ╠═82039c35-f97c-476c-9efd-6a468bffe7f8
# ╠═ddd1f894-dde5-46ab-8559-ec5aee176bb6
# ╠═6eb8d6c6-029a-46cd-9ab3-bfd9a3133c69
# ╠═33e9f7ac-0a1b-424d-999b-31538c40b77e
# ╠═571a12c5-cea8-44d6-b5d0-25404f55d3c9
