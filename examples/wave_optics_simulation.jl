### A Pluto.jl notebook ###
# v0.19.37

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

# ╔═╡ 25005820-b553-11ee-1189-1976f571f4de
begin
	using Pkg
	Pkg.activate(".")
	Pkg.instantiate()
	using Revise
end

# ╔═╡ e1cdf0a0-4e76-46de-b4fc-70c212bf66c4
using SwissVAMyKnife, ImageShow, ImageIO, PlutoUI, IndexFunArrays, FileIO, Plots, NDTools, CUDA, WaveOpticsPropagation, Optim, RadonKA

# ╔═╡ 8770891f-6f58-4e2e-8d19-dcf6a492c7d4
begin
	# use CUDA if functional
	use_CUDA = Ref(true && CUDA.functional())
	var"@mytime" = use_CUDA[] ? CUDA.var"@time" : var"@time"    
	togoc(x) = use_CUDA[] ? CuArray(x) : x
end

# ╔═╡ f8217d7c-656b-4e1e-a5d9-1fd8bb485596
md"# Load Benchy Boat"

# ╔═╡ 03e90156-d112-4a79-b758-b91e99261973
function load_benchy(sz, sz_file, path)
	target = zeros(Float32, sz_file)

	for i in 1:sz_file[1]
		target[i, :, :] .= select_region(Gray.(load(joinpath(path, string("boat_", string(i, pad=2) ,".png")))), new_size=(sz_file))
	end

	#target2 = zeros(Float32, sz)
	#WaveOpticsPropagation.set_center!(target2, target
	target2 = select_region(target, new_size=sz)
	return togoc(target2)
end

# ╔═╡ 3e8e3d96-2f08-48b6-8bce-632bfff3267c
target = togoc(load_benchy((100, 100, 100), (70, 70, 70), "/home/felix/Downloads/benchy/files/output_70/"));

# ╔═╡ 0fc3a1f1-98b5-46ce-9adf-e31240876f4f
simshow(Array(target[:, :,35]))

# ╔═╡ 043edfd9-734b-4f50-9998-5f64b63dd209
md"# Set up Optimization
It is important to stick with Float32 datatypes for CUDA acceleration
"

# ╔═╡ bc212128-dde3-4835-a36d-00d2630a1ec4
n_resin = 1.5f0

# ╔═╡ 6c8b875e-19bd-4a66-9090-d356090f34df
angles = range(0, 1f0*π, 200)[1:end-1]

# ╔═╡ 518238ba-bfc9-4bc5-9544-8e59de8fcd51


# ╔═╡ afffcc37-b215-4c11-a51e-9435c79338c2
wo = WaveOptics(
	z=togoc(range(-250f-6, 250f-6, size(target,1))), 
	L=500f-6, 
	λ=405f-9 / n_resin, 
	μ=nothing, 
	angles=angles
)

# ╔═╡ b678d10d-ef6d-485e-b752-925f396d8e65
o = GradientBased(optimizer=Optim.LBFGS(),
			thresholds=(0.7f0, 0.8f0),
			iterations=50, loss=:object_space)

# ╔═╡ a90c142a-bd1f-47aa-bbac-c12aee4ffbf9
@mytime patterns, printed, res = optimize_patterns(target, wo, o)

# ╔═╡ 779d383d-faae-41cb-8ceb-75670ff3d2c2
Revise.errors()

# ╔═╡ f6e561ca-9aaf-4178-8fdd-9f7ab706badf
res

# ╔═╡ e1aad0ac-55e3-4bb8-91b6-e8f90033a45d
plot([r.value for r in res.trace], yscale=:log10)

# ╔═╡ f3d71668-131f-486a-8fd0-4878e014d246
typeof(res.trace)

# ╔═╡ 6f135dcf-b7da-45ea-93cd-160d369b3f87
llll[1].value

# ╔═╡ 09c182d7-78a8-4198-ba98-532c028226a1
res

# ╔═╡ c50e3165-0bc0-4698-83da-a14b533997fd
begin
	z=togoc(range(0, 500f-6, size(target,1)))
	L=500f-6
	λ=405f-9 / n_resin
end

# ╔═╡ 7e29a5d9-fa0a-4d74-8e4e-2604dec1a23d
patterns2 = CuArray(PermutedDimsArray(patterns, (1,3,2)));

# ╔═╡ 43d55490-4f6b-4e3e-a97f-8a4013357406
begin
	     AS, _ = AngularSpectrum(patterns2[:, :, 1] .+ 0im, z, λ, L, padding=false)
	     AS_abs2 = let target=target, AS=AS, langles=length(angles)
	             function AS_abs2(x)
	                 abs2.(AS(abs2.(x) .+ 0im)[1]) ./ langles
	             end
	     end 
	     
	     fwd2 = let AS_abs2=AS_abs2, angles=angles
	         function fwd2(x)
	             SwissVAMyKnife.fwd_wave(x, AS_abs2, angles)
	         end
	     end 
end

# ╔═╡ ce5e7dda-2a3a-4828-bf46-8585a4f83936
out2 = fwd2(patterns2);

# ╔═╡ b80bc559-e884-40f1-8283-5079377f6efc
@bind iz2 PlutoUI.Slider(1:size(out2, 3), show_value=true)

# ╔═╡ 64ee978e-241b-49ed-b554-18caafa70ab2
simshow(Array(AS.HW[:,:, iz2]))

# ╔═╡ 7da7b701-3914-4eb3-aec2-3ab7ae0c3d3c
simshow(Array(sqrt.(out2[iz2, :, :])),  cmap=:turbo)

# ╔═╡ 5216196b-08a1-4967-ae39-61c481fc1d1a
size(patterns)

# ╔═╡ 9c9d98d6-f78e-4e87-beee-f0dddef51a2c
extrema(patterns)

# ╔═╡ 3ffeca66-bfc7-4e6e-9da4-30ca74e7bcd8
findmax(patterns)

# ╔═╡ cce6c336-60d7-4e60-a414-cc9a611a1489
sum(patterns) / length(patterns)

# ╔═╡ 2926cc07-1dd0-46f4-8e31-1503380c5a19
sum(printed .> 0.8)

# ╔═╡ 1d5bf241-6dae-4086-9e58-7a2d66ed2ac7
rad2deg(angles[34])

# ╔═╡ 1a41eb6d-797d-4605-8ae0-5a3a57b30651
Revise.errors()

# ╔═╡ 920710f3-5138-42d9-960e-eac0f1c69ca1
findmin(patterns)

# ╔═╡ cddc4559-7ece-4900-b08d-0837486c66eb
@bind angle PlutoUI.Slider(1:size(patterns, 2), show_value=true)

# ╔═╡ 96c54213-28e8-4a54-b938-8a088f8bb1b0


# ╔═╡ a978b980-2279-4118-8085-c714af60cae5
simshow(Array(patterns[:, angle, :]) .|> abs2, cmap=:turbo, set_one=true, γ=1)

# ╔═╡ 62770a3c-6adb-4ad5-85f1-48be4bab8856
simshow(Array(radon(target, angles)[:, angle, :])', cmap=:jet)

# ╔═╡ ce406860-dac6-4404-ac85-8970f1bfe9ee
size(patterns)

# ╔═╡ fa2b9543-f7a7-4f02-8caa-c31d7b1f9a3c
CUDA.@allowscalar patterns[20, 1, 20]

# ╔═╡ 872633b8-8910-484d-96a8-2fa6520642bd
histogram(Array(abs2.(patterns))[:], yscale=:log10)

# ╔═╡ 5ab26a51-8193-4606-9dfd-d7723c60237e
extrema((patterns[:, 10, :]))

# ╔═╡ 1d571981-7585-433e-a12d-8945d700d068
extrema(patterns)

# ╔═╡ acc1b7ab-3f9d-4398-b978-e019231727c0
sum(abs2.(patterns[:, 10, :])) / length(patterns[:, 10, :])

# ╔═╡ 17bac0c3-46e6-47c5-bd55-ec17af4cbb20
@bind depth PlutoUI.Slider(1:size(patterns, 3), show_value=true)

# ╔═╡ 41d71bf9-d874-455d-8010-9d358bdde789
[simshow(Array(printed[:, :, depth]), cmap=:jet) simshow(Array(printed[:, :, depth]) .> 0.65) simshow(Array(target[:, :, depth])) simshow(Array(target[:, :, depth]) .!= Array(printed[:, :, depth] .> 0.65))]

# ╔═╡ b2336899-c42c-4d48-8b7d-eef21cb5531f
@bind iangle3 PlutoUI.Slider(1:size(angles, 1), show_value=true)

# ╔═╡ ac8aa117-098e-4cd9-888f-3a96a26fc70c
simshow(Array(SwissVAMyKnife.imrotate(1 .+ target[:, :, 25], angles[iangle3])))

# ╔═╡ a1d99ac4-777d-496a-a47e-99e49108d64c
simshow(Array(SwissVAMyKnife.imrotate(SwissVAMyKnife.imrotate(1 .+ target[:, :, 25], -angles[iangle3]), angles[iangle3])))

# ╔═╡ e33fbff4-5912-4e33-b824-1447d45494b0
simshow(Array(SwissVAMyKnife.DiffImageRotation.∇imrotate(SwissVAMyKnife.imrotate(1 .+ target[:, :, 25], angles[iangle3]), one.(target[:, :, 45]), angles[iangle3])))

# ╔═╡ 9f58c6c2-57e7-4a3d-92b5-ab25caa9d012
p = plot_histogram(Array(target), Array(printed), (0.7, 0.8); yscale=:log10)

# ╔═╡ ff9624f4-e6f4-4f8b-9dc3-22ea59127511
SwissVAMyKnife.printing_errors(target, printed, (0.7, 0.8))

# ╔═╡ d9e9418a-fc4f-4984-a659-1abd53c12dac
# ╠═╡ disabled = true
#=╠═╡
savefig(p, "/home/felix/Documents/data/wave_optics_simulation_L=400f-6_N=256_different_init/histogram.pdf")
  ╠═╡ =#

# ╔═╡ b2299378-df4d-4f41-b40b-f982a5306e17
fg! = SwissVAMyKnife.make_fg!(identity, target, (0.7f0, 0.8f0), loss=:object_space)

# ╔═╡ 26cd2bf4-6181-45e1-b008-594fb12b4125
g(x) = rand() > 0.5 ? Int : Float32

# ╔═╡ b59792dd-73f4-410f-927e-a1addf3bd72d
SwissVAMyKnife.printing_errors(target, printed, (0.4, 0.43))

# ╔═╡ 12d85121-bc5f-4627-aad1-dddb0b7903e3
extrema(abs2.(patterns))

# ╔═╡ 8aadf716-9ffd-41e4-ae01-3d4494af8526
plot(SwissVAMyKnife.leaky_relu)

# ╔═╡ 5a9ba7d9-ff31-4737-8cad-a91baa788b2f
# ╠═╡ disabled = true
#=╠═╡
#save_patterns("/home/felix/Documents/data/wave_optics_simulation_L=400f-6_N=256_different_init", patterns, printed, wo.angles, target)
  ╠═╡ =#

# ╔═╡ 8f9eb6a8-a0fd-4fdc-8177-079df96964cb
field = randn(ComplexF32, (256, 256));

# ╔═╡ 5fb33bb3-ce9e-444c-8a90-64a377917f1e
sum(abs2, AngularSpectrum(field, 100f-9, 405f-9, 100f-6)[1](field)[1])

# ╔═╡ bed8e523-4c7c-45e2-a73e-a9f6971af8be
sum(abs2, field)

# ╔═╡ Cell order:
# ╠═25005820-b553-11ee-1189-1976f571f4de
# ╠═e1cdf0a0-4e76-46de-b4fc-70c212bf66c4
# ╠═8770891f-6f58-4e2e-8d19-dcf6a492c7d4
# ╟─f8217d7c-656b-4e1e-a5d9-1fd8bb485596
# ╠═03e90156-d112-4a79-b758-b91e99261973
# ╠═3e8e3d96-2f08-48b6-8bce-632bfff3267c
# ╠═0fc3a1f1-98b5-46ce-9adf-e31240876f4f
# ╟─043edfd9-734b-4f50-9998-5f64b63dd209
# ╠═bc212128-dde3-4835-a36d-00d2630a1ec4
# ╠═6c8b875e-19bd-4a66-9090-d356090f34df
# ╠═518238ba-bfc9-4bc5-9544-8e59de8fcd51
# ╠═afffcc37-b215-4c11-a51e-9435c79338c2
# ╠═b678d10d-ef6d-485e-b752-925f396d8e65
# ╠═a90c142a-bd1f-47aa-bbac-c12aee4ffbf9
# ╠═779d383d-faae-41cb-8ceb-75670ff3d2c2
# ╠═f6e561ca-9aaf-4178-8fdd-9f7ab706badf
# ╠═e1aad0ac-55e3-4bb8-91b6-e8f90033a45d
# ╠═f3d71668-131f-486a-8fd0-4878e014d246
# ╠═6f135dcf-b7da-45ea-93cd-160d369b3f87
# ╠═09c182d7-78a8-4198-ba98-532c028226a1
# ╠═c50e3165-0bc0-4698-83da-a14b533997fd
# ╠═43d55490-4f6b-4e3e-a97f-8a4013357406
# ╠═64ee978e-241b-49ed-b554-18caafa70ab2
# ╠═ce5e7dda-2a3a-4828-bf46-8585a4f83936
# ╠═b80bc559-e884-40f1-8283-5079377f6efc
# ╠═7da7b701-3914-4eb3-aec2-3ab7ae0c3d3c
# ╠═7e29a5d9-fa0a-4d74-8e4e-2604dec1a23d
# ╠═5216196b-08a1-4967-ae39-61c481fc1d1a
# ╠═9c9d98d6-f78e-4e87-beee-f0dddef51a2c
# ╠═3ffeca66-bfc7-4e6e-9da4-30ca74e7bcd8
# ╠═cce6c336-60d7-4e60-a414-cc9a611a1489
# ╠═2926cc07-1dd0-46f4-8e31-1503380c5a19
# ╠═1d5bf241-6dae-4086-9e58-7a2d66ed2ac7
# ╠═1a41eb6d-797d-4605-8ae0-5a3a57b30651
# ╠═920710f3-5138-42d9-960e-eac0f1c69ca1
# ╠═cddc4559-7ece-4900-b08d-0837486c66eb
# ╠═96c54213-28e8-4a54-b938-8a088f8bb1b0
# ╠═a978b980-2279-4118-8085-c714af60cae5
# ╠═62770a3c-6adb-4ad5-85f1-48be4bab8856
# ╠═ce406860-dac6-4404-ac85-8970f1bfe9ee
# ╠═fa2b9543-f7a7-4f02-8caa-c31d7b1f9a3c
# ╠═872633b8-8910-484d-96a8-2fa6520642bd
# ╠═5ab26a51-8193-4606-9dfd-d7723c60237e
# ╠═1d571981-7585-433e-a12d-8945d700d068
# ╠═acc1b7ab-3f9d-4398-b978-e019231727c0
# ╠═17bac0c3-46e6-47c5-bd55-ec17af4cbb20
# ╠═41d71bf9-d874-455d-8010-9d358bdde789
# ╠═ac8aa117-098e-4cd9-888f-3a96a26fc70c
# ╠═a1d99ac4-777d-496a-a47e-99e49108d64c
# ╠═e33fbff4-5912-4e33-b824-1447d45494b0
# ╠═b2336899-c42c-4d48-8b7d-eef21cb5531f
# ╠═9f58c6c2-57e7-4a3d-92b5-ab25caa9d012
# ╠═ff9624f4-e6f4-4f8b-9dc3-22ea59127511
# ╠═d9e9418a-fc4f-4984-a659-1abd53c12dac
# ╠═b2299378-df4d-4f41-b40b-f982a5306e17
# ╠═26cd2bf4-6181-45e1-b008-594fb12b4125
# ╠═b59792dd-73f4-410f-927e-a1addf3bd72d
# ╠═12d85121-bc5f-4627-aad1-dddb0b7903e3
# ╠═8aadf716-9ffd-41e4-ae01-3d4494af8526
# ╠═5a9ba7d9-ff31-4737-8cad-a91baa788b2f
# ╠═8f9eb6a8-a0fd-4fdc-8177-079df96964cb
# ╠═5fb33bb3-ce9e-444c-8a90-64a377917f1e
# ╠═bed8e523-4c7c-45e2-a73e-a9f6971af8be
