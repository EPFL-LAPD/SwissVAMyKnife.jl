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
angles = range(0, 1f0*π, 100)[1:end-1]

# ╔═╡ 22fb0a2b-bf29-405a-bf08-a2c118de4276
L = 100f-6

# ╔═╡ afffcc37-b215-4c11-a51e-9435c79338c2
waveoptics = WaveOptics(
	z=togoc(range(-L/2, L/2, size(target,1))), 
	L=L, 
	λ=405f-9 / n_resin, 
	μ=nothing, 
	angles=angles)

# ╔═╡ b678d10d-ef6d-485e-b752-925f396d8e65
optimizer = GradientBased(optimizer=Optim.LBFGS(), options=Optim.Options(iterations=30, store_trace=true))

# ╔═╡ a77abf06-4c40-4053-9702-57c1da461417
loss = Threshold(thresholds=(0.7f0, 0.8f0))

# ╔═╡ a90c142a-bd1f-47aa-bbac-c12aee4ffbf9
@mytime patterns, printed, res = optimize_patterns(target, waveoptics, optimizer, loss)

# ╔═╡ f6e561ca-9aaf-4178-8fdd-9f7ab706badf
res

# ╔═╡ e1aad0ac-55e3-4bb8-91b6-e8f90033a45d
plot([0.001 + r.value for r in res.trace], yscale=:log10)

# ╔═╡ a6b96898-b9e2-4e7f-b146-9a16ad67eaea
md"# Inspect"

# ╔═╡ 17bac0c3-46e6-47c5-bd55-ec17af4cbb20
@bind depth PlutoUI.Slider(1:size(patterns, 3), show_value=true)

# ╔═╡ 2a1451e9-27ef-4498-ab74-07c8fd55d08a
[simshow(Array(printed[:, :, depth]), cmap=:jet) simshow(Array(printed[:, :, depth]) .> 0.75) simshow(Array(target[:, :, depth])) simshow(Array(target[:, :, depth]) .!= Array(printed[:, :, depth] .> 0.75))]

# ╔═╡ 4df4a07c-6000-48db-a0fd-596786f574c4


# ╔═╡ cddc4559-7ece-4900-b08d-0837486c66eb
@bind angle PlutoUI.Slider(1:size(patterns, 2), show_value=true)

# ╔═╡ a978b980-2279-4118-8085-c714af60cae5
simshow(Array(patterns[:, angle, :]), cmap=:turbo, set_one=true, γ=1)

# ╔═╡ 9f58c6c2-57e7-4a3d-92b5-ab25caa9d012
p = plot_histogram(Array(target), Array(printed), (0.7, 0.8); yscale=:log10)

# ╔═╡ ff9624f4-e6f4-4f8b-9dc3-22ea59127511
SwissVAMyKnife.printing_errors(target, printed, (0.7, 0.8))

# ╔═╡ 82b58277-ea83-471e-a9e3-df340fa0f941
save_patterns("/home/felix/Documents/data/wave_optics_simulation_L=$(round(L, digits=6))_N=$(size(target,1))/", patterns, printed, angles, target; overwrite=true)

# ╔═╡ d9e9418a-fc4f-4984-a659-1abd53c12dac
savefig(p, "/home/felix/Documents/data/wave_optics_simulation_L=$(round(L, digits=6))_N=$(size(target,1))/histogram.pdf")

# ╔═╡ c50e3165-0bc0-4698-83da-a14b533997fd
begin
	z=togoc(range(-L/2, L/2, size(target,1)))
	λ=405f-9 / n_resin
end

# ╔═╡ 4b93227d-6beb-45f1-b8ad-3f57b86005ae
CuArray([1,2,3f0]) .* 2.0

# ╔═╡ 4ec4f26e-cfbb-4da6-8130-fddf542c4ecd


# ╔═╡ c8665e57-d4f6-448f-9c18-1fb16b317a7b
@bind iangle3 PlutoUI.Slider(1:size(angles, 1), show_value=true, default=50)

# ╔═╡ 43a8e1c3-3178-4a20-8448-f7e140a747de
angles2 = angles[iangle3:iangle3]

# ╔═╡ 7e29a5d9-fa0a-4d74-8e4e-2604dec1a23d
patterns2 = CuArray(PermutedDimsArray(patterns, (1,3,2)));

# ╔═╡ 43d55490-4f6b-4e3e-a97f-8a4013357406
begin
	 AS, _ = AngularSpectrum(patterns2[:, :, 1] .+ 0im, z, λ, L, padding=false)
	 AS_abs2 = let target=target, AS=AS, langles=length(angles2)
			 function AS_abs2(x)
				 abs2.(AS(max.(0, x) .+ 0im)[1]) ./ langles
			 end
	 end 
	 
	 fwd2 = let AS_abs2=AS_abs2, angles=angles2
		 function fwd2(x)
			 SwissVAMyKnife.fwd_wave(x, AS_abs2, angles2)
		 end
	 end 
end

# ╔═╡ 43f2e173-bb96-461d-b8ce-e4539f9e14d1
size(patterns2)

# ╔═╡ ce5e7dda-2a3a-4828-bf46-8585a4f83936
out2 = fwd2(patterns2[:,:, iangle3:iangle3]);

# ╔═╡ b80bc559-e884-40f1-8283-5079377f6efc
@bind iz2 PlutoUI.Slider(1:size(out2, 3), show_value=true, default=50)

# ╔═╡ 622a0419-2e9e-4254-8fbb-3c8bd340b00f
mask = IndexFunArrays.rr2(size(out2[iz2, :, :])) .<= size(out2[iz2, :, :], 1).^2 ./ 4;

# ╔═╡ 7da7b701-3914-4eb3-aec2-3ab7ae0c3d3c
p3 = simshow(Array(out2[iz2, :, :]) .* mask,  cmap=:turbo)

# ╔═╡ 3c5a61ea-b299-46a7-87d1-935386473119
save("/home/felix/Documents/data/wave_optics_simulation_L=$(round(L, digits=6))_N=$(size(target,1))/backprojection_1.png", simshow(Array(out2[iz2, :, :]) .* mask,  cmap=:turbo))

# ╔═╡ 74508e95-735d-4f79-94c6-64662ca06e4e
simshow(mask)

# ╔═╡ 72fa1dbb-0cc7-4283-85fe-b70fd3864d70
size(patterns2)

# ╔═╡ 62770a3c-6adb-4ad5-85f1-48be4bab8856
simshow(Array(radon(target, angles)[:, angle, :])', cmap=:jet)

# ╔═╡ ce406860-dac6-4404-ac85-8970f1bfe9ee
size(patterns)

# ╔═╡ 872633b8-8910-484d-96a8-2fa6520642bd
histogram(Array(abs2.(patterns))[:], yscale=:log10)

# ╔═╡ Cell order:
# ╠═25005820-b553-11ee-1189-1976f571f4de
# ╠═e1cdf0a0-4e76-46de-b4fc-70c212bf66c4
# ╠═8770891f-6f58-4e2e-8d19-dcf6a492c7d4
# ╟─f8217d7c-656b-4e1e-a5d9-1fd8bb485596
# ╠═03e90156-d112-4a79-b758-b91e99261973
# ╠═3e8e3d96-2f08-48b6-8bce-632bfff3267c
# ╠═0fc3a1f1-98b5-46ce-9adf-e31240876f4f
# ╠═043edfd9-734b-4f50-9998-5f64b63dd209
# ╠═bc212128-dde3-4835-a36d-00d2630a1ec4
# ╠═6c8b875e-19bd-4a66-9090-d356090f34df
# ╠═22fb0a2b-bf29-405a-bf08-a2c118de4276
# ╠═afffcc37-b215-4c11-a51e-9435c79338c2
# ╠═b678d10d-ef6d-485e-b752-925f396d8e65
# ╠═a77abf06-4c40-4053-9702-57c1da461417
# ╠═a90c142a-bd1f-47aa-bbac-c12aee4ffbf9
# ╠═f6e561ca-9aaf-4178-8fdd-9f7ab706badf
# ╠═e1aad0ac-55e3-4bb8-91b6-e8f90033a45d
# ╟─a6b96898-b9e2-4e7f-b146-9a16ad67eaea
# ╠═17bac0c3-46e6-47c5-bd55-ec17af4cbb20
# ╠═2a1451e9-27ef-4498-ab74-07c8fd55d08a
# ╠═4df4a07c-6000-48db-a0fd-596786f574c4
# ╠═cddc4559-7ece-4900-b08d-0837486c66eb
# ╠═a978b980-2279-4118-8085-c714af60cae5
# ╠═9f58c6c2-57e7-4a3d-92b5-ab25caa9d012
# ╠═ff9624f4-e6f4-4f8b-9dc3-22ea59127511
# ╠═82b58277-ea83-471e-a9e3-df340fa0f941
# ╠═d9e9418a-fc4f-4984-a659-1abd53c12dac
# ╠═c50e3165-0bc0-4698-83da-a14b533997fd
# ╠═43a8e1c3-3178-4a20-8448-f7e140a747de
# ╠═4b93227d-6beb-45f1-b8ad-3f57b86005ae
# ╠═4ec4f26e-cfbb-4da6-8130-fddf542c4ecd
# ╠═43d55490-4f6b-4e3e-a97f-8a4013357406
# ╠═c8665e57-d4f6-448f-9c18-1fb16b317a7b
# ╠═43f2e173-bb96-461d-b8ce-e4539f9e14d1
# ╠═ce5e7dda-2a3a-4828-bf46-8585a4f83936
# ╠═72fa1dbb-0cc7-4283-85fe-b70fd3864d70
# ╠═b80bc559-e884-40f1-8283-5079377f6efc
# ╠═7da7b701-3914-4eb3-aec2-3ab7ae0c3d3c
# ╠═3c5a61ea-b299-46a7-87d1-935386473119
# ╠═622a0419-2e9e-4254-8fbb-3c8bd340b00f
# ╠═74508e95-735d-4f79-94c6-64662ca06e4e
# ╠═7e29a5d9-fa0a-4d74-8e4e-2604dec1a23d
# ╠═62770a3c-6adb-4ad5-85f1-48be4bab8856
# ╠═ce406860-dac6-4404-ac85-8970f1bfe9ee
# ╠═872633b8-8910-484d-96a8-2fa6520642bd
