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

# ╔═╡ feea18a4-bf54-11ee-0a29-359b1cb4c099
begin
	using Pkg
	Pkg.activate(".")
	Pkg.instantiate()
	using Revise
end

# ╔═╡ d25e6ac5-e9ad-408f-9855-c444ea98b73e
using SwissVAMyKnife, ImageShow, ImageIO, PlutoUI, IndexFunArrays, FileIO, Plots, NDTools, CUDA, WaveOpticsPropagation, Optim, RadonKA

# ╔═╡ 0af67a5c-b689-44a9-999b-89f4c3dc40a0
md"# 0. Load Packages"

# ╔═╡ 8662f199-f816-4572-aa7e-7043bee5fca8
begin
	# use CUDA if functional
	use_CUDA = Ref(true && CUDA.functional())
	var"@mytime" = use_CUDA[] ? CUDA.var"@time" : var"@time"    
	togoc(x) = use_CUDA[] ? CuArray(x) : x
	togoc(x::AbstractArray{Float64}) = use_CUDA[] ? CuArray(Float32.(x)) : x
end

# ╔═╡ 1ac1d9f9-e864-464a-abf7-ec6a5faa8c98
md"## CUDA works: $(use_CUDA[])

Without CUDA, the optimization takes longer
"

# ╔═╡ 7ada448e-761b-4330-8f0a-a9b15767c2e6
TableOfContents()

# ╔═╡ 896ebd29-f858-425e-9bdd-9b1d1fae0d1f
md"# 1. Load Benchy"

# ╔═╡ f507d697-d918-4b41-a99d-cc3aa1c2e7d3
function load_benchy(sz, sz_file, path)
	target = zeros(Float32, sz_file)

	pad = sz_file[1] > 99 ? 3 : 2 
	for i in 1:sz_file[1]
		target[i, :, :] .= select_region(Gray.(load(joinpath(path, string("boat_", string(i, pad=pad) ,".png")))), new_size=(sz_file))
	end

	target2 = select_region(target, new_size=sz)
	return togoc(target2)
end

# ╔═╡ 1ee80c9c-1100-4312-9123-4d34b840f2ef
#=╠═╡
simshow(Array(target[:, :,size(target,3)÷2]))
  ╠═╡ =#

# ╔═╡ 163a7676-51d5-43b9-8855-bc8dcb64c8ee
md"# 2. Set up Optimization
It is important to stick with Float32 datatypes for CUDA acceleration
"

# ╔═╡ a7e0922d-7895-4168-9bb3-1048dbf40673
angles = range(0, 2π, 401)[begin:end-1];

# ╔═╡ c86a820c-1861-4daf-b1cb-3b5a8ead5f9b
RO = RayOptics(togoc(angles), nothing)

# ╔═╡ 8dc3a31a-1f64-4e3d-9f4f-1a2933860d24
L = Threshold(thresholds=(0.7f0, 0.8f0))

# ╔═╡ fbadfd7e-9412-42da-af8a-20b4b1923505
@mytime patterns, printed, ores = optimize_patterns(target, RO, O, L)

# ╔═╡ 6092b668-56fa-46ab-bb91-3a45ffd3c658
ores

# ╔═╡ ccacbac5-d677-45ef-9d1d-bf5b63a24333
plot([r.value for r in ores.trace], yscale=:log10)

# ╔═╡ e6e7fc52-fc05-445b-ac27-313aef4432b9
md"# 3. Inspect Results"

# ╔═╡ beaffc0a-3059-4264-a781-13374aa9c322
@bind depth PlutoUI.Slider(1:size(patterns, 3), show_value=true, default=size(patterns,3) ÷ 2)

# ╔═╡ 0169fd94-5508-408d-8246-9bfd6f113487
#=╠═╡
[simshow(Array(printed[:, :, depth]), cmap=:jet) simshow(Array(printed[:, :, depth]) .> 0.75) simshow(Array(target[:, :, depth])) simshow(Array(target[:, :, depth]) .!= Array(printed[:, :, depth] .> 0.75))]
  ╠═╡ =#

# ╔═╡ 7278489a-ea23-4d55-91bf-dc05195e56f2
@bind angle PlutoUI.Slider(1:size(patterns, 2), show_value=true)

# ╔═╡ 3abce972-1906-4d75-a680-3d6311a4922a
simshow(Array(patterns[:, angle, :]), cmap=:turbo, set_one=true, γ=1)

# ╔═╡ bc38468d-4404-4d0b-9515-c6dda0b0bc44
#=╠═╡
p = plot_histogram(Array(target), Array(printed), (0.7, 0.8); yscale=:log10)
  ╠═╡ =#

# ╔═╡ a66d2c87-950c-4a28-8a1f-8e935bd510f7
md"# 4. OSMO Optimization
See: Rackson, Charles M., et al. Object-space optimization of tomographic reconstructions for additive manufacturing. Additive Manufacturing 48 (2021): 102367.
"

# ╔═╡ a62e7a84-7fc9-44ce-82e0-039128fe00a0
# ╠═╡ disabled = true
#=╠═╡
@mytime patterns2, printed2, ores2 = optimize_patterns(target, RO, OSMO(iterations=30, thresholds=(0.7f0, 0.8f0)))
  ╠═╡ =#

# ╔═╡ 44e5598d-52e7-4346-8bcc-229ab049a654
simshow(Array(patterns2[:, angle, :]), cmap=:turbo)

# ╔═╡ 1a46d06d-18ad-4780-8357-2dd66574c1f1
#=╠═╡
plot_histogram(Array(target), Array(printed2), (0.7, 0.8); yscale=:log10)
  ╠═╡ =#

# ╔═╡ 751a024a-ce8f-4900-915b-6db5620f8e52


# ╔═╡ d19e12eb-514f-4b0a-b197-d6813d24243e
md"# 5. Use Radon patterns in Wave Optics Forward model"

# ╔═╡ 030d6ed0-d4f3-4581-8afc-5dc8b5d6e7d9
L_x = 100f-6

# ╔═╡ f1e420a5-b319-469f-af20-b7b8a3ce4842
λ = 405f-9 / 1.5f0

# ╔═╡ 164f0d03-030a-4dae-aa94-24c2335bd88a
#=╠═╡
z = togoc(range(-L_x/2, L_x/2, size(target, 1)));
  ╠═╡ =#

# ╔═╡ 058a5892-283a-4208-8217-a4520e2ee4b8
patterns_reshape = select_region(CuArray(PermutedDimsArray(patterns, (1,3,2))), new_size=(size(patterns, 1), size(patterns,3) + 1, size(patterns, 2)));

# ╔═╡ 588885e9-5b0a-4e1b-96c1-0addf1728802
# ╠═╡ disabled = true
#=╠═╡
     x = similar(patterns_0, (size(patterns_0), 1)) 
     x .= fftpos(ps.L, size(x,1), CenterFT)
     mask = reshape((x.^2 .+ x'.^2) .<= ps.L^2, (1, size(x,1), size(x,1)))
          
     mask = exp()
  ╠═╡ =#

# ╔═╡ edc23eac-b631-4f31-a061-f4b1becc0bc5
#=╠═╡
begin
     #x = similar(patterns_0, (size(patterns_0), 1)) 
     #x .= fftpos(ps.L, size(x,1), CenterFT)
     #mask = reshape((x.^2 .+ x'.^2) .<= ps.L^2, (1, size(x,1), size(x,1)))
          
     #absorption = exp()
	
	 AS, _ = AngularSpectrum(patterns_reshape[:, :, 1] .+ 0im, z, λ, L_x, padding=false)
	 AS_abs2 = let target=target, AS=AS, langles=length(angles)
			 function AS_abs2(x)
				 abs2.(AS(x .+ 0im)[1]) ./ langles
			 end
	 end 
	#AS.HW .*= reshape(sqrt.(exp.(.- 2 .* z2 ./ L_x)), (1,1, size(z2, 1)))
	
	 fwd2 = let AS_abs2=AS_abs2, angles=angles
		 function fwd2(x)
			 SwissVAMyKnife.fwd_wave(x, AS_abs2, angles)
		 end
	 end 
end
  ╠═╡ =#

# ╔═╡ d992c283-64a5-4302-abd2-8fe9dc1023bb
#=╠═╡
z2 = togoc(range(0, L_x, size(target, 1)));
  ╠═╡ =#

# ╔═╡ 5b811fe3-0704-419a-9265-70271e775564
#=╠═╡
sum(abs2, AS.HW, dims=(1,2))
  ╠═╡ =#

# ╔═╡ d1172ce3-7d33-47f7-b571-b6e1d1f63719
 16384.0 /  2217.333

# ╔═╡ b6fbf8ac-8ceb-48de-b4bf-1599eb5835af
size(patterns_reshape)

# ╔═╡ ceed70dc-c062-4ae5-80b2-03d6ec3c9e45
#=╠═╡
out2 = fwd2(sqrt.(patterns_reshape));
  ╠═╡ =#

# ╔═╡ 1f13798c-859c-4626-a5a0-003f0068c92c
#=╠═╡
size(out2)
  ╠═╡ =#

# ╔═╡ d9cbd7a1-87b2-4f07-a8b7-bc540019e86b
@bind depth2 PlutoUI.Slider(1:size(patterns, 3), show_value=true, default=size(patterns,3) ÷ 2)

# ╔═╡ 116a0878-2b1f-48d1-af87-bee5f4b03800
#=╠═╡
[simshow(Array(out2[depth2, :, :]'), cmap=:jet) simshow(Array(out2[depth2, :, :]') .> 0.75) simshow(Array(target[:, :, depth2])) simshow(Array(target[:, :, depth2]) .!= Array(out2[depth2, :, :]' .> 0.75))]
  ╠═╡ =#

# ╔═╡ c883e5eb-be9c-4c26-8815-e9f1300f8b1f
md"$(round(L_x, digits=6))"

# ╔═╡ 5433a530-a9db-408c-911f-b69446f2c7e3
#=╠═╡
save("/tmp/figures/radon_threshold_wave_L=$(round(L_x, digits=6)).png", simshow(Array(out2[depth2, :, :]') .> 0.75))
  ╠═╡ =#

# ╔═╡ 0ef3f0df-a228-4f50-b3fe-f18c08e905bf
#=╠═╡
heatmap(Array(z), Array(z), Array(out2[depth2, :, :]') .> 0.75, aspect_ratio=:equal, size=(400,355))
  ╠═╡ =#

# ╔═╡ 4b5aa2b7-32f4-444d-bab2-5818189f95f7
#=╠═╡
heatmap(Array(z), Array(z), Array(target[:, :, depth2]) .> 0.75, aspect_ratio=:equal, size=(400,355))
  ╠═╡ =#

# ╔═╡ fc0d8d72-dd62-4861-b096-1f554cbdbae8
@bind iangle3 PlutoUI.Slider(1:size(angles, 1), show_value=true, default=50)

# ╔═╡ ceb2ecc6-cae8-4703-af9c-afcecaef02f8
#=╠═╡
fwd2_single_angle = let AS_abs2=AS_abs2, angles=angles[iangle3:iangle3]
	function fwd2(x)
	 SwissVAMyKnife.fwd_wave(x, AS_abs2, angles)
	end
end 
  ╠═╡ =#

# ╔═╡ 3f2536d8-1737-4bf1-8b68-473664c06fab
#=╠═╡
out3 = fwd2_single_angle(sqrt.(patterns_reshape[:, :,  iangle3:iangle3]));
  ╠═╡ =#

# ╔═╡ 47066d76-1e8b-473d-868c-1ebd743f87e5
#=╠═╡
simshow(Array(out3[depth2, :,  :]))
  ╠═╡ =#

# ╔═╡ 345cc956-ea78-4bf3-9f43-0c7db453cab9
#=╠═╡
target = togoc(load_benchy((230, 230, 230), (200, 200, 200), "/home/felix/Downloads/benchy/files/output_200//"));
  ╠═╡ =#

# ╔═╡ 9e0402cd-db92-41c1-98e0-3a7fd351ecd1
# ╠═╡ disabled = true
#=╠═╡
O = GradientBased(optimizer=Optim.Adam(alpha=0.1), options=Optim.Options(iterations=100, store_trace=true))
  ╠═╡ =#

# ╔═╡ 331b50f6-ead1-41f8-bab5-725eabe3b666
# ╠═╡ disabled = true
#=╠═╡
target = togoc(load_benchy((512, 512, 512), (480, 480, 480), "/home/felix/Downloads/benchy/files/output_480/"));
  ╠═╡ =#

# ╔═╡ e7b98330-f33e-4bb6-84c2-e3df1b572936
# ╠═╡ disabled = true
#=╠═╡
target = permutedims(togoc(load_benchy((80, 80, 80), (70, 70, 70), "/home/felix/Downloads/benchy/files/output_70/")), (3,2,1));
  ╠═╡ =#

# ╔═╡ 6c438e0a-00f3-45d6-a7e2-46cfa31cfc23
#=╠═╡
O = GradientBased(optimizer=Optim.LBFGS(), options=Optim.Options(iterations=30, store_trace=true))
  ╠═╡ =#

# ╔═╡ Cell order:
# ╟─0af67a5c-b689-44a9-999b-89f4c3dc40a0
# ╠═feea18a4-bf54-11ee-0a29-359b1cb4c099
# ╠═d25e6ac5-e9ad-408f-9855-c444ea98b73e
# ╠═8662f199-f816-4572-aa7e-7043bee5fca8
# ╟─1ac1d9f9-e864-464a-abf7-ec6a5faa8c98
# ╠═7ada448e-761b-4330-8f0a-a9b15767c2e6
# ╟─896ebd29-f858-425e-9bdd-9b1d1fae0d1f
# ╠═f507d697-d918-4b41-a99d-cc3aa1c2e7d3
# ╠═e7b98330-f33e-4bb6-84c2-e3df1b572936
# ╠═345cc956-ea78-4bf3-9f43-0c7db453cab9
# ╠═331b50f6-ead1-41f8-bab5-725eabe3b666
# ╠═1ee80c9c-1100-4312-9123-4d34b840f2ef
# ╟─163a7676-51d5-43b9-8855-bc8dcb64c8ee
# ╠═9e0402cd-db92-41c1-98e0-3a7fd351ecd1
# ╠═6c438e0a-00f3-45d6-a7e2-46cfa31cfc23
# ╠═a7e0922d-7895-4168-9bb3-1048dbf40673
# ╠═c86a820c-1861-4daf-b1cb-3b5a8ead5f9b
# ╠═8dc3a31a-1f64-4e3d-9f4f-1a2933860d24
# ╠═fbadfd7e-9412-42da-af8a-20b4b1923505
# ╠═6092b668-56fa-46ab-bb91-3a45ffd3c658
# ╠═ccacbac5-d677-45ef-9d1d-bf5b63a24333
# ╟─e6e7fc52-fc05-445b-ac27-313aef4432b9
# ╠═beaffc0a-3059-4264-a781-13374aa9c322
# ╠═0169fd94-5508-408d-8246-9bfd6f113487
# ╠═7278489a-ea23-4d55-91bf-dc05195e56f2
# ╠═3abce972-1906-4d75-a680-3d6311a4922a
# ╠═bc38468d-4404-4d0b-9515-c6dda0b0bc44
# ╟─a66d2c87-950c-4a28-8a1f-8e935bd510f7
# ╠═a62e7a84-7fc9-44ce-82e0-039128fe00a0
# ╠═44e5598d-52e7-4346-8bcc-229ab049a654
# ╠═1a46d06d-18ad-4780-8357-2dd66574c1f1
# ╠═751a024a-ce8f-4900-915b-6db5620f8e52
# ╟─d19e12eb-514f-4b0a-b197-d6813d24243e
# ╠═030d6ed0-d4f3-4581-8afc-5dc8b5d6e7d9
# ╠═f1e420a5-b319-469f-af20-b7b8a3ce4842
# ╠═164f0d03-030a-4dae-aa94-24c2335bd88a
# ╠═058a5892-283a-4208-8217-a4520e2ee4b8
# ╠═588885e9-5b0a-4e1b-96c1-0addf1728802
# ╠═edc23eac-b631-4f31-a061-f4b1becc0bc5
# ╠═d992c283-64a5-4302-abd2-8fe9dc1023bb
# ╠═5b811fe3-0704-419a-9265-70271e775564
# ╠═d1172ce3-7d33-47f7-b571-b6e1d1f63719
# ╠═b6fbf8ac-8ceb-48de-b4bf-1599eb5835af
# ╠═ceed70dc-c062-4ae5-80b2-03d6ec3c9e45
# ╠═1f13798c-859c-4626-a5a0-003f0068c92c
# ╠═d9cbd7a1-87b2-4f07-a8b7-bc540019e86b
# ╠═116a0878-2b1f-48d1-af87-bee5f4b03800
# ╠═c883e5eb-be9c-4c26-8815-e9f1300f8b1f
# ╠═5433a530-a9db-408c-911f-b69446f2c7e3
# ╠═0ef3f0df-a228-4f50-b3fe-f18c08e905bf
# ╠═4b5aa2b7-32f4-444d-bab2-5818189f95f7
# ╠═fc0d8d72-dd62-4861-b096-1f554cbdbae8
# ╠═ceb2ecc6-cae8-4703-af9c-afcecaef02f8
# ╠═3f2536d8-1737-4bf1-8b68-473664c06fab
# ╠═47066d76-1e8b-473d-868c-1ebd743f87e5
