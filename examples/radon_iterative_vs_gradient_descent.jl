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

# ╔═╡ ef41c478-ae20-11ee-1ba6-ff04bb6e0dd5
begin
	using Pkg
	Pkg.activate(".")
	Pkg.instantiate()
	using Revise
end

# ╔═╡ 96d33037-f981-4c71-af8d-e1f6f6e70edf
using SwissVAMyKnife, WaveOpticsPropagation,  ImageShow, ImageIO, PlutoUI, IndexFunArrays, Optim, RadonKA, FileIO, Plots, NDTools, CUDA

# ╔═╡ 13a6f5c9-1036-4f6b-8408-989f691d2971
using Optimization, OptimizationOptimisers, OptimizationOptimJL

# ╔═╡ 1c6cfb5e-f969-4ff8-92b8-9effbe577afc
begin
	# use CUDA if functional
	use_CUDA = Ref(true && CUDA.functional())
	var"@mytime" = use_CUDA[] ? CUDA.var"@time" : var"@time"    
	togoc(x) = use_CUDA[] ? CuArray(x) : x
end

# ╔═╡ 563dab7f-fbc2-4e62-bb31-89f4a58aa2ed
function load_benchy(sz, sz_file, path)
	target = zeros(Float32, sz_file)

	for i in 1:sz_file[1]
		target[i, :, :] .= Gray.(load(joinpath(path, string("boat_", string(i, pad=3) ,".png"))))
	end

	target2 = zeros(Float32, sz)
	WaveOpticsPropagation.set_center!(target2, target)
	return togoc(target2)
end

# ╔═╡ 802f6620-cdca-4abf-b192-f2970a09b80e
md"# Specify target"

# ╔═╡ 04effcee-3ac9-408e-b024-e610253ea1dc
target = togoc(load_benchy((230, 230, 230), (200, 200, 200), "/home/felix/Downloads/benchy/files/output/"));

# ╔═╡ 3c26e0fc-38bb-4365-9dc7-3f3a4487f1fb
angles = range(0f0, 2f0 * π, round(Int, 800))

# ╔═╡ 2acdb1cf-3f14-46d6-8b95-80262f14797e
size(target, 1) * π

# ╔═╡ b5554eb7-a576-4fb0-b985-c1159a0dc23d
opt_f = let target=target, 
			angles=angles, 
			isobject= target .== 1,
			notobject = target .== 0,
			thresholds=(0.6f0, 0.7f0)
	opt_f(x, p) = begin
		f = iradon(abs2.(x), angles)
		f = f ./ maximum(f)
		return (sum(abs2, max.(0, thresholds[2] .- view(f, isobject))) + sum(abs2, max.(0, view(f, notobject) .- thresholds[1])))
	end
end

# ╔═╡ 21986b64-5dfd-45b4-ab47-bf6d3c3fe4c7
opt_fun = OptimizationFunction(opt_f, AutoZygote())

# ╔═╡ e7a7b501-3705-48af-98ea-24e363e33035
@mytime init0 = sqrt.(radon(target, angles));

# ╔═╡ 7aacfd00-3431-4beb-b32d-aa4b57f4a814
@mytime problem = OptimizationProblem(opt_fun, init0);

# ╔═╡ 1f12326d-4937-4611-80c8-8d4429a49237
res = solve(problem, OptimizationOptimisers.Adam(2), maxiters=100)

# ╔═╡ 512ca16f-ee8e-4410-8b79-99c385f0dd39
@mytime patterns, printed_i, res2 = optimize_patterns(target, angles, iterations=100,
											method=:radon_iterative,
											thresholds=(0.6f0, 0.7f0),
											μ=nothing)

# ╔═╡ 098e4ce8-7a35-4a44-96ea-57f7fdadf4e3
md"# Conclusion
Seems like that OSMO is faster
"

# ╔═╡ e3fc69fe-418f-4e66-bbf3-314da05262c9
@bind angle Slider(1:size(angles, 1), show_value=true)

# ╔═╡ df7fdfb3-d5d5-4e10-a2cf-0a7752a9b491
simshow(Array(abs2.(res.u[:, angle, :])))

# ╔═╡ 8d0b3988-2bde-4efe-a68f-5133192135a3
simshow(Array(patterns[:, angle, :]))

# ╔═╡ 581da7db-c4cc-4d6f-8e12-dc13b88e9d4e
simshow(Array(abs2.(init0))[:, angle, :])

# ╔═╡ 8cfece4f-9be1-48af-95e8-37a5641e6607
extrema(res.u)

# ╔═╡ cbe32a71-ebea-49aa-b6ce-a1ec2f9b6551
printed = Array(iradon(abs2.(res.u), angles));

# ╔═╡ 1d82998e-b293-4270-a1a3-4c971cab8519
printed ./= maximum(printed);

# ╔═╡ f1360ccf-6a24-40bc-a01c-95fb1db6599b
simshow(Array(printed[:, :, 50] .> 0.65))

# ╔═╡ aad7f8a0-db9a-454d-ba8a-01315282a252
plot_intensity(Array(target), Array(printed), (0.6, 0.7))

# ╔═╡ 06386993-e031-4a6b-ae55-8a95cc949d92
plot_intensity(Array(target), Array(printed_i), (0.6, 0.7))

# ╔═╡ Cell order:
# ╠═ef41c478-ae20-11ee-1ba6-ff04bb6e0dd5
# ╠═96d33037-f981-4c71-af8d-e1f6f6e70edf
# ╠═13a6f5c9-1036-4f6b-8408-989f691d2971
# ╠═1c6cfb5e-f969-4ff8-92b8-9effbe577afc
# ╠═563dab7f-fbc2-4e62-bb31-89f4a58aa2ed
# ╠═802f6620-cdca-4abf-b192-f2970a09b80e
# ╠═04effcee-3ac9-408e-b024-e610253ea1dc
# ╠═3c26e0fc-38bb-4365-9dc7-3f3a4487f1fb
# ╠═2acdb1cf-3f14-46d6-8b95-80262f14797e
# ╠═b5554eb7-a576-4fb0-b985-c1159a0dc23d
# ╠═21986b64-5dfd-45b4-ab47-bf6d3c3fe4c7
# ╠═e7a7b501-3705-48af-98ea-24e363e33035
# ╠═7aacfd00-3431-4beb-b32d-aa4b57f4a814
# ╠═1f12326d-4937-4611-80c8-8d4429a49237
# ╠═512ca16f-ee8e-4410-8b79-99c385f0dd39
# ╠═098e4ce8-7a35-4a44-96ea-57f7fdadf4e3
# ╠═e3fc69fe-418f-4e66-bbf3-314da05262c9
# ╠═df7fdfb3-d5d5-4e10-a2cf-0a7752a9b491
# ╠═8d0b3988-2bde-4efe-a68f-5133192135a3
# ╠═581da7db-c4cc-4d6f-8e12-dc13b88e9d4e
# ╠═8cfece4f-9be1-48af-95e8-37a5641e6607
# ╠═cbe32a71-ebea-49aa-b6ce-a1ec2f9b6551
# ╠═1d82998e-b293-4270-a1a3-4c971cab8519
# ╠═f1360ccf-6a24-40bc-a01c-95fb1db6599b
# ╠═aad7f8a0-db9a-454d-ba8a-01315282a252
# ╠═06386993-e031-4a6b-ae55-8a95cc949d92
