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

# ╔═╡ 55c4ebc4-489d-453b-bf71-c49362bb871b
Pkg.develop(path="../../SwissVAMyKnife.jl/")

# ╔═╡ 6f8d4329-9b45-410b-98a2-17cb22475060
using SwissVAMyKnife

# ╔═╡ f4772bab-4700-443f-9e86-7fb35b616551
using ImageShow, ImageIO, PlutoUI, IndexFunArrays, Optim

# ╔═╡ 2a21b5cd-9a73-4da8-b5b6-2a59aeefaf03
using Plots

# ╔═╡ 22c4be0e-555b-4ed6-b4a0-1062e539f498
target = zeros(Float32, (200, 200, 1));

# ╔═╡ a6fdcf6f-24dc-44cd-996f-8eb8f5b651af
begin
	target .= box(target, (100, 100, 1))
	target .-= box(target, (70, 70, 1))
end;

# ╔═╡ f90cf4dd-6844-4515-a877-19b0d6a07f3f
simshow(target[:, :, 1])

# ╔═╡ fd7a91df-5edd-479d-a810-3f30e38a2a21
angles = range(0f0, 1f0 * π, 100)	

# ╔═╡ f4d93076-8728-46f3-bc8d-da42b9024e3c
thresholds = (0.65f0, 0.75f0)

# ╔═╡ f5d4ab19-55fb-4408-9f97-a9e98278a222
@time patterns, printed_i, res = optimize_patterns(target, angles, iterations=40,
											 optimizer=LBFGS(),
											thresholds=thresholds,
											μ=1/100f0)

# ╔═╡ 9db3f536-34ea-4738-830e-36eceef97b22
Revise.retry()

# ╔═╡ 1ecb3d90-4b50-4107-92c7-f2474675bbb8
res

# ╔═╡ 463b7a89-4df0-4442-8d6e-2b6b251fbc1b
plot([a.value for a in res.trace], yscale=:log10)

# ╔═╡ c443703d-cd2a-4e2a-bfd0-742b4fcb001f
SwissVAMyKnife.plot_intensity(target, printed_i, thresholds)

# ╔═╡ 0ac5bb39-f621-4c94-8e5b-5b148f7221a2
@bind threshold Slider(0.0:0.01:1, show_value=true)

# ╔═╡ 347a0bc9-e387-4dab-a74c-3ff5d5f1a33a
[simshow(printed_i[:,:, 1], cmap=:turbo) simshow(printed_i[:, :, 1] .> threshold)]

# ╔═╡ 2e8caf8a-d2d2-4326-b089-70ef18d74630
simshow(patterns[:, :, 1], cmap=:turbo)

# ╔═╡ fc5c2bf9-9aa1-4391-82f0-bf7c538d6985
sum(patterns) / length(patterns)

# ╔═╡ e46dca16-bfc8-4776-a3fa-febe057da107
histogram(patterns[:], yscale=:log10)

# ╔═╡ Cell order:
# ╠═30aab62e-a0eb-11ee-06db-11eb60e1a051
# ╠═6f8d4329-9b45-410b-98a2-17cb22475060
# ╠═55c4ebc4-489d-453b-bf71-c49362bb871b
# ╠═f4772bab-4700-443f-9e86-7fb35b616551
# ╠═2a21b5cd-9a73-4da8-b5b6-2a59aeefaf03
# ╠═22c4be0e-555b-4ed6-b4a0-1062e539f498
# ╠═a6fdcf6f-24dc-44cd-996f-8eb8f5b651af
# ╠═f90cf4dd-6844-4515-a877-19b0d6a07f3f
# ╠═fd7a91df-5edd-479d-a810-3f30e38a2a21
# ╠═f4d93076-8728-46f3-bc8d-da42b9024e3c
# ╠═f5d4ab19-55fb-4408-9f97-a9e98278a222
# ╠═9db3f536-34ea-4738-830e-36eceef97b22
# ╠═1ecb3d90-4b50-4107-92c7-f2474675bbb8
# ╠═463b7a89-4df0-4442-8d6e-2b6b251fbc1b
# ╠═c443703d-cd2a-4e2a-bfd0-742b4fcb001f
# ╠═0ac5bb39-f621-4c94-8e5b-5b148f7221a2
# ╠═347a0bc9-e387-4dab-a74c-3ff5d5f1a33a
# ╠═2e8caf8a-d2d2-4326-b089-70ef18d74630
# ╠═fc5c2bf9-9aa1-4391-82f0-bf7c538d6985
# ╠═e46dca16-bfc8-4776-a3fa-febe057da107
