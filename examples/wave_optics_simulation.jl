### A Pluto.jl notebook ###
# v0.19.36

using Markdown
using InteractiveUtils

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
		target[i, :, :] .= select_region(Gray.(load(joinpath(path, string("boat_", string(i, pad=3) ,".png")))), new_size=(sz_file))
	end

	target2 = zeros(Float32, sz)
	WaveOpticsPropagation.set_center!(target2, target)
	return togoc(target2)
end

# ╔═╡ 3e8e3d96-2f08-48b6-8bce-632bfff3267c
target = togoc(load_benchy((130, 130, 130), (100, 100, 100), "/home/felix/Downloads/benchy/files/output/"));

# ╔═╡ 0fc3a1f1-98b5-46ce-9adf-e31240876f4f
simshow(Array(target[:, :,30]))

# ╔═╡ 043edfd9-734b-4f50-9998-5f64b63dd209
md"# Set up Optimization
It is important to stick with Float32 datatypes for CUDA acceleration
"

# ╔═╡ bc212128-dde3-4835-a36d-00d2630a1ec4
n_resin = 1.5f0

# ╔═╡ 6c8b875e-19bd-4a66-9090-d356090f34df
angles = range(0, 2f0*π, 200)[1:end-1]

# ╔═╡ 518238ba-bfc9-4bc5-9544-8e59de8fcd51


# ╔═╡ afffcc37-b215-4c11-a51e-9435c79338c2
wo = WaveOptics(
	z=togoc(range(0, 500f-6, size(target,1))), 
	L=500f-6, 
	λ=405f-9 / n_resin, 
	μ=nothing, 
	angles=angles
)

# ╔═╡ b678d10d-ef6d-485e-b752-925f396d8e65
o = GradientBased(optimizer=Optim.LBFGS(),
			thresholds=(0.7f0, 0.8f0),
			iterations=50, loss=:object_space)

# ╔═╡ e07a894e-4939-43f8-8ab8-cc2a79fe04e6
Tuple{Float32, Float32}

# ╔═╡ a90c142a-bd1f-47aa-bbac-c12aee4ffbf9
@mytime patterns, printed, res = optimize_patterns(target, wo, o)

# ╔═╡ a978b980-2279-4118-8085-c714af60cae5
simshow(Array(patterns[:, 1, :]), cmap=:turbo)

# ╔═╡ 41d71bf9-d874-455d-8010-9d358bdde789
simshow(Array(printed[:, :, 30]) .> 0.75)

# ╔═╡ 9f58c6c2-57e7-4a3d-92b5-ab25caa9d012
plot_histogram(Array(target), Array(printed), (0.7, 0.8); yscale=:log10)

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
# ╠═e07a894e-4939-43f8-8ab8-cc2a79fe04e6
# ╠═a90c142a-bd1f-47aa-bbac-c12aee4ffbf9
# ╠═a978b980-2279-4118-8085-c714af60cae5
# ╠═41d71bf9-d874-455d-8010-9d358bdde789
# ╠═9f58c6c2-57e7-4a3d-92b5-ab25caa9d012
