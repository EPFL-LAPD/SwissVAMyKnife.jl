### A Pluto.jl notebook ###
# v0.19.40

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

# ╔═╡ 21958916-b59a-4e35-b28a-e6bc858ea091
begin
	using Pkg
	Pkg.activate(".")
	Pkg.instantiate()
	using Revise
end

# ╔═╡ 5fc3f3e0-1fa2-42e7-83eb-08c506ceba74
# this is our main package
using SwissVAMyKnife

# ╔═╡ cd28e366-ad68-44ef-8063-336363f4fa9b
using ImageShow, ImageIO, PlutoUI, IndexFunArrays, Optim, RadonKA, FileIO, Colors, Plots

# ╔═╡ 8c3af451-37dd-4e19-92c9-a7f1b756d7c9
using NDTools

# ╔═╡ 984baa5c-93a6-4e07-a19b-0f42ff58cb35
using CUDA

# ╔═╡ 8c3907a5-631e-4020-be39-bb5c19284f46
using NPZ

# ╔═╡ 66e44b7c-7614-45db-9551-b2d2ef561d78
TableOfContents()

# ╔═╡ b2529843-19bf-4d2e-9820-a1f819567c07
use_CUDA = Ref(true && CUDA.functional())

# ╔═╡ f8bb1a64-d95f-4a4f-ad7e-39d5bcd90c70
md" ## CUDA
CUDA accelerates the pattern generation easily by 5-20 times!
Otherwise most of the code will be multithreaded on your CPU but we strongly recommended the usage of CUDA for large scale 3D pattern generation.

Your CUDA is functional: **$(use_CUDA[])**
"

# ╔═╡ 659bf5b0-f762-4d3a-9c98-72781f0a1187
var"@mytime" = use_CUDA[] ? CUDA.var"@time" : var"@time"

# ╔═╡ 9e1614d9-7210-4071-a42b-ab8f88262852
togoc(x) = use_CUDA[] ? CuArray(x) : x

# ╔═╡ 2a8fce7a-90c1-470d-bac2-0b3b9ba29673
target = select_region(permutedims(npzread("/home/felix/Downloads/target_deconv.npy")[:, :, :, 1], (3,2,1))[:, :, 110:130], new_size=(270, 270, 20));

# ╔═╡ 03b0892a-e1c9-4d4c-ac88-80a54be680de
simshow(target[:, :, 10])

# ╔═╡ a81d54f1-be29-48b2-ac2b-184e86c6150f
any(isnan.(target))

# ╔═╡ cbb06ff8-dcf5-4d24-b2f7-2c6721a4596a
angles = range(0, 2π, 400)

# ╔═╡ b3b3a86c-56b1-4bbd-9269-a6b2bb38cf41
μ = nothing

# ╔═╡ 5da75f58-cdbf-4677-95e3-ddf6182fe8e2
geometry = ParallelRayOptics(angles, μ)

# ╔═╡ 9aa33730-b3c2-4399-98c5-061ff54302c7
md" As optimizer we use a gradient descent based variant"

# ╔═╡ e4b5f6f6-aebc-4a18-87ce-555630c949c1
optimizer = GradientBased(optimizer=Optim.Adam(alpha=10.0), options=Optim.Options(iterations=10, store_trace=true))

# ╔═╡ 9327f476-a339-43bf-a279-3ea285382602
md"
Let's try to run the optimization: `togoc` moves the target onto the CUDA device, if possible"

# ╔═╡ 92f938a7-a89f-4e5a-b859-ee15e102be5a
@mytime patterns, printed_intensity, optim_res = optimize_patterns(togoc(target), geometry, optimizer, loss)

# ╔═╡ 9cc4060e-6fe2-427a-b04d-8257b657d88a
loss(printed_intensity, togoc(target), nothing)

# ╔═╡ bd14f5f7-8cdc-491d-84f9-a365452e2bd1
sum(patterns)

# ╔═╡ 821301ac-6e37-4e22-a0b2-79da2f1a2a61
optim_res

# ╔═╡ 9693eb50-670f-4bc4-b3d8-2ed6dfae238d
md"z slider value $(@bind slice2 PlutoUI.Slider(axes(target, 3), show_value=true, default=0.5))"

# ╔═╡ 4adf1d37-e464-4b44-9c5f-8379edefb0c6
plot(optim_res.trace)

# ╔═╡ b1c86ece-4a42-4614-9b83-2e57a474567f
plot([v.value for v in optim_res.trace])

# ╔═╡ 42cd201c-d703-4175-91dc-0fa38f55e3be
sum(patterns)

# ╔═╡ 589a7ce2-615c-4a0d-93e8-ae4c89bdcda3
Revise.errors()

# ╔═╡ 3362632a-4bf2-4a1e-bd89-1851e56b0b0f
[simshow(Array((printed_intensity[:, :, slice2]))) simshow(ones((size(target, 1), 5))) simshow(ones((size(target, 1), 5))) simshow(target[:, :, slice2])]

# ╔═╡ 95966605-5314-4954-b354-13a6842ce0f2
heatmap((target[:, :, slice2]) ./ Array(1f-8 .+ printed_intensity[:, :, slice2]))

# ╔═╡ 2b434cbb-21a0-468c-b8d5-ba62972d411d
plot_intensity_histogram(target .> 0, printed_intensity, (0.0, 1.0), xlim=(0.0, 1.3))

# ╔═╡ 038f3414-abf2-42b1-83b7-d803742a4d18
md"angle $(@bind angle PlutoUI.Slider(axes(patterns, 2), show_value=true, default=0.5))"

# ╔═╡ 9c9be83f-b178-4b2f-b3f7-2071feb11273
simshow(Array(abs.(patterns[:, angle, :])))

# ╔═╡ 3bccab49-7b01-40cc-9717-77b2f8b0493b
loss = LossGrayScale()

# ╔═╡ ad26faf5-6039-4b45-ab71-20e45b3f0618
# ╠═╡ disabled = true
#=╠═╡
loss = LossThreshold(thresholds=(0.85, 0.95))
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═21958916-b59a-4e35-b28a-e6bc858ea091
# ╠═5fc3f3e0-1fa2-42e7-83eb-08c506ceba74
# ╠═cd28e366-ad68-44ef-8063-336363f4fa9b
# ╠═8c3af451-37dd-4e19-92c9-a7f1b756d7c9
# ╠═66e44b7c-7614-45db-9551-b2d2ef561d78
# ╟─f8bb1a64-d95f-4a4f-ad7e-39d5bcd90c70
# ╠═984baa5c-93a6-4e07-a19b-0f42ff58cb35
# ╠═b2529843-19bf-4d2e-9820-a1f819567c07
# ╠═659bf5b0-f762-4d3a-9c98-72781f0a1187
# ╠═9e1614d9-7210-4071-a42b-ab8f88262852
# ╠═8c3907a5-631e-4020-be39-bb5c19284f46
# ╠═2a8fce7a-90c1-470d-bac2-0b3b9ba29673
# ╠═03b0892a-e1c9-4d4c-ac88-80a54be680de
# ╠═a81d54f1-be29-48b2-ac2b-184e86c6150f
# ╠═cbb06ff8-dcf5-4d24-b2f7-2c6721a4596a
# ╠═b3b3a86c-56b1-4bbd-9269-a6b2bb38cf41
# ╠═5da75f58-cdbf-4677-95e3-ddf6182fe8e2
# ╠═9aa33730-b3c2-4399-98c5-061ff54302c7
# ╠═ad26faf5-6039-4b45-ab71-20e45b3f0618
# ╠═e4b5f6f6-aebc-4a18-87ce-555630c949c1
# ╠═3bccab49-7b01-40cc-9717-77b2f8b0493b
# ╟─9327f476-a339-43bf-a279-3ea285382602
# ╠═9cc4060e-6fe2-427a-b04d-8257b657d88a
# ╠═92f938a7-a89f-4e5a-b859-ee15e102be5a
# ╠═bd14f5f7-8cdc-491d-84f9-a365452e2bd1
# ╠═821301ac-6e37-4e22-a0b2-79da2f1a2a61
# ╠═9693eb50-670f-4bc4-b3d8-2ed6dfae238d
# ╠═4adf1d37-e464-4b44-9c5f-8379edefb0c6
# ╠═b1c86ece-4a42-4614-9b83-2e57a474567f
# ╠═42cd201c-d703-4175-91dc-0fa38f55e3be
# ╠═589a7ce2-615c-4a0d-93e8-ae4c89bdcda3
# ╠═3362632a-4bf2-4a1e-bd89-1851e56b0b0f
# ╠═95966605-5314-4954-b354-13a6842ce0f2
# ╠═2b434cbb-21a0-468c-b8d5-ba62972d411d
# ╠═038f3414-abf2-42b1-83b7-d803742a4d18
# ╠═9c9be83f-b178-4b2f-b3f7-2071feb11273
