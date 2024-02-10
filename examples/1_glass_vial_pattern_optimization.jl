### A Pluto.jl notebook ###
# v0.19.38

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

# ╔═╡ fb4bb1e2-c5e3-11ee-2f22-0bf5f621e215
begin
	using Pkg
	Pkg.activate(".")
	Pkg.instantiate()
	using Revise
end

# ╔═╡ 2487fdcc-f8c0-426d-9eff-ca9c8ce98100
# this is our main package
using SwissVAMyKnife

# ╔═╡ 53c94ae4-3a86-4f3b-8e35-832859c5b465
using ImageShow, ImageIO, PlutoUI, IndexFunArrays, Optim, RadonKA, FileIO, Colors, Plots

# ╔═╡ 2dc23db0-87b8-4044-9408-b2ffcde33ee8
using CUDA

# ╔═╡ 4b946954-9432-4ffe-9b4d-8c590be96906
using Statistics

# ╔═╡ 9c478480-163a-4fe0-ad6b-32a337a616b9
md"# 0. Load packages
On the first run, Julia is going to install some packages automatically. So start this notebook and give it some minutes (5-10min) to install all packages. 
No worries, any future runs will be much faster to start!
"

# ╔═╡ daf8043e-293f-495c-9b0d-b4e2b50de4a7
TableOfContents()

# ╔═╡ 9ec75d19-98be-4e04-990e-fd82b967525c
use_CUDA = Ref(true && CUDA.functional())

# ╔═╡ de126e87-55c7-44ef-935d-b4bd96591914
md" ## CUDA
CUDA accelerates the pattern generation easily by 5-20 times!
Otherwise most of the code will be multithreaded on your CPU but we strongly recommended the usage of CUDA for large scale 3D pattern generation.

Your CUDA is functional: **$(use_CUDA[])**
"

# ╔═╡ bd516459-0be9-4ccb-a95b-32d07c15b410
var"@mytime" = use_CUDA[] ? CUDA.var"@time" : var"@time"

# ╔═╡ 9d1ffd4d-1fe3-4f5c-bdc3-817fd0f23c7b
togoc(x) = use_CUDA[] ? CuArray(x) : x

# ╔═╡ 465e830f-ca33-427a-ad7b-e18f3bbaa3c4
md"# 1. Let's start with a simple 2D pattern
However, we add a trailing 1 z dimension.
"

# ╔═╡ 4475e0c5-4ae6-403f-bb2b-c5bc8a25d18a
sz = (256, 256, 1)

# ╔═╡ cec09ba8-5707-495f-b4ac-94f346c33ce5


# ╔═╡ b2dae3ea-5a40-498f-a134-5a9a98e58de0
target = box(Float32, sz, (150, 110, 1)) .-  box(Float32, sz, (80, 50, 1));

# ╔═╡ 047023bf-96dd-4427-aa13-3fb2340af90e
simshow(target[:, :, 1])

# ╔═╡ 1d2bdd44-95d2-45ef-8e24-c8ca3df7485a
md"# 2. Simple optimization


## Loss Function
One simple optimization algorithm tries to distribute the intensity as following.
Object pixels should receive enough intensity to polymerize.
Void pixels should stay below this intensity threshold and should not polymerize.
Also, object pixels should not strongly over polymerize

Let's use the `LossThreshold` to achieve this:
"

# ╔═╡ ae3fdae9-7b3c-4391-8fff-530f7758942c
loss = LossThreshold(thresholds=(0.65, 0.75))

# ╔═╡ b58c9e1f-7ca6-4171-8004-2b84d33cf335
md"## Specify Geometry

First we try with a parallel geometry. Meaning that the rays are not refracted upon entry into the glass vial.
This corresponds to the situation with an index matching bath!

For this optimization we use 400 angles in the range from 0 to 2π
As absorption we set $μ=nothing$, so we ignore absorption first!
"

# ╔═╡ 0dd200a7-a0f0-416e-ad69-7343e210a36e
angles = range(0, 2π, 400)

# ╔═╡ 303a363f-f941-45b9-9194-62a2c411964a
μ = nothing

# ╔═╡ 6d130809-8b78-485c-bba1-fc958bc9d873
geometry = ParallelRayOptics(angles, μ)

# ╔═╡ cb8b9ccc-4085-4abe-bd71-5e70cc8decfa
md" As optimizer we use a gradient descent based variant"

# ╔═╡ e5f0660d-2adb-4c82-8945-e8164ec500f6
optimizer = GradientBased(optimizer=Optim.LBFGS(), options=Optim.Options(iterations=10, store_trace=true))

# ╔═╡ b767afa8-784c-453b-ad0d-a1daf4d5ba5c
md"
Let's try to run the optimization: `togoc` moves the target onto the CUDA device, if possible"

# ╔═╡ 07e97af6-f30b-4100-999c-c61b52cec76e
@mytime patterns, printed_intensity, optim_res = optimize_patterns(togoc(target), geometry, optimizer, loss)

# ╔═╡ c13abe2d-ff27-42e2-9534-cb1936e68236
optim_res

# ╔═╡ 9887da10-96fa-4509-9b91-a6c21354f4ba
md"# 3. Look at the results"

# ╔═╡ f79bdf14-fefa-449f-b13a-49e6fb9153bf
md"We can inspect the histogram of the optimization.

We see, void pixels receive less intensity than the threshold.
Object pixels stay well above it
"

# ╔═╡ 66560fd0-3cee-41ca-bb13-dee2a78bbb9b
plot_intensity_histogram(target, printed_intensity, (0.65, 0.75))

# ╔═╡ dc538ea6-54e3-4a0f-a5ef-427802239cf2
md"threshold value $(@bind thresh PlutoUI.Slider(0:0.01:1, show_value=true, default=0.5))"

# ╔═╡ 71333d50-fa28-4bdf-bad4-d787682e043a
md"

Printed intensity ----------------------------- Printed intensity after threshold ------------------- target 
"

# ╔═╡ 5801f61a-79fa-4326-82dd-3901dd4d493a
[simshow(Array(printed_intensity[:, :, 1])) simshow(ones((sz[1], 5))) simshow(thresh .< Array(printed_intensity[:, :, 1])) simshow(ones((sz[1], 5))) simshow(target[:, :, 1])]

# ╔═╡ 92f5b987-13c7-463c-b150-71c51ec4aa72
simshow(Array(patterns)[:, :, 1])

# ╔═╡ 11af6da4-5fa2-4e48-a1c8-32bc686b8239
md"# 4. Include Absorption of the Photo initiator
Let's add some absorption!

256 is the size of the object in pixels.
A $μ=2/(256)$ would mean that after propagation through the vial, the ray has only $I(x) = I_0 \cdot \exp(-2) ≈ I_0 \cdot 0.135$ of the initial intensity left.

And the printing is still possible!
Empirically, a factor of 3 might be possible to print. A higher $\mu$ means actually shorter printing times.
"

# ╔═╡ 2f51ec68-09c6-4475-81b9-ec252fba5df7
μ2 = 2 / (256)

# ╔═╡ 2e76c088-e61f-4deb-ac69-b73d6e3f361c
geometry_μ = ParallelRayOptics(angles, μ2)

# ╔═╡ 0e4da85b-99bd-47e0-a8fe-3c69c3a05708
@mytime patterns_μ, printed_intensity_μ, optim_res_μ = optimize_patterns(togoc(target), geometry_μ, 
								GradientBased(optimizer=Optim.LBFGS(), options=Optim.Options(iterations=10, store_trace=true))					
								, loss)

# ╔═╡ a04d12c2-f718-44e0-a755-da1132c0330e
plot_intensity_histogram(target, printed_intensity_μ, (0.65, 0.75))

# ╔═╡ 5ade07d4-e7b3-4e78-94fa-454924a4066c
@bind thresh2 PlutoUI.Slider(0:0.01:1, show_value=true, default=0.5)

# ╔═╡ ccd0ad5e-5c01-4ec0-afc4-0dcce85dc9ef
[simshow(Array(printed_intensity_μ[:, :, 1])) simshow(ones((sz[1], 5))) simshow(thresh2 .< Array(printed_intensity_μ[:, :, 1])) simshow(ones((sz[1], 5))) simshow(target[:, :, 1])]

# ╔═╡ 17621d91-1f93-4fcf-abf1-5e97463a0bdf
simshow(Array(patterns_μ[:,:,1]), cmap=:gray)

# ╔═╡ 05c6a38e-a87d-46da-acb6-33b393369f4c
sum(patterns_μ) / (maximum(patterns_μ) * length(patterns))

# ╔═╡ f940c2a6-ebc2-4dca-986c-fe25bfa9e4f0
md"""# 5. Include refraction of the glass vial 

Printing without index matching bath is more challenging but still possible.
The glass vial changes the direction.
Because of our flexible [RadonKA.jl](https://github.com/roflmaostc/RadonKA.jl) package, we can simulate those scenarios.

Analytically, we can derive the ray direction after propagation through a glass interface of the vial and then entering the resin.

Luckily, you just have to specify the following parameters for the geometry:
 
 - `angles` is a range or `Vector` (or `CuVector`) storing the illumination angles.
 - `μ` is the absorption coefficient of the resin in units of pixels.
 - So `μ=0.1` means that after ten pixels of propagation the intensity is `I(10) = I_0 * exp(-10 * 0.1)`.
 - `R_outer` is the outer radius of the glass vial.
 - `R_inner` is the inner radius of the glass vial.
 - `n_vial` is the refractive index of the glass vial.
 - `n_resin` is the refractive index of the resin.
"""

# ╔═╡ f9a82207-6841-44c9-9aec-e8da3de8e0b0
geometry_vial = VialRayOptics(
	angles=angles,
	μ=2/256,
	R_outer=8e-3,
	R_inner=7.5e-3,
	n_vial=1.5,
	n_resin=1.48
)

# ╔═╡ 675764cb-721a-4b89-92c4-8a1ab7f5867f
@mytime patterns_vial, printed_intensity_vial, optim_res_vial = optimize_patterns(togoc(target), geometry_vial, 
								GradientBased(optimizer=Optim.LBFGS(), options=Optim.Options(iterations=8, store_trace=true))					
								, loss)

# ╔═╡ 9ddd098d-2d78-4de8-a322-40a2463adcda
plot_intensity_histogram(target, printed_intensity_vial, (0.65, 0.75))

# ╔═╡ d1f33cd0-0927-417e-a024-6a8c76da1fb9


# ╔═╡ d56eb3ab-4698-4ee4-aae1-461b964b778c
@bind thresh3 PlutoUI.Slider(0:0.01:1, show_value=true, default=0.5)

# ╔═╡ b6a6237f-b03f-402d-9693-f00044e3539e
[simshow(Array(printed_intensity_vial[:, :, 1])) simshow(ones((sz[1], 5))) simshow(thresh3 .< Array(printed_intensity_vial[:, :, 1])) simshow(ones((sz[1], 5))) simshow(target[:, :, 1])]

# ╔═╡ eda29eed-507c-4f55-b1e6-590c9616db8e
md"
The sinogram looks very similar to the one with an index matching bath.
Some people do post-optimization interpolation.
However, this is not exactly if absorption takes place.
"

# ╔═╡ fc239b37-dc88-48cd-9479-99074d96ece5
simshow(Array(patterns_vial[:,:,1]), cmap=:turbo)

# ╔═╡ 7de5d398-20d6-44e2-b2eb-3007111e146f
sum(patterns_vial) / (maximum(patterns_vial) * length(patterns_vial))

# ╔═╡ a7cafba6-849a-4eb8-9709-a76cb98e9879
md"# 6. Let's do a bigger 3D object!
Be a little patient, this might take ~30s to a minute on a GPU.
On a CPU much longer accordingly!

We deactive the cell by default. However of the (...) next to the cell and click *Enable Cell*

"

# ╔═╡ 0a655d51-e3b6-413b-83de-9781974242a2
begin
	target_3D = box(Float32, (256, 256, 256), (80, 80, 100)) .-  box(Float32, (256, 256, 256), (60, 50, 80));
	target_3D = Float32.(Bool.(target_3D) .|| (rr2(size(target_3D)) .< 30^2))
end;

# ╔═╡ 877b1484-969c-45d7-a3e7-4f0301a81a4b
md"z slide value $(@bind slice PlutoUI.Slider(axes(target_3D, 3), show_value=true, default=0.5))"

# ╔═╡ a05d8dd4-40c3-41da-9dbd-f58ab161b2d4
simshow(target_3D[:, :, slice])

# ╔═╡ 8134de5c-4bc2-498d-af0e-06ed1a1e8539


# ╔═╡ d6a59254-bffe-4116-9335-8884eb44556f
@mytime patterns_3D, printed_intensity_3D, optim_res_3D = optimize_patterns(togoc(target_3D), geometry_vial, 
								GradientBased(optimizer=Optim.LBFGS(), options=Optim.Options(iterations=10, store_trace=true))					
								, loss)

# ╔═╡ 329e47a6-9778-43a8-b74f-74f0d59458de


# ╔═╡ 6009b601-6988-4be8-a519-7c59660c73ab
optim_res_3D

# ╔═╡ f15224d2-6eb6-4919-aa79-30a4728edc20
Revise.errors()

# ╔═╡ 1ab55d13-5c68-40fe-875b-c29e4298f936
SwissVAMyKnife.Zygote.refresh()

# ╔═╡ 85d9c76b-9ee9-4b81-b9a8-6597309fc999


# ╔═╡ 1ba0adc2-8dbe-4f25-9746-cb344e9fa7c7


# ╔═╡ 9af3c5ac-934c-4d07-95ee-5b2f947b0e22
CuArray{Bool}([1,1]) .- CuArray{Int32}([1, 0])

# ╔═╡ 5607deac-8814-4810-a009-14dc223c8fa7
CuArray{Bool}([1, 0])

# ╔═╡ 2be5a610-e553-4dcb-833f-6febc813f324


# ╔═╡ 4d0d1f34-1068-427b-b2b7-3966b533b6d6


# ╔═╡ 0c917ace-401c-447c-b021-23aa401be718
CuArray{Bool}(CuArray([1f0, 0f0]))

# ╔═╡ fa45b520-18a8-4465-b642-a1c08de48e20
@bind thresh4 PlutoUI.Slider(0:0.01:1, show_value=true, default=0.5)

# ╔═╡ 10cc0e79-5d5b-43cf-afb8-ad21875d6d97
md"z slider value $(@bind slice2 PlutoUI.Slider(axes(target_3D, 3), show_value=true, default=0.5))"

# ╔═╡ 0835d1b0-a562-4edf-939e-3044f94e53c7
[simshow(Array(printed_intensity_3D[:, :, slice2]), set_one=false) simshow(ones((size(target_3D, 1), 5))) simshow(thresh4 .< Array(printed_intensity_3D[:, :, slice2])) simshow(ones((size(target_3D, 1), 5))) simshow(target_3D[:, :, slice2])]

# ╔═╡ 83793ffe-38de-4a76-b80f-8dee1d18c5a5
plot_intensity_histogram(target_3D, printed_intensity_3D, (0.65, 0.75))

# ╔═╡ 262b96e8-a78c-441c-9b46-2d87636286d7
md"angle $(@bind angle PlutoUI.Slider(axes(patterns_3D, 2), show_value=true, default=0.5))"

# ╔═╡ c8774a2b-47e0-4be6-b213-bdef6d7b0726
simshow(Array(patterns_3D[:,angle,:]), cmap=:turbo, set_one=true)

# ╔═╡ 497cd20f-5544-43dc-8037-1263afb5cf46


# ╔═╡ 3d4e99a1-ebf3-4b51-a3d9-8ada7fb3267c
sz2 = (32, 32, 2)

# ╔═╡ b99cbb89-6dd3-4832-ac66-d24fbc3fc060
target2 = box(Float32, sz2, (17, 17, 1)) .-  box(Float32, sz2, (9, 9, 1));

# ╔═╡ 3fadfd32-5552-43ea-9bcb-725a1a78a00d
simshow(target2[:,:,1])

# ╔═╡ f50ade1d-37f1-4f03-a081-e74642798124
angles2 = range(0, 2π, 64)

# ╔═╡ 510bd5b1-f91f-4f7b-8a84-f8eebacb9aff
optimizer2 = GradientBased(optimizer=Optim.LBFGS(), options=Optim.Options(iterations=10, store_trace=true))

# ╔═╡ fa639ca1-cf25-41d0-9a29-8c325d808ba4
geometry2 = ParallelRayOptics(angles2, nothing)

# ╔═╡ 47f86ac2-a688-40c3-a4c3-fd750b698c27
target == (0.7 .< optimize_patterns((target), geometry2, optimizer2, loss2)[2])

# ╔═╡ 8e61addd-16f1-430e-86c3-a9de5568a2cb


# ╔═╡ 28022772-2e3a-4bb7-8ad1-cbb68003e6c9
target_c = CuArray(target_3D);

# ╔═╡ 0b9f7439-cbed-45f9-b3a9-634fd2cbb87b
target_c2 = CuArray(target_3D) .> 0.5;

# ╔═╡ dc277892-b9c0-4f00-8f93-d5d5309b17dd
isobject = target_c .> 0.5;

# ╔═╡ 6a7b08e1-aa7f-42e5-88b0-f80c30e6501f
notobject = target_c .< 0.5;

# ╔═╡ ac674a47-03b6-407f-968a-274f3583b035
Revise.retry()

# ╔═╡ 55bcfcd9-8916-42ab-9621-29b0197b6a71
x = CUDA.rand(size(target_3D)...);

# ╔═╡ 25d9eb7a-e8c9-437c-bbf9-97b93f1f07b3
CUDA.@time loss(x, isobject, notobject)

# ╔═╡ 75084371-028e-4033-a34d-3790d5cf4c8e
CUDA.@time loss(x, target_c2)

# ╔═╡ 87fd8a28-3961-4074-a408-9d9de836abbc
CUDA.@time b = SwissVAMyKnife.Zygote.gradient(loss, x, isobject, notobject)[1];

# ╔═╡ 1d7c9a12-0f92-4388-95ff-0d75b68a411c
CUDA.@time a = SwissVAMyKnife.Zygote.gradient(loss, x, target_c2)[1];

# ╔═╡ db939035-e20c-49e7-8e77-5fce26bf656b
all(a .== b)

# ╔═╡ 532231b2-dd7c-4b83-882b-ffc0ff0057c0
SwissVAMyKnife.Zygote.refresh()

# ╔═╡ 9cc02718-fbc9-4d4a-9db8-f4cc467b99ae
Revise.errors()

# ╔═╡ 7a6c4c82-d695-4eaa-b7df-988a193bf34c
g(x, target, y, l, T) = @inbounds (2 .* y .* ((.- SwissVAMyKnife.NNlib.relu.(T(l.thresholds[2]) .- x) .* target) .+  (SwissVAMyKnife.NNlib.relu.(x .- Int(1))                .* target) .+ (SwissVAMyKnife.NNlib.relu.(x .- T(l.thresholds[1])) .* (1 .- target))))

# ╔═╡ b2b7338e-8c56-48bc-ad41-1c5f1cf59335
typeof(target_3D)

# ╔═╡ 2290ed9e-ce05-4e8b-868d-6cade38183be
size(b)

# ╔═╡ 021fc647-ad9f-4d1c-9b00-9be740fda4a0
g(x) = mean(x)

# ╔═╡ 8181cd4b-1d07-4497-ba71-5bbfa8668334
CUDA.@time g

# ╔═╡ 0deddef6-ec33-4702-be37-7a0a3483fc77
CUDA.@time c = g(x, target_c, 1, loss, Float32);

# ╔═╡ be52a362-9e32-4b3b-a3df-5ed70396ee73
sum(c .≈ a) / length(c)

# ╔═╡ c1ab2677-d00a-46b7-bacc-38ce0b7493f6
typeof(c)

# ╔═╡ 5781a316-ca96-4b2e-8040-4b42633c1d06
CUDA.@time SwissVAMyKnife.Zygote.gradient(mean, x);

# ╔═╡ a485a422-f5d2-4c9b-af88-99b161dce06d
CUDA.@time SwissVAMyKnife.Zygote.gradient(sum, x);

# ╔═╡ 5af70e11-0a8c-4858-b82d-fad554104a8c
loss2 = LossThreshold(sum_f=abs, thresholds=(0.65f0, 0.75f0))

# ╔═╡ 6b413ee2-be9f-499f-8a79-472fe205fc3f
# ╠═╡ disabled = true
#=╠═╡
loss2 = LossThreshold(thresholds=(0.65, 0.75))
  ╠═╡ =#

# ╔═╡ Cell order:
# ╟─9c478480-163a-4fe0-ad6b-32a337a616b9
# ╠═fb4bb1e2-c5e3-11ee-2f22-0bf5f621e215
# ╠═2487fdcc-f8c0-426d-9eff-ca9c8ce98100
# ╠═53c94ae4-3a86-4f3b-8e35-832859c5b465
# ╟─daf8043e-293f-495c-9b0d-b4e2b50de4a7
# ╟─de126e87-55c7-44ef-935d-b4bd96591914
# ╠═2dc23db0-87b8-4044-9408-b2ffcde33ee8
# ╠═9ec75d19-98be-4e04-990e-fd82b967525c
# ╠═bd516459-0be9-4ccb-a95b-32d07c15b410
# ╠═9d1ffd4d-1fe3-4f5c-bdc3-817fd0f23c7b
# ╟─465e830f-ca33-427a-ad7b-e18f3bbaa3c4
# ╠═4475e0c5-4ae6-403f-bb2b-c5bc8a25d18a
# ╠═cec09ba8-5707-495f-b4ac-94f346c33ce5
# ╠═b2dae3ea-5a40-498f-a134-5a9a98e58de0
# ╠═047023bf-96dd-4427-aa13-3fb2340af90e
# ╟─1d2bdd44-95d2-45ef-8e24-c8ca3df7485a
# ╠═ae3fdae9-7b3c-4391-8fff-530f7758942c
# ╟─b58c9e1f-7ca6-4171-8004-2b84d33cf335
# ╠═0dd200a7-a0f0-416e-ad69-7343e210a36e
# ╠═303a363f-f941-45b9-9194-62a2c411964a
# ╠═6d130809-8b78-485c-bba1-fc958bc9d873
# ╟─cb8b9ccc-4085-4abe-bd71-5e70cc8decfa
# ╠═e5f0660d-2adb-4c82-8945-e8164ec500f6
# ╟─b767afa8-784c-453b-ad0d-a1daf4d5ba5c
# ╠═07e97af6-f30b-4100-999c-c61b52cec76e
# ╠═c13abe2d-ff27-42e2-9534-cb1936e68236
# ╟─9887da10-96fa-4509-9b91-a6c21354f4ba
# ╟─f79bdf14-fefa-449f-b13a-49e6fb9153bf
# ╠═66560fd0-3cee-41ca-bb13-dee2a78bbb9b
# ╟─dc538ea6-54e3-4a0f-a5ef-427802239cf2
# ╟─71333d50-fa28-4bdf-bad4-d787682e043a
# ╟─5801f61a-79fa-4326-82dd-3901dd4d493a
# ╠═92f5b987-13c7-463c-b150-71c51ec4aa72
# ╟─11af6da4-5fa2-4e48-a1c8-32bc686b8239
# ╠═2f51ec68-09c6-4475-81b9-ec252fba5df7
# ╠═2e76c088-e61f-4deb-ac69-b73d6e3f361c
# ╠═0e4da85b-99bd-47e0-a8fe-3c69c3a05708
# ╠═a04d12c2-f718-44e0-a755-da1132c0330e
# ╟─5ade07d4-e7b3-4e78-94fa-454924a4066c
# ╟─ccd0ad5e-5c01-4ec0-afc4-0dcce85dc9ef
# ╠═17621d91-1f93-4fcf-abf1-5e97463a0bdf
# ╠═05c6a38e-a87d-46da-acb6-33b393369f4c
# ╟─f940c2a6-ebc2-4dca-986c-fe25bfa9e4f0
# ╠═f9a82207-6841-44c9-9aec-e8da3de8e0b0
# ╠═675764cb-721a-4b89-92c4-8a1ab7f5867f
# ╠═9ddd098d-2d78-4de8-a322-40a2463adcda
# ╠═d1f33cd0-0927-417e-a024-6a8c76da1fb9
# ╠═d56eb3ab-4698-4ee4-aae1-461b964b778c
# ╠═b6a6237f-b03f-402d-9693-f00044e3539e
# ╟─eda29eed-507c-4f55-b1e6-590c9616db8e
# ╠═fc239b37-dc88-48cd-9479-99074d96ece5
# ╠═7de5d398-20d6-44e2-b2eb-3007111e146f
# ╟─a7cafba6-849a-4eb8-9709-a76cb98e9879
# ╠═0a655d51-e3b6-413b-83de-9781974242a2
# ╟─877b1484-969c-45d7-a3e7-4f0301a81a4b
# ╠═a05d8dd4-40c3-41da-9dbd-f58ab161b2d4
# ╠═8134de5c-4bc2-498d-af0e-06ed1a1e8539
# ╠═d6a59254-bffe-4116-9335-8884eb44556f
# ╠═329e47a6-9778-43a8-b74f-74f0d59458de
# ╠═5af70e11-0a8c-4858-b82d-fad554104a8c
# ╠═6009b601-6988-4be8-a519-7c59660c73ab
# ╠═f15224d2-6eb6-4919-aa79-30a4728edc20
# ╠═1ab55d13-5c68-40fe-875b-c29e4298f936
# ╠═85d9c76b-9ee9-4b81-b9a8-6597309fc999
# ╠═1ba0adc2-8dbe-4f25-9746-cb344e9fa7c7
# ╠═9af3c5ac-934c-4d07-95ee-5b2f947b0e22
# ╠═5607deac-8814-4810-a009-14dc223c8fa7
# ╠═2be5a610-e553-4dcb-833f-6febc813f324
# ╠═4d0d1f34-1068-427b-b2b7-3966b533b6d6
# ╠═0c917ace-401c-447c-b021-23aa401be718
# ╠═fa45b520-18a8-4465-b642-a1c08de48e20
# ╟─10cc0e79-5d5b-43cf-afb8-ad21875d6d97
# ╟─0835d1b0-a562-4edf-939e-3044f94e53c7
# ╠═83793ffe-38de-4a76-b80f-8dee1d18c5a5
# ╟─262b96e8-a78c-441c-9b46-2d87636286d7
# ╠═c8774a2b-47e0-4be6-b213-bdef6d7b0726
# ╠═497cd20f-5544-43dc-8037-1263afb5cf46
# ╠═3d4e99a1-ebf3-4b51-a3d9-8ada7fb3267c
# ╠═b99cbb89-6dd3-4832-ac66-d24fbc3fc060
# ╠═3fadfd32-5552-43ea-9bcb-725a1a78a00d
# ╠═6b413ee2-be9f-499f-8a79-472fe205fc3f
# ╠═f50ade1d-37f1-4f03-a081-e74642798124
# ╠═510bd5b1-f91f-4f7b-8a84-f8eebacb9aff
# ╠═fa639ca1-cf25-41d0-9a29-8c325d808ba4
# ╠═47f86ac2-a688-40c3-a4c3-fd750b698c27
# ╠═8e61addd-16f1-430e-86c3-a9de5568a2cb
# ╠═28022772-2e3a-4bb7-8ad1-cbb68003e6c9
# ╠═0b9f7439-cbed-45f9-b3a9-634fd2cbb87b
# ╠═dc277892-b9c0-4f00-8f93-d5d5309b17dd
# ╠═6a7b08e1-aa7f-42e5-88b0-f80c30e6501f
# ╠═ac674a47-03b6-407f-968a-274f3583b035
# ╠═25d9eb7a-e8c9-437c-bbf9-97b93f1f07b3
# ╠═75084371-028e-4033-a34d-3790d5cf4c8e
# ╠═55bcfcd9-8916-42ab-9621-29b0197b6a71
# ╠═87fd8a28-3961-4074-a408-9d9de836abbc
# ╠═1d7c9a12-0f92-4388-95ff-0d75b68a411c
# ╟─db939035-e20c-49e7-8e77-5fce26bf656b
# ╠═8181cd4b-1d07-4497-ba71-5bbfa8668334
# ╠═532231b2-dd7c-4b83-882b-ffc0ff0057c0
# ╠═9cc02718-fbc9-4d4a-9db8-f4cc467b99ae
# ╠═be52a362-9e32-4b3b-a3df-5ed70396ee73
# ╠═c1ab2677-d00a-46b7-bacc-38ce0b7493f6
# ╠═7a6c4c82-d695-4eaa-b7df-988a193bf34c
# ╠═0deddef6-ec33-4702-be37-7a0a3483fc77
# ╠═b2b7338e-8c56-48bc-ad41-1c5f1cf59335
# ╠═2290ed9e-ce05-4e8b-868d-6cade38183be
# ╠═4b946954-9432-4ffe-9b4d-8c590be96906
# ╠═021fc647-ad9f-4d1c-9b00-9be740fda4a0
# ╠═5781a316-ca96-4b2e-8040-4b42633c1d06
# ╠═a485a422-f5d2-4c9b-af88-99b161dce06d
