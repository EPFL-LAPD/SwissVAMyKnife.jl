### A Pluto.jl notebook ###
# v0.19.38

using Markdown
using InteractiveUtils

# ╔═╡ 32d9b7b4-bacd-48c5-8ac5-a7d16bc26f3f
begin
	using Pkg
	Pkg.activate(".")
	Pkg.instantiate()
	using Revise
end

# ╔═╡ fb00f77e-3a43-43c6-83f5-ab9ffa690585
# this is our main package
using SwissVAMyKnife

# ╔═╡ 2a2c5c57-3b51-463b-90b3-bf4733df1c71
using ImageShow, ImageIO, PlutoUI, IndexFunArrays, Optim, RadonKA, FileIO, Colors, Plots, NDTools, TestImages, Noise, Zygote

# ╔═╡ efc9b013-0452-482f-9112-d444d2197ba9
using CUDA

# ╔═╡ c973ccfd-11b7-4e0a-9d1a-32b0d449bfd1
md"# 0. Load packages
On the first run, Julia is going to install some packages automatically. So start this notebook and give it some minutes (5-10min) to install all packages. 
No worries, any future runs will be much faster to start!
"

# ╔═╡ 9122b9a9-43f4-48be-88f6-6d02a110f900
TableOfContents()

# ╔═╡ 54e9b00d-2589-430e-9ac9-ffdb48a54c0e
use_CUDA = Ref(true && CUDA.functional())

# ╔═╡ 5c3dda08-010e-4d76-bf35-1c3385a2a0c0
md" ## CUDA
CUDA accelerates the pattern generation easily by 5-20 times!
Otherwise most of the code will be multithreaded on your CPU but we strongly recommended the usage of CUDA for large scale 3D pattern generation.

Your CUDA is functional: **$(use_CUDA[])**
"

# ╔═╡ 93591474-679b-4944-a673-616cb5f354e0
var"@mytime" = use_CUDA[] ? CUDA.var"@time" : var"@time"

# ╔═╡ 0a7cc390-892a-4973-a725-f9765f7aaaff
togoc(x) = use_CUDA[] ? CuArray(x) : x

# ╔═╡ aebb7657-4b6c-4f22-b6be-c7ba054d7aa3
md"# 1. Load TestImage"

# ╔═╡ 9aa5e2c5-f0d0-43dc-9c3a-b4cebc5bb100
target = Float32.(togoc(Float32.(load_example_target("3DBenchy_550")[begin:1:end, begin:1:end, 320:320])));

# ╔═╡ 018120f1-8859-4606-ac62-507fbcaf4097
simshow(Array(target[:, :, 1]))

# ╔═╡ f1cbc5fd-8454-430c-9267-73c67eca623b
md"# 2. Create Dataset

First we need to create an artifical dataset. 
For that we need to specify some parameter such as the camera_diameter, resin and vial properties.
"

# ╔═╡ e5bec833-47c5-4f65-8667-f28d4ccb1fbc
angles = range(0, 2π, 101)[begin:end-1]

# ╔═╡ f618cc32-6851-42de-b2c2-0ddc70b4668d
geometry_OST = OSTRayOptics(
	angles=angles,
	μ=0/256,
	R_outer=(16.60e-3) / 2,
	R_inner=(15.2e-3) / 2,
	camera_diameter=16.6e-3,
	n_vial=1.47,
	n_resin=1.4849
)

# ╔═╡ 60542c0f-875d-4f16-a25b-aec697753b72
geometry_radon = SwissVAMyKnife.prepare_OST_geometry(target, geometry_OST)

# ╔═╡ 6eb9c0bc-2aa7-470d-96ca-3716c3c75e4e
sinogram = radon(target, angles; geometry=geometry_radon);

# ╔═╡ 6660d36d-d1ba-4a48-ad4b-2e9c486cd7c2
@time begin
	filtered = backproject_filtered(sinogram, angles; geometry=geometry_radon);
	filtered ./= maximum(filtered)
end;

# ╔═╡ 3b51d546-80f3-4cc0-8cf6-fd31834a070d
measured = togoc(poisson(Array(sinogram), 100_000));

# ╔═╡ b0a5c9f6-448c-46b5-b031-773f489551bc
simshow(Array(measured)[:, :, 1])

# ╔═╡ 3dfccb9a-8f21-4695-965c-c1d97077ba52
md"# 3. Reconstruct"

# ╔═╡ 6138e8d0-c457-4916-a520-b1d70c26997b
@time rec0, ores = reconstruct_OST(measured, geometry_OST, λ=1f-5, 
							iterations=20)

# ╔═╡ 76839478-b978-47cf-9641-7b4e85c8c2d6
begin
	rec0 ./= maximum(rec0);
	simshow([Array(rec0)[:, :, 1] Array(filtered)[:, :, 1] Array(target)[:, :, 1]])
end

# ╔═╡ Cell order:
# ╟─c973ccfd-11b7-4e0a-9d1a-32b0d449bfd1
# ╠═32d9b7b4-bacd-48c5-8ac5-a7d16bc26f3f
# ╠═fb00f77e-3a43-43c6-83f5-ab9ffa690585
# ╠═2a2c5c57-3b51-463b-90b3-bf4733df1c71
# ╠═9122b9a9-43f4-48be-88f6-6d02a110f900
# ╟─5c3dda08-010e-4d76-bf35-1c3385a2a0c0
# ╠═efc9b013-0452-482f-9112-d444d2197ba9
# ╠═54e9b00d-2589-430e-9ac9-ffdb48a54c0e
# ╠═93591474-679b-4944-a673-616cb5f354e0
# ╠═0a7cc390-892a-4973-a725-f9765f7aaaff
# ╟─aebb7657-4b6c-4f22-b6be-c7ba054d7aa3
# ╠═9aa5e2c5-f0d0-43dc-9c3a-b4cebc5bb100
# ╠═018120f1-8859-4606-ac62-507fbcaf4097
# ╟─f1cbc5fd-8454-430c-9267-73c67eca623b
# ╠═f618cc32-6851-42de-b2c2-0ddc70b4668d
# ╠═e5bec833-47c5-4f65-8667-f28d4ccb1fbc
# ╠═60542c0f-875d-4f16-a25b-aec697753b72
# ╠═6eb9c0bc-2aa7-470d-96ca-3716c3c75e4e
# ╠═6660d36d-d1ba-4a48-ad4b-2e9c486cd7c2
# ╠═3b51d546-80f3-4cc0-8cf6-fd31834a070d
# ╠═b0a5c9f6-448c-46b5-b031-773f489551bc
# ╟─3dfccb9a-8f21-4695-965c-c1d97077ba52
# ╠═6138e8d0-c457-4916-a520-b1d70c26997b
# ╠═76839478-b978-47cf-9641-7b4e85c8c2d6
