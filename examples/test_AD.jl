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

# ╔═╡ 62673cc2-9b5b-11ee-2687-f3c65e3a77fa
begin
	using Pkg
	Pkg.activate(".")
	using Revise
end

# ╔═╡ 28e9ad0b-6c91-4ab0-90a6-d0866d0654b0
Pkg.add("JLD2")

# ╔═╡ 32ff74bd-8ca6-40e0-828b-ca30dc2bea2d
using JLD2

# ╔═╡ 9b0fe39d-d340-40bf-bdb7-22a57bae1885
using ChainRulesCore, Plots

# ╔═╡ af984a5b-7834-4823-b93a-c5d07e3f95a4
using CUDA, WaveOpticsPropagation, Zygote, Optim, ImageShow, IndexFunArrays

# ╔═╡ 1f602abd-cd1e-4a1f-89e1-b11d43c4d424
using DiffImageRotation

# ╔═╡ e689c752-0c65-44de-b998-b6ecbb3b949c
using PlutoUI, FileIO, Colors, NDTools

# ╔═╡ 800d09a3-41db-401d-bdcf-1f72d1443ca1
using CUDA.CUFFT

# ╔═╡ dd968fb3-7935-438c-8df8-abdd1597b48f
string(999, pad=3)

# ╔═╡ 66ca565d-83ca-46f2-a13d-9667c0d2aeee
use_CUDA = Ref(true && CUDA.functional())

# ╔═╡ cd884189-1d50-4468-ad69-5a679a6a81af
togoc(x) = use_CUDA[] ? CuArray(x) : x

# ╔═╡ 76832d96-0f13-4dda-8306-af0407e6d239
function load_benchy(sz, sz_file, path)
	target = zeros(Float32, sz_file)

	for i in 1:sz_file[1]
		target[i, :, :] .= Gray.(load(joinpath(path, string("boat_", string(i, pad=3) ,".png"))))
	end

	target2 = zeros(Float32, sz)
	WaveOpticsPropagation.set_center!(target2, target)
	return togoc(target2)
end

# ╔═╡ c84b69d7-086c-445b-afe0-9724b780d806
var"@mytime" = use_CUDA[] ? CUDA.var"@time" : var"@time"

# ╔═╡ 43b9fa00-7cca-4e68-925a-5e2ff308805f
target = load_benchy((270, 270, 270), (200, 200, 200), "/home/felix/Downloads/benchy/files/output/")

# ╔═╡ 7d43cea2-c1d5-459b-8279-7593e52fde77
# ╠═╡ disabled = true
#=╠═╡
target = repeat(reshape(select_region(togoc(Float32.(Gray.(load("/home/felix/Documents/code/Printing_Siemens_Star_Target/siemens_star_350px.png")))), new_size=(200, 200)), 1, 200, 200), 200, 1, 1)
  ╠═╡ =#

# ╔═╡ 355984ea-e957-45fa-a17f-9a853839e29b
@bind ix Slider(1:size(target, 1))

# ╔═╡ 30c6daa8-9d40-4c42-82a0-b78ea7843f03
simshow(Array(target[ix, :, :]))

# ╔═╡ 35d0680c-f075-4113-8720-bf447b8f0b3d


# ╔═╡ 87668471-f7e2-4ace-a95a-82a3f089b447
# ╠═╡ disabled = true
#=╠═╡
begin
	target = togoc(zeros(Float32, 350, 350, 350));
	offset = 200
	target[30 + offset: offset + 60, offset + 30: offset + 60, offset + 30: offset + 60] .= 1;
	target[offset + 35: offset + 52, offset + 35:offset + 54, offset + 35:offset + 56] .= 0;

	#target = imrotate(target, deg2rad(50))
end;
  ╠═╡ =#

# ╔═╡ e9e23d62-0779-4d63-8485-2ff5677cd8ba
simshow(Array(target[:, 60, :]))

# ╔═╡ 067d9ac3-4bdc-44e5-87c2-e0936cbcd8f8
sum(isone.(target))

# ╔═╡ 226dfbbe-2277-4ffe-b15c-8f2e1d434a86
simshow(Array(target[:, :, 25]))

# ╔═╡ a628b053-7d6a-4aac-80b8-1e560a0e8a49
function make_fg!(fwd, AS_abs2, angles, target, loss=:L2,
				  thresholds=(0.7, 0.8))
	
	mask = togoc(rr2(eltype(target), (size(target)[1:2]..., )) .<= (size(target, 1) ÷ 2  - 1)^2)

	mask = reshape(mask, (1, size(target, 1), size(target, 2)))

	notobject = iszero.(target)
	isobject = isone.(target)
	
    L2 = let target=target, fwd=fwd, AS_abs2=AS_abs2, angles=angles, mask=mask
        function L2(x)
            return sum(abs2, (fwd(x, AS_abs2, angles) .- target) .* mask)
		end
    end

    L_VAM = let target=target, fwd=fwd, AS_abs2=AS_abs2, angles=angles, mask=mask, notobject=notobject, isobject=isobject, thresholds=thresholds
        function L_VAM(x)
			f = fwd(x, AS_abs2, angles)# .* mask
			f = f ./ maximum(f)
			return sum(abs2, max.(0, thresholds[2] .- view(f, isobject))) .+ sum(abs2, max.(0, view(f, notobject) .- thresholds[1]))
		end
    end

    f = L_VAM

    g! = let f=L_VAM
        function g!(G, rec)
            if !isnothing(G)
                return G .= Zygote.gradient(f, rec)[1]
            end
        end
    end
    return f, g!
end

# ╔═╡ e6df8ac8-35b5-455e-8c5a-6195f153e4e1
gradient(x -> sum(max.(0, x .- 0.5) .+ sum(max.(0, x))), rand(10,10))

# ╔═╡ 12e74bc5-9144-46c8-9219-90f1e5edff20
isone.(CUDA.rand(2,2))

# ╔═╡ 140082c8-cbe8-40d2-9e1f-a076db805156
Zygote._pullback(x -> abs2.(x), [1 2; 3 4])[2](rand((2,2)))

# ╔═╡ 380386a4-0166-4cac-b46c-ab69bbc189db
begin
		mask = togoc(rr2(eltype(target), (size(target)[1:2]..., )) .<= (size(target, 1) ÷ 2  - 1)^2)
	
		mask = reshape(mask, (1, size(target, 1), size(target, 2)))
end

# ╔═╡ ebf4598b-ab9d-415f-95d7-7e19731ae966
function fwd(x, AS_abs2, angles)
	intensity = copy(similar(x, real(eltype(x)), (size(x, 1), size(x, 2), size(x, 2))))
	fill!(intensity, 0)
	
	tmp_rot = similar(intensity);
	tmp_rot .= 0;
	for (i, angle) in enumerate(angles)		
		CUDA.@sync tmp = AS_abs2(view(x, :, :, i))
		CUDA.@sync intensity .+= PermutedDimsArray(imrotate!(tmp_rot, PermutedDimsArray(tmp, (2, 3, 1)), angle), (3, 1, 2))
		tmp_rot .= 0
	end	
	return intensity .* mask
end

# ╔═╡ 8a37ec9d-b984-4c7b-b0a7-435305806366
function ChainRulesCore.rrule(::typeof(fwd), x, AS_abs2, angles)
	res = fwd(x, AS_abs2, angles)
	pb_rotate = let AS_abs2=AS_abs2, angles=angles
	function pb_rotate(ȳ)
		grad = similar(x, real(eltype(x)), size(x, 1), size(x, 2), size(angles, 1))
		fill!(grad, 0)
		#@show sum(ȳ)
		tmp_rot = similar(res);
		tmp_rot .= 0;
		for (i, angle) in enumerate(angles)		
			CUDA.@sync tmp::CuArray = Zygote._pullback(AS_abs2, view(x, :, :, i))[2](
					PermutedDimsArray(imrotate!(tmp_rot, PermutedDimsArray(ȳ, (2, 3, 1)), angle, adjoint=true), (3, 1, 2))
			)[2]
			tmp_rot .= 0
			#@show sum(tmp)
			#@show size(x), angle, size(tmp)
			grad[:, :, i] .= tmp
		end	
		return NoTangent(), grad, NoTangent(), NoTangent()
	end    
	end
	return res, pb_rotate
 end

# ╔═╡ 323098fe-072f-4f17-b8e6-4144770029f0
simshow(Array(mask[1, :, :]))

# ╔═╡ 1ae5c61f-d34d-4618-8679-c28ecbb757bf
L = 400f-6

# ╔═╡ eaf74b38-3aae-4651-8083-930ff3ae8eed
λ = 405f-9 ./ 1.5

# ╔═╡ 6e681b2d-8510-4970-b8f8-98840af0988e
angles = deg2rad.(range(0, 360, 100))

# ╔═╡ 9f0d2df4-e516-4b2e-9361-9cfceb6e262c
begin
	patterns_0 = togoc(zeros(Float32, (size(target,1), size(target,2), size(angles, 1))));
	# zero init fails somehow
	patterns_0[begin+6:end-6, begin+6:end-6, :] .= 0.001
end;

# ╔═╡ 3bc8c469-6e88-4131-a5e1-409dcb447031
z = togoc(range(0, L, size(patterns_0, 1)));

# ╔═╡ b271f58d-8195-4cc2-8682-160bc531c744
#AS, _ = Angular_Spectrum(repeat(0im .+ patterns_0[:, :, 1], 1, 1, size(target, 1)), z, λ, L)
AS, _ = Angular_Spectrum(patterns_0[:, :, 1] .+ 0im, z, λ, L, padding=false)

# ╔═╡ eb53a1a9-5fc9-4e65-a326-0a9c816dcb71
AS_abs2 = 
	let target=target, AS=AS
	function AS_abs2(x)
		#abs2.(AS(repeat(0im .+ abs2.(x), 1, 1, size(target, 1)))[1])
		# normalize to 1
		abs2.(AS(abs2.(x) .+ 0im)[1])
	end
end

# ╔═╡ e3f9bf50-26de-467e-b4f3-54eb753278ab
f, g! = make_fg!(fwd, AS_abs2, angles, target)

# ╔═╡ 3748e3d2-7ebf-496e-83fb-996d6d1a025d
simshow(Array(target[20, :, :]))

# ╔═╡ 3ccb6a09-a948-4db0-be74-42859c403b06
sizeof(target) / 2^20

# ╔═╡ bc83c7f1-e50c-4d2e-b4d1-9edaa56da33b
CUDA.@time CUDA.@sync f(patterns_0);

# ╔═╡ bf0735d8-dc10-43c7-91fa-e7b6cc0099dc
CUDA.@time CUDA.@sync AS_abs2(patterns_0[:, :, 1]);

# ╔═╡ f74283b3-66ce-4407-b259-388a1926dc54
p = plan_fft(patterns_0);

# ╔═╡ 3644650d-a213-465e-bc96-737d099447d8
CUDA.@time CUDA.@sync abs2.(inv(p) * (patterns_0 .* (p * patterns_0)));

# ╔═╡ 2c4f00b3-d800-49ae-bd47-fe26c84dc887
tmp = copy(patterns_0);

# ╔═╡ 542f517b-81ae-44e3-9cd5-32377153700f
# ╠═╡ disabled = true
#=╠═╡
CUDA.@time CUDA.@sync g!(copy(patterns_0), patterns_0);
  ╠═╡ =#

# ╔═╡ 318c0c63-ab4f-4b80-9029-508dddfa0821
# ╠═╡ disabled = true
#=╠═╡
sum(g!(zero.(patterns_0), patterns_0 .+ 1f0))
  ╠═╡ =#

# ╔═╡ 2f790a45-7594-4dc2-a06f-6ff48355fc42
Zygote.refresh()

# ╔═╡ 41897c92-462e-479c-94ee-5a8f60503343
md"# Optimize"

# ╔═╡ d2fdb8ae-ad9a-43ef-84ab-f3e5f15337b0
CUDA.@time res = Optim.optimize(f, g!, patterns_0, LBFGS(),
                                 Optim.Options(iterations = 40,  
                                               store_trace=true))

# ╔═╡ 7e992329-d52b-415f-a305-35288a8dd1f7
@bind iangle Slider(1:size(angles, 1), show_value=true)

# ╔═╡ ce946b50-aa9b-418d-9fdd-fa8d3d1ae315
simshow(abs2.(Array(res.minimizer[:, :, iangle])), cmap=:turbo, γ=1, set_one=true)

# ╔═╡ 6760558a-888a-4576-979a-fd6ca2b6126c
@bind thresh Slider(0.0:0.01:2, show_value=true)

# ╔═╡ 5a243638-6ccd-410b-8b78-ca3e7b2eee05
@bind ixx Slider(1:size(res.minimizer, 1), show_value=true)

# ╔═╡ 770577a4-43e2-45ab-9b3d-36331951f729
@bind anglei Slider(1:size(angles, 1), show_value=true)

# ╔═╡ ee828eca-0a43-4aed-9d33-c1a09ccfbc01
begin
	fwdd = Array(fwd(res.minimizer, AS_abs2, angles));
	fwdd ./= maximum(fwdd);
end

# ╔═╡ 580b6b33-2364-4764-a226-6554a3c2e184
[simshow(view(fwdd, ixx, :, :), γ=1, set_one=false, cmap=:turbo) simshow(view(fwdd, ixx, :, :) .> thresh, γ=1, set_one=true, cmap=:turbo) simshow(view(Array(target), ixx, :, :))]

# ╔═╡ 04cf4529-75bb-4f84-9aee-a61231471b4b
extrema(fwdd)

# ╔═╡ b56ab2f3-0a3d-4223-874d-459bd2e30fe9
begin
	fwdd2 = Array(fwd(res.minimizer[:, :, anglei:anglei], AS_abs2, angles[anglei:anglei]));
	fwdd2 ./= maximum(fwdd2);
end

# ╔═╡ 27d54e14-e1c9-406b-b55f-20ce55aa3eca
simshow(view(fwdd2, ixx, :, :), γ=1, set_one=false, cmap=:turbo)

# ╔═╡ 963dd982-d5c5-41ab-9938-208bd4960209
extrema(res.minimizer)

# ╔═╡ a671b312-7202-41e7-960c-e96a5607be1d
histogram(Array(abs2.(res.minimizer[:])), ylim=(1, 10_000000), yscale=:log10, bins=(0.0:0.000001:0.0005))

# ╔═╡ 229a2d8d-e8c3-4a40-949e-8e67352fc8b2
extrema(abs2.(res.minimizer))

# ╔═╡ 2c497f7a-d85c-4cca-9b61-3d0e034e5ec8
extrema(abs2.(res.minimizer))

# ╔═╡ bbda7743-1c92-4565-add8-1228e3cc8cee
sum(fwdd .== 0)

# ╔═╡ 5798474f-5e40-4cbf-a3ac-2b1ffc9da6c0
begin
	plot_font = "Computer Modern"
	default(fontfamily=plot_font,
	        linewidth=2, framestyle=:box, label=nothing, grid=false)
	scalefontsizes(1.3)
end

# ╔═╡ 1dfebc87-8675-43e3-8623-e6885dcb0e8c
function plot_histogram(img, object_printed, thresholds, chosen_threshold)
	histogram(object_printed[img .== 0], bins=(0.0:0.01:2), xlim=(0.0, 1.0), label="dose distribution void", ylabel="voxel count", xlabel="normalized intensity",  ylim=(1, 1000000),  yscale=:log10, linewidth=1, legend=:topleft)
	histogram!(object_printed[img .== 1], bins=(0.0:0.01:1), label="dose distribution object", ylabel="voxel count", xlabel="normalized intensity",  ylim=(1, 1000000), linewidth=1,  yscale=:log10,)
	plot!([thresholds[1], thresholds[1]], [1, 10000_000], label="lower threshold", linewidth=3)
	plot!([thresholds[2], thresholds[2]], [1, 10000_000], label="upper threshold", linewidth=3)
	#plot!([chosen_threshold, chosen_threshold], [1, 30000000], label="chosen threshold", linewidth=3)
end

# ╔═╡ 99ae1172-c178-49b7-91ff-48170d1e2357
plot_histogram(Array(target), fwdd, (0.7, 0.8), 0.75)

# ╔═╡ 83c230c6-8fe3-4c24-b3da-64cdaf944d50
histogram(rand(1:100, (1_000)), yscale=:log10, ylim=(1, 10000))

# ╔═╡ Cell order:
# ╠═62673cc2-9b5b-11ee-2687-f3c65e3a77fa
# ╠═28e9ad0b-6c91-4ab0-90a6-d0866d0654b0
# ╠═32ff74bd-8ca6-40e0-828b-ca30dc2bea2d
# ╠═9b0fe39d-d340-40bf-bdb7-22a57bae1885
# ╠═af984a5b-7834-4823-b93a-c5d07e3f95a4
# ╠═1f602abd-cd1e-4a1f-89e1-b11d43c4d424
# ╠═e689c752-0c65-44de-b998-b6ecbb3b949c
# ╠═800d09a3-41db-401d-bdcf-1f72d1443ca1
# ╠═cd884189-1d50-4468-ad69-5a679a6a81af
# ╠═76832d96-0f13-4dda-8306-af0407e6d239
# ╠═dd968fb3-7935-438c-8df8-abdd1597b48f
# ╠═66ca565d-83ca-46f2-a13d-9667c0d2aeee
# ╠═c84b69d7-086c-445b-afe0-9724b780d806
# ╠═43b9fa00-7cca-4e68-925a-5e2ff308805f
# ╠═7d43cea2-c1d5-459b-8279-7593e52fde77
# ╠═30c6daa8-9d40-4c42-82a0-b78ea7843f03
# ╠═355984ea-e957-45fa-a17f-9a853839e29b
# ╠═35d0680c-f075-4113-8720-bf447b8f0b3d
# ╠═87668471-f7e2-4ace-a95a-82a3f089b447
# ╠═e9e23d62-0779-4d63-8485-2ff5677cd8ba
# ╠═067d9ac3-4bdc-44e5-87c2-e0936cbcd8f8
# ╠═226dfbbe-2277-4ffe-b15c-8f2e1d434a86
# ╠═a628b053-7d6a-4aac-80b8-1e560a0e8a49
# ╠═e6df8ac8-35b5-455e-8c5a-6195f153e4e1
# ╠═ebf4598b-ab9d-415f-95d7-7e19731ae966
# ╠═12e74bc5-9144-46c8-9219-90f1e5edff20
# ╠═140082c8-cbe8-40d2-9e1f-a076db805156
# ╠═8a37ec9d-b984-4c7b-b0a7-435305806366
# ╠═9f0d2df4-e516-4b2e-9361-9cfceb6e262c
# ╠═380386a4-0166-4cac-b46c-ab69bbc189db
# ╠═323098fe-072f-4f17-b8e6-4144770029f0
# ╠═1ae5c61f-d34d-4618-8679-c28ecbb757bf
# ╠═eaf74b38-3aae-4651-8083-930ff3ae8eed
# ╠═3bc8c469-6e88-4131-a5e1-409dcb447031
# ╠═b271f58d-8195-4cc2-8682-160bc531c744
# ╠═6e681b2d-8510-4970-b8f8-98840af0988e
# ╠═eb53a1a9-5fc9-4e65-a326-0a9c816dcb71
# ╠═e3f9bf50-26de-467e-b4f3-54eb753278ab
# ╠═3748e3d2-7ebf-496e-83fb-996d6d1a025d
# ╠═3ccb6a09-a948-4db0-be74-42859c403b06
# ╠═bc83c7f1-e50c-4d2e-b4d1-9edaa56da33b
# ╠═bf0735d8-dc10-43c7-91fa-e7b6cc0099dc
# ╠═f74283b3-66ce-4407-b259-388a1926dc54
# ╠═3644650d-a213-465e-bc96-737d099447d8
# ╠═2c4f00b3-d800-49ae-bd47-fe26c84dc887
# ╠═542f517b-81ae-44e3-9cd5-32377153700f
# ╠═318c0c63-ab4f-4b80-9029-508dddfa0821
# ╠═2f790a45-7594-4dc2-a06f-6ff48355fc42
# ╟─41897c92-462e-479c-94ee-5a8f60503343
# ╠═d2fdb8ae-ad9a-43ef-84ab-f3e5f15337b0
# ╠═7e992329-d52b-415f-a305-35288a8dd1f7
# ╠═ce946b50-aa9b-418d-9fdd-fa8d3d1ae315
# ╠═6760558a-888a-4576-979a-fd6ca2b6126c
# ╠═5a243638-6ccd-410b-8b78-ca3e7b2eee05
# ╠═580b6b33-2364-4764-a226-6554a3c2e184
# ╠═770577a4-43e2-45ab-9b3d-36331951f729
# ╠═27d54e14-e1c9-406b-b55f-20ce55aa3eca
# ╠═04cf4529-75bb-4f84-9aee-a61231471b4b
# ╠═ee828eca-0a43-4aed-9d33-c1a09ccfbc01
# ╠═b56ab2f3-0a3d-4223-874d-459bd2e30fe9
# ╠═963dd982-d5c5-41ab-9938-208bd4960209
# ╠═99ae1172-c178-49b7-91ff-48170d1e2357
# ╠═a671b312-7202-41e7-960c-e96a5607be1d
# ╠═229a2d8d-e8c3-4a40-949e-8e67352fc8b2
# ╠═2c497f7a-d85c-4cca-9b61-3d0e034e5ec8
# ╠═bbda7743-1c92-4565-add8-1228e3cc8cee
# ╠═5798474f-5e40-4cbf-a3ac-2b1ffc9da6c0
# ╠═1dfebc87-8675-43e3-8623-e6885dcb0e8c
# ╠═83c230c6-8fe3-4c24-b3da-64cdaf944d50
