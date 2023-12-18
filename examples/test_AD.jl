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

# ╔═╡ 62673cc2-9b5b-11ee-2687-f3c65e3a77fa
begin
	using Pkg
	Pkg.activate(".")
	using Revise
end

# ╔═╡ 9b0fe39d-d340-40bf-bdb7-22a57bae1885
using ChainRulesCore

# ╔═╡ af984a5b-7834-4823-b93a-c5d07e3f95a4
using CUDA, WaveOpticsPropagation, Zygote, Optim, DiffImageRotation, ImageShow, IndexFunArrays

# ╔═╡ e689c752-0c65-44de-b998-b6ecbb3b949c
using PlutoUI

# ╔═╡ 66ca565d-83ca-46f2-a13d-9667c0d2aeee
use_CUDA = Ref(true && CUDA.functional())

# ╔═╡ c84b69d7-086c-445b-afe0-9724b780d806
var"@mytime" = use_CUDA[] ? CUDA.var"@time" : var"@time"

# ╔═╡ cd884189-1d50-4468-ad69-5a679a6a81af
togoc(x) = use_CUDA[] ? CuArray(x) : x

# ╔═╡ 35d0680c-f075-4113-8720-bf447b8f0b3d


# ╔═╡ 87668471-f7e2-4ace-a95a-82a3f089b447
begin
	target = togoc(zeros(Float32, 100, 100, 100));
	target[30:60, 30:60, 30:60] .= 1;
	target[35:52, 35:54, 35:56] .= 0;

	#target = imrotate(target, deg2rad(50))
end;

# ╔═╡ e9e23d62-0779-4d63-8485-2ff5677cd8ba
simshow(Array(target[:, 35, :]))

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

    L_VAM = let target=target, fwd=fwd, AS_abs2=AS_abs2, angles=angles, mask=mask, notobject=notobject, isobject=isobject
        function L_VAM(x)
			f = fwd(x, AS_abs2, angles)# .* mask
			#f = f ./ maximum(f)
			return sum(max.(0, thresholds[2] .- f[isobject])) .+ sum(max.(0, f[notobject] .- thresholds[1]))
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

# ╔═╡ 140082c8-cbe8-40d2-9e1f-a076db805156
Zygote._pullback(x -> abs2.(x), [1 2; 3 4])[2](rand((2,2)))

# ╔═╡ 380386a4-0166-4cac-b46c-ab69bbc189db
begin
		mask = togoc(rr2(eltype(target), (size(target)[1:2]..., )) .<= (size(target, 1) ÷ 2  - 1)^2)
	
		mask = reshape(mask, (1, size(target, 1), size(target, 2)))
end

# ╔═╡ ebf4598b-ab9d-415f-95d7-7e19731ae966
function fwd(x, AS_abs2, angles)
	intensity = similar(x, real(eltype(x)), (size(x, 1), size(x, 2), size(x, 2)))
	fill!(intensity, 0)
	for (i, angle) in enumerate(angles)		
		tmp = AS_abs2(x[:, :, i])
		size(tmp), size(intensity)	
		intensity .+= permutedims(imrotate(permutedims(tmp, (2, 3, 1)), angle), (3, 1, 2))
	end	
	return intensity .* mask
end

# ╔═╡ 8a37ec9d-b984-4c7b-b0a7-435305806366
function ChainRulesCore.rrule(::typeof(fwd), x, AS_abs2, angles)
	res = fwd(x, AS_abs2, angles)
	
	function pb_rotate(ȳ)
		grad = similar(x, real(eltype(x)), size(x, 1), size(x, 2), size(angles, 1))
		fill!(grad, 0)
		#@show sum(ȳ)

		for (i, angle) in enumerate(angles)		
			tmp::CuArray = Zygote._pullback(AS_abs2, x[:, :, i])[2](
					permutedims(imrotate(permutedims(ȳ, (2, 3, 1)), angle, adjoint=true), (3, 1, 2))
			)[2]
			#@show sum(tmp)
			#@show size(x), angle, size(tmp)
			grad[:, :, i] .= tmp
		end	
		return NoTangent(), grad, NoTangent(), NoTangent()
	end    
	return res, pb_rotate
 end

# ╔═╡ 323098fe-072f-4f17-b8e6-4144770029f0
simshow(Array(mask[1, :, :]))

# ╔═╡ 1ae5c61f-d34d-4618-8679-c28ecbb757bf
L = 100f-6

# ╔═╡ eaf74b38-3aae-4651-8083-930ff3ae8eed
λ = 405f-9

# ╔═╡ d65698b4-f803-4706-ab1e-b905b7983850
Zygote.gradient(x -> sum(abs2.(1 .+ repeat(x, 1,1,41))), rand(2,3))

# ╔═╡ e36ca93a-3831-4c6b-a903-c1a73e005bfb
Zygote.refresh()

# ╔═╡ 6e681b2d-8510-4970-b8f8-98840af0988e
angles = deg2rad.(range(0, 360, 100))

# ╔═╡ 9f0d2df4-e516-4b2e-9361-9cfceb6e262c
begin
	patterns_0 = togoc(ones(Float32, (size(target,1), size(target,2), size(angles, 1))));
	#patterns_0[20:40, 20:40, :] .= 1
end

# ╔═╡ 3bc8c469-6e88-4131-a5e1-409dcb447031
z = togoc(range(0, L, size(patterns_0, 1)));

# ╔═╡ b271f58d-8195-4cc2-8682-160bc531c744
AS, _ = Angular_Spectrum(repeat(0im .+ patterns_0[:, :, 1], 1, 1, size(target, 1)), z, λ, L)

# ╔═╡ eb53a1a9-5fc9-4e65-a326-0a9c816dcb71
AS_abs2(x) = abs2.(AS(repeat(0im .+ abs2.(x), 1, 1, size(target, 1)))[1])

# ╔═╡ 14db8343-a033-4241-b16f-e46de1e59a27
Zygote._pullback(AS_abs2, patterns_0[:, :, 1])[2]#(
					#imrotate(CUDA.rand(64,64,64), 0.1f0, adjoint=true)
					#)

# ╔═╡ 0254053e-2859-4342-b620-6bd31805ce00
Zygote.gradient(x -> sum(AS_abs2(x)), patterns_0[:, :, 1])

# ╔═╡ bfec391c-7fad-44ac-9d9e-b4dc923427af
AS_abs2(patterns_0[:, :, 1]);

# ╔═╡ b216f6bc-25dd-4b98-b744-d8f88e0567c6
AS_abs2(patterns_0[:, :, 1]);

# ╔═╡ e3f9bf50-26de-467e-b4f3-54eb753278ab
f, g! = make_fg!(fwd, AS_abs2, angles, target)

# ╔═╡ 3748e3d2-7ebf-496e-83fb-996d6d1a025d
simshow(Array(target[:, :, 25]))

# ╔═╡ 00891090-ade5-4b58-8f4f-45177957944b
fwd(patterns_0, AS_abs2, angles) |> size

# ╔═╡ 217481d9-5258-4c5b-9925-14349e61e1fe
sum(fwd(patterns_0, AS_abs2, angles), dims=(1,2))[:]

# ╔═╡ 95b9ed14-1a62-4f55-b3f2-df9256c4cefe
sum(gradient(x -> sum(fwd(x, AS_abs2, angles)), patterns_0)[1])

# ╔═╡ 3cf7ca46-fe1a-4c0e-81f4-eb836acb5329


# ╔═╡ bc83c7f1-e50c-4d2e-b4d1-9edaa56da33b
CUDA.@time CUDA.@sync f(patterns_0)

# ╔═╡ 542f517b-81ae-44e3-9cd5-32377153700f
CUDA.@time CUDA.@sync g!(copy(patterns_0), patterns_0 .+ 1);

# ╔═╡ 318c0c63-ab4f-4b80-9029-508dddfa0821
sum(g!(zero.(patterns_0), patterns_0 .+ 1f0))

# ╔═╡ 2f790a45-7594-4dc2-a06f-6ff48355fc42
Zygote.refresh()

# ╔═╡ 41897c92-462e-479c-94ee-5a8f60503343
md"# Optimize"

# ╔═╡ d2fdb8ae-ad9a-43ef-84ab-f3e5f15337b0
CUDA.@time res = Optim.optimize(f, g!, patterns_0, ConjugateGradient(),
                                 Optim.Options(iterations = 10,  
                                               store_trace=true))

# ╔═╡ 7e992329-d52b-415f-a305-35288a8dd1f7
@bind iangle Slider(1:size(angles, 1), show_value=true)

# ╔═╡ ce946b50-aa9b-418d-9fdd-fa8d3d1ae315
simshow(abs2.(Array(res.minimizer[:, :, iangle])), γ=1)

# ╔═╡ 6760558a-888a-4576-979a-fd6ca2b6126c
@bind thresh Slider(0.0:0.01:2, show_value=true)

# ╔═╡ 580b6b33-2364-4764-a226-6554a3c2e184
[simshow(Array(fwd(res.minimizer, AS_abs2, angles)[25, :, :]), γ=1) simshow(Array(fwd(res.minimizer, AS_abs2, angles)[25, :, :]) .> thresh, γ=1)]

# ╔═╡ d4485912-e3d3-4078-8b6f-c4e0a33ca3e7
b = [1.0 2; 3 4]

# ╔═╡ d8c28969-27e8-4c95-8a19-3469ae5f0e2f
Zygote.gradient(x -> sum(sin.(abs2.(x .* 5))), b)[1] 

# ╔═╡ 4f86cfad-26c8-4baf-a47f-3539d8e82694
cos.(b) .* (2 .* b)

# ╔═╡ e7fa025e-3985-4202-bb67-88c3b7c79d47
2 .* b

# ╔═╡ cf336478-3b1a-4546-8b96-b6784b16ab69
  Δu = real(ΔΩ)
                return (NoTangent(), 2Δu*z)
            end

# ╔═╡ Cell order:
# ╠═62673cc2-9b5b-11ee-2687-f3c65e3a77fa
# ╠═9b0fe39d-d340-40bf-bdb7-22a57bae1885
# ╠═af984a5b-7834-4823-b93a-c5d07e3f95a4
# ╠═e689c752-0c65-44de-b998-b6ecbb3b949c
# ╠═66ca565d-83ca-46f2-a13d-9667c0d2aeee
# ╠═c84b69d7-086c-445b-afe0-9724b780d806
# ╠═cd884189-1d50-4468-ad69-5a679a6a81af
# ╠═35d0680c-f075-4113-8720-bf447b8f0b3d
# ╠═87668471-f7e2-4ace-a95a-82a3f089b447
# ╠═e9e23d62-0779-4d63-8485-2ff5677cd8ba
# ╠═067d9ac3-4bdc-44e5-87c2-e0936cbcd8f8
# ╠═226dfbbe-2277-4ffe-b15c-8f2e1d434a86
# ╠═a628b053-7d6a-4aac-80b8-1e560a0e8a49
# ╠═e6df8ac8-35b5-455e-8c5a-6195f153e4e1
# ╠═ebf4598b-ab9d-415f-95d7-7e19731ae966
# ╠═140082c8-cbe8-40d2-9e1f-a076db805156
# ╠═8a37ec9d-b984-4c7b-b0a7-435305806366
# ╠═9f0d2df4-e516-4b2e-9361-9cfceb6e262c
# ╠═380386a4-0166-4cac-b46c-ab69bbc189db
# ╠═323098fe-072f-4f17-b8e6-4144770029f0
# ╠═1ae5c61f-d34d-4618-8679-c28ecbb757bf
# ╠═eaf74b38-3aae-4651-8083-930ff3ae8eed
# ╠═3bc8c469-6e88-4131-a5e1-409dcb447031
# ╠═b271f58d-8195-4cc2-8682-160bc531c744
# ╠═14db8343-a033-4241-b16f-e46de1e59a27
# ╠═0254053e-2859-4342-b620-6bd31805ce00
# ╠═d65698b4-f803-4706-ab1e-b905b7983850
# ╠═bfec391c-7fad-44ac-9d9e-b4dc923427af
# ╠═b216f6bc-25dd-4b98-b744-d8f88e0567c6
# ╠═e36ca93a-3831-4c6b-a903-c1a73e005bfb
# ╠═6e681b2d-8510-4970-b8f8-98840af0988e
# ╠═eb53a1a9-5fc9-4e65-a326-0a9c816dcb71
# ╠═e3f9bf50-26de-467e-b4f3-54eb753278ab
# ╠═3748e3d2-7ebf-496e-83fb-996d6d1a025d
# ╠═00891090-ade5-4b58-8f4f-45177957944b
# ╠═217481d9-5258-4c5b-9925-14349e61e1fe
# ╠═95b9ed14-1a62-4f55-b3f2-df9256c4cefe
# ╠═3cf7ca46-fe1a-4c0e-81f4-eb836acb5329
# ╠═bc83c7f1-e50c-4d2e-b4d1-9edaa56da33b
# ╠═542f517b-81ae-44e3-9cd5-32377153700f
# ╠═318c0c63-ab4f-4b80-9029-508dddfa0821
# ╠═2f790a45-7594-4dc2-a06f-6ff48355fc42
# ╟─41897c92-462e-479c-94ee-5a8f60503343
# ╠═d2fdb8ae-ad9a-43ef-84ab-f3e5f15337b0
# ╠═7e992329-d52b-415f-a305-35288a8dd1f7
# ╠═ce946b50-aa9b-418d-9fdd-fa8d3d1ae315
# ╠═6760558a-888a-4576-979a-fd6ca2b6126c
# ╠═580b6b33-2364-4764-a226-6554a3c2e184
# ╠═d4485912-e3d3-4078-8b6f-c4e0a33ca3e7
# ╠═d8c28969-27e8-4c95-8a19-3469ae5f0e2f
# ╠═4f86cfad-26c8-4baf-a47f-3539d8e82694
# ╠═e7fa025e-3985-4202-bb67-88c3b7c79d47
# ╠═cf336478-3b1a-4546-8b96-b6784b16ab69
