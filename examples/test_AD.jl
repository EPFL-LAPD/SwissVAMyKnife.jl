### A Pluto.jl notebook ###
# v0.19.36

using Markdown
using InteractiveUtils

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

# ╔═╡ 66ca565d-83ca-46f2-a13d-9667c0d2aeee
use_CUDA = Ref(true && CUDA.functional())

# ╔═╡ c84b69d7-086c-445b-afe0-9724b780d806
var"@mytime" = use_CUDA[] ? CUDA.var"@time" : var"@time"

# ╔═╡ cd884189-1d50-4468-ad69-5a679a6a81af
togoc(x) = use_CUDA[] ? CuArray(x) : x

# ╔═╡ 35d0680c-f075-4113-8720-bf447b8f0b3d


# ╔═╡ 87668471-f7e2-4ace-a95a-82a3f089b447
begin
	target = togoc(zeros(Float32, 64, 64, 64));
	target[:, 20:23, :] .= 1;

	target = imrotate(target, deg2rad(50))
end;

# ╔═╡ e9e23d62-0779-4d63-8485-2ff5677cd8ba
simshow(Array(target[:, :, 16]))

# ╔═╡ 226dfbbe-2277-4ffe-b15c-8f2e1d434a86


# ╔═╡ a628b053-7d6a-4aac-80b8-1e560a0e8a49
function make_fg!(fwd, target, loss=:L2)
	mask = togoc(rr2(eltype(target), (size(target)[1:2]..., )) .<= (size(target, 1) ÷ 2  - 1)^2)
	
    L2 = let target=target, fwd=fwd
        function L2(x)
            return sum(abs2, (fwd(x) .- target) .* mask)
		end
    end

    f = let 
        if loss == :L2
            L2
        end
    end

    g! = let f=f
        function g!(G, rec)
            if !isnothing(G)
                return G .= Zygote.gradient(f, rec)[1]
            end
        end
    end
    return f, g!
end

# ╔═╡ ebf4598b-ab9d-415f-95d7-7e19731ae966
function fwd(x, AS_abs2, angles)
		
		tmp = AS_abs2(x[:, :, 1])
		intensity = imrotate(tmp, angles[begin])

		for (i, angle) in enumerat(angles[begin+1:end])		
			tmp = AS_abs2(x[:, :, i])
			intensity .+= imrotate(intensity, angles[begin])
		end	
		return intensity
end

# ╔═╡ 140082c8-cbe8-40d2-9e1f-a076db805156
Zygote._pullback(x -> abs2.(x), [1 2; 3 4])[2](rand((2,2)))

# ╔═╡ 07ae5f72-55af-4d7b-8a54-aad82b5139b5
size(target)

# ╔═╡ 9f0d2df4-e516-4b2e-9361-9cfceb6e262c
rec0 = togoc(ones(ComplexF32, (size(target,1), size(target,2), 1)));

# ╔═╡ 1ae5c61f-d34d-4618-8679-c28ecbb757bf
L = 100f-6

# ╔═╡ eaf74b38-3aae-4651-8083-930ff3ae8eed
λ = 405f-9

# ╔═╡ 3bc8c469-6e88-4131-a5e1-409dcb447031
z = togoc(range(0, L, size(rec0, 1)));

# ╔═╡ b271f58d-8195-4cc2-8682-160bc531c744
AS, _ = Angular_Spectrum(repeat(rec0, 1, 1, size(target, 1)), z, λ, L)

# ╔═╡ eb53a1a9-5fc9-4e65-a326-0a9c816dcb71
AS_abs2(x) = abs2.(AS(repeat(x, 1, 1, size(x, 1)))[1])

# ╔═╡ 8a37ec9d-b984-4c7b-b0a7-435305806366
 # is this rrule good? 
 # no @thunk and @unthunk
 function ChainRulesCore.rrule(::typeof(fwd), x, AS, angles)
     res = fwd(x, AS, angles)
	 
     function pb_rotate(ȳ)
		grad = Zygote._pullback(AS_abs2, AS_abs2(x[:, :, 1]))[2](ȳ)
		intensity = imrotate(tmp, angles[begin])
		 
		for (i, angle) in enumerat(angles[begin+1:end])		
			tmp = AS_abs2(x[:, :, i])
			intensity .+= imrotate(intensity, angles[begin])
		end	
         return NoTangent(), ad, NoTangent(), NoTangent()
     end    
     return res, pb_rotate
 end

# ╔═╡ e3f9bf50-26de-467e-b4f3-54eb753278ab
f, g! = make_fg!(fwd, target)

# ╔═╡ 3748e3d2-7ebf-496e-83fb-996d6d1a025d
simshow(Array(target[:, :, 1]))

# ╔═╡ bc83c7f1-e50c-4d2e-b4d1-9edaa56da33b
CUDA.@time CUDA.@sync f(rec0)

# ╔═╡ 38d86ed0-d075-409a-a561-96eb69b57a20
CUDA.@time CUDA.@sync gradient(f, rec0)[1]

# ╔═╡ 41897c92-462e-479c-94ee-5a8f60503343
md"# Optimize"

# ╔═╡ d2fdb8ae-ad9a-43ef-84ab-f3e5f15337b0
CUDA.@time res = Optim.optimize(f, g!, rec0, ConjugateGradient(),
                                 Optim.Options(iterations = 10,  
                                               store_trace=true))

# ╔═╡ ce946b50-aa9b-418d-9fdd-fa8d3d1ae315
simshow(Array(res.minimizer[:, :, 1]))

# ╔═╡ 580b6b33-2364-4764-a226-6554a3c2e184
simshow(Array(fwd(res.minimizer)[:, :, 20]))

# ╔═╡ Cell order:
# ╠═62673cc2-9b5b-11ee-2687-f3c65e3a77fa
# ╠═9b0fe39d-d340-40bf-bdb7-22a57bae1885
# ╠═af984a5b-7834-4823-b93a-c5d07e3f95a4
# ╠═66ca565d-83ca-46f2-a13d-9667c0d2aeee
# ╠═c84b69d7-086c-445b-afe0-9724b780d806
# ╠═cd884189-1d50-4468-ad69-5a679a6a81af
# ╠═35d0680c-f075-4113-8720-bf447b8f0b3d
# ╠═87668471-f7e2-4ace-a95a-82a3f089b447
# ╠═e9e23d62-0779-4d63-8485-2ff5677cd8ba
# ╠═226dfbbe-2277-4ffe-b15c-8f2e1d434a86
# ╠═a628b053-7d6a-4aac-80b8-1e560a0e8a49
# ╠═ebf4598b-ab9d-415f-95d7-7e19731ae966
# ╠═140082c8-cbe8-40d2-9e1f-a076db805156
# ╠═8a37ec9d-b984-4c7b-b0a7-435305806366
# ╠═07ae5f72-55af-4d7b-8a54-aad82b5139b5
# ╠═9f0d2df4-e516-4b2e-9361-9cfceb6e262c
# ╠═1ae5c61f-d34d-4618-8679-c28ecbb757bf
# ╠═eaf74b38-3aae-4651-8083-930ff3ae8eed
# ╠═3bc8c469-6e88-4131-a5e1-409dcb447031
# ╠═b271f58d-8195-4cc2-8682-160bc531c744
# ╠═eb53a1a9-5fc9-4e65-a326-0a9c816dcb71
# ╠═e3f9bf50-26de-467e-b4f3-54eb753278ab
# ╠═3748e3d2-7ebf-496e-83fb-996d6d1a025d
# ╠═bc83c7f1-e50c-4d2e-b4d1-9edaa56da33b
# ╠═38d86ed0-d075-409a-a561-96eb69b57a20
# ╟─41897c92-462e-479c-94ee-5a8f60503343
# ╠═d2fdb8ae-ad9a-43ef-84ab-f3e5f15337b0
# ╠═ce946b50-aa9b-418d-9fdd-fa8d3d1ae315
# ╠═580b6b33-2364-4764-a226-6554a3c2e184
