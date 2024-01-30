export WaveOptics

@with_kw struct WaveOptics{T2, T, A, ToN} <: PropagationScheme
    z::T2
    λ::T
    L::T
    μ::ToN
    angles::A
end


"""
    optimize_patterns(target, angles, thresholds=(0.7f0, 0.8f0),
                      method=:radon, μ=nothing,
                      optimizer=LBFGS(), iterations=30)


"""
function optimize_patterns(target, ps::WaveOptics, op::GradientBased, loss::Threshold)
    angles = ps.angles
    μ = ps.μ
    L = ps.L
    λ = ps.λ
    z = ps.z


    patterns_0 = similar(target, (size(target)[1:2]..., size(angles, 1)))
    x = similar(patterns_0, (size(patterns_0, 1)))
    x .= fftpos(ps.L, size(x,1), CenterFT)
    mask = reshape((x.^2 .+ x'.^2) .<= ps.L^2, (1, size(x,1), size(x,1)))



    AS, _ = AngularSpectrum(patterns_0[:, :, 1] .+ 0im, z, λ, L, padding=false)
    AS_abs2 = let target=target, AS=AS, langles=length(angles)
            function AS_abs2(x)
                abs2.(AS(NNlib.relu.(x) .+ 0im)[1]) ./ langles
            end
    end
    
    fwd2 = let AS_abs2=AS_abs2, angles=angles
        function fwd2(x)
            fwd_wave(x, AS_abs2, angles)
        end
    end
    
    target_permuted = permutedims(target, (3, 2, 1))
    fg! = make_fg!(fwd2, target_permuted, loss)
    
    # initialize with almost 0
    patterns_0 .= 0.001f0
    
    res = Optim.optimize(Optim.only_fg!(fg!), patterns_0, op.optimizer, op.options)
   

    printed = fwd2(res.minimizer)
    patterns = NNlib.relu.(res.minimizer)
    printed_perm = permutedims(printed, (3, 2, 1));
    patterns_perm = permutedims(patterns, (1, 3, 2));
    return patterns_perm, printed_perm, res
end




"""
    fwd_wave(x, AS_abs2, angles)

`x` is the collection of patterns. `AS_abs2` propagates one pattern (one angle) into space.
Also it takes its `abs2` for intensity.

Then we rotate with the correct angle and propagate 
"""
function fwd_wave(x, AS_abs2, angles)
	intensity = copy(similar(x, real(eltype(x)), (size(x, 1), size(x, 2), size(x, 2))))
	fill!(intensity, 0)

	tmp_rot = similar(intensity)
	tmp_rot .= 0
	for (i, angle) in enumerate(angles)
		CUDA.@sync tmp = AS_abs2(view(x, :, :, i))
		CUDA.@sync intensity .+= PermutedDimsArray(imrotate!(tmp_rot, PermutedDimsArray(tmp, (2, 3, 1)), angle), (3, 1, 2))
		tmp_rot .= 0
	end
	return intensity
end


function ChainRulesCore.rrule(::typeof(fwd_wave), x, AS_abs2, angles)
	res = fwd_wave(x, AS_abs2, angles)
	pb_rotate = let AS_abs2=AS_abs2, angles=angles
	function pb_rotate(ȳ)
		grad = similar(x, real(eltype(x)), size(x, 1), size(x, 2), size(angles, 1))
		fill!(grad, 0)
		#@show sum(ȳ)
		tmp_rot = similar(res);
		tmp_rot .= 0;
		for (i, angle) in enumerate(angles)
			CUDA.@sync tmp = Zygote._pullback(AS_abs2, view(x, :, :, i))[2](
					PermutedDimsArray(DiffImageRotation.∇imrotate!(tmp_rot, PermutedDimsArray(ȳ, (2, 3, 1)), res, angle), (3, 1, 2))
			)[2]
			tmp_rot .= 0
			grad[:, :, i] .= tmp
		end
		return NoTangent(), grad, NoTangent(), NoTangent()
	end
	end
	return res, pb_rotate
 end
