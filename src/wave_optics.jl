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
