export ParallelRayOptics

"""
    ParallelRayOptics(angles, μ)

Type to represent the parallel ray optical approach.
This is equivalent to an inverse Radon transform as the forward model to the printer.


`angles` is a range or `Vector` (or `CuVector`) storing the illumination angles.
`μ` is the absorption coefficient of the resin in units of pixels.
So `μ=0.1` means that after ten pixels of propagation the intensity is `I(10) = I_0 * exp(-10 * 0.1)`.

"""
struct ParallelRayOptics{T, A} <: PropagationScheme
    angles::A
    μ::T
end

"""
    optimize_patterns(target, ps::PropagationScheme, op::GradientBased, loss::LossTarget)

Abstract method to optimize a `target` volume.

See [`PropagationScheme`](@ref) for the options for the different propagation schemes.
See [`OptimizationScheme`](@ref) for the options for the different optimization schemes.
See [`LossTarget`](@ref) for the options for the different loss functions.

"""
optimize_patterns



function optimize_patterns(target::AbstractArray{T}, ps::ParallelRayOptics, op::GradientBased, loss::LossThreshold) where T
    if T == Float64 && target isa CuArray
        @warn "Target seems to be Float64. For CUDA it is recommended to use a Float32 element type"
    end
    # create forward model
    fwd = let angles=ps.angles, μ=ps.μ
        fwd(x) = iradon(NNlib.relu.(x) ./ length(angles), angles, μ)
    end
    # create loss evaluation and gradient function
    fg! = make_fg!(fwd, target, loss)
    
    # optimize
    rec0 = radon(target, ps.angles, ps.μ)
    # very low initialization
    # 0 fails with optim
    rec0 .= 0.001
    res = Optim.optimize(Optim.only_fg!(fg!), rec0, op.optimizer, op.options)
    
    # post processing
    patterns = NNlib.relu(res.minimizer) 
    printed_intensity = fwd(res.minimizer) 
    return permutedims(patterns, (3,2,1)), printed_intensity, res
end


"""
    optimize_patterns(target::AbstractArray{T}, ps::ParallelRayOptics, op::OSMO) where T

Optimize patterns with the `OSMO` optimization algorithm.
This is only supported for `ParallelRayOptics`.
"""
function optimize_patterns(target::AbstractArray{T}, ps::ParallelRayOptics, op::OSMO) where T
    if T == Float64 && target isa CuArray
        @warn "Target seems to be Float64. For CUDA it is recommended to use a Float32 element type"
    end
    iterative_optimization(target, ps.angles, ps.μ; op.thresholds, op.iterations) 
end

"""
    
iterate one step in the OSMO algorithm, allocation free.
"""
function iter!(buffer, img, θs, μ; clip_sinogram=true)
	sinogram = radon(img, θs, μ)
	
	if clip_sinogram
		sinogram .= max.(sinogram, 0)
	end
	
	img_recon = iradon(sinogram, θs, μ)
    img_recon ./= maximum(img_recon)


	buffer .= max.(img_recon, 0)
	return buffer, sinogram
end


"""

iterative optimization with the `OSMO` algorithm. 
Don't use this function, use `optimize_patterns`.
"""
function iterative_optimization(img::AbstractArray{T}, θs, μ=nothing; thresholds=(0.65, 0.75), iterations = 2) where T
	N = size(img, 1)
	fx = (-N / 2):1:(N /2 -1)
	R2D = similar(img)
	R2D .= sqrt.(fx'.^2 .+ fx.^2)

    p = plan_fft(similar(img), (1,2))
	guess = max.(0, real.(inv(p) * ((p * img) .* ifftshift(R2D, (1,2)))))
	guess ./= maximum(guess)

	loss(x) = (sum(max.(0,thresholds[2] .- x[isobject])) + sum(max.(0, x[notobject] .- thresholds[1]))) / length(x)
	#guess = copy(img)
	notobject = iszero.(img)
	isobject = isone.(img)

	losses = T[]
	buffer = copy(img)
	tmp, s = iter!(buffer, guess, θs, μ; clip_sinogram=true)
	for i in 1:iterations
		guess[notobject] .-= max.(0, tmp[notobject] .- thresholds[1])

		tmp, s = iter!(buffer, guess, θs, μ; clip_sinogram=true)

		guess[isobject] .+= max.(0, thresholds[2] .- tmp[isobject])

		push!(losses, loss(tmp))
	end

	printed = iradon(s, θs, μ)
    printed ./= maximum(printed)
    return permutedims(s, (3,2,1)), printed, losses
end
