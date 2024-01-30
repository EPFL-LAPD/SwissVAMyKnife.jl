export RayOptics

struct RayOptics{T, A} <: PropagationScheme
    angles::A
    μ::T
end


function optimize_patterns(target::AbstractArray{T}, ps::RayOptics, op::GradientBased, loss::Threshold) where T
    if T == Float64
        @warn "Target seems to be Float64. For CUDA it is recommended to use Float32 element type"
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


function optimize_patterns(target::AbstractArray{T}, ps::RayOptics, op::OSMO) where T
    iterative_optimization(target, ps.angles, ps.μ; op.thresholds, op.iterations) 
end

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
