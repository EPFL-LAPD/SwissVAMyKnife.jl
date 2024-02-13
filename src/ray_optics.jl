export ParallelRayOptics, VialRayOptics
export Polarization, PolarizationParallel, PolarizationPerpendicular, PolarizationRandom


"""
    Polarization

- `PolarizationParallel()` describes a parallel polarization.
- `PolarizationPerpendicular()` describes a perpendicular polarization. 
- `PolarizationRandom()` describes a random polarization.
"""
abstract type Polarization end

struct PolarizationParallel <: Polarization end
struct PolarizationPerpendicular <: Polarization end
struct PolarizationRandom <: Polarization end


"""
    ParallelRayOptics(angles, μ)

Type to represent the parallel ray optical approach.
This is suited for a printer with an index matching bath.
This is equivalent to an inverse (attenuated) Radon transform as the forward model of the printer. 


- `angles` is a range or `Vector` (or `CuVector`) storing the illumination angles.
- `μ` is the absorption coefficient of the resin in units of pixels.
   So `μ=0.1` means that after ten pixels of propagation the intensity is `I(10) = I_0 * exp(-10 * 0.1)`.

See also [`VialRayOptics`](@ref) for a printer without index matching bath.

# Examples
```jldoctest
julia> ParallelRayOptics(range(0, 2π, 401)[begin:end-1], 1 / 256)
ParallelRayOptics{Float64, StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}}(0.0:0.015707963267948967:6.267477343911637, 0.00390625)

julia> ParallelRayOptics(range(0, 2π, 401)[begin:end-1], nothing)
ParallelRayOptics{Nothing, StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}}(0.0:0.015707963267948967:6.267477343911637, nothing)
```
"""
struct ParallelRayOptics{T, A} <: PropagationScheme
    angles::A
    μ::T
end


"""

Type to represent a ray optical approach where refraction and reflection intensity loss at the glass vial is considered.
This is equivalent to an inverse (attenuated) Radon transform as the forward model of the printer. 


- `angles` is a range or `Vector` (or `CuVector`) storing the illumination angles.
- `μ` is the absorption coefficient of the resin in units of pixels.
- So `μ=0.1` means that after ten pixels of propagation the intensity is `I(10) = I_0 * exp(-10 * 0.1)`.
- `R_outer` is the outer radius of the glass vial.
- `R_inner` is the inner radius of the glass vial.
- `n_vial` is the refractive index of the glass vial.
- `n_resin` is the refractive index of the resin.
- `polarization=PolarizationRandom()` is the polarization of the light. See [`Polarization`](@ref) for the options. 


# Examples
```jldoctest
julia> VialRayOptics(angles=range(0,2π, 501)[begin:end-1],
                     μ=nothing,
                     R_outer=6e-3,
                     R_inner=5.5e-3,
                     n_vial=1.47,
                     n_resin=1.48,
                     polarization=PolarizationRandom()
                     )
VialRayOptics{Float64, Nothing, StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}, PolarizationRandom}
  angles: StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}
  μ: Nothing nothing
  R_outer: Float64 0.006
  R_inner: Float64 0.0055
  n_vial: Float64 1.47
  n_resin: Float64 1.48
  polarization: PolarizationRandom PolarizationRandom()
```
"""
@with_kw struct VialRayOptics{T, ToN, A, P} <: PropagationScheme
    angles::A
    μ::ToN=nothing
    R_outer::T=8e-3
    R_inner::T=7.6e-3
    n_vial::T
    n_resin::T
    polarization::P=PolarizationRandom()
end



"""
    optimize_patterns(target, ps::{VialRayOptics, ParallelRayOptics}, op::GradientBased, loss::LossTarget)

Function to optimize a `target` volume.
This method returns the optimized patterns, the printed intensity and the optimization result.

See [`VialRayOptics`](@ref) how to specify the geometry of the vial.
See [`ParallelRayOptics`](@ref) how to specify the geometry of the vial.

See [`PropagationScheme`](@ref) for the options for the different propagation schemes.
See [`OptimizationScheme`](@ref) for the options for the different optimization schemes.
See [`LossTarget`](@ref) for the options for the different loss functions.

# Examples
```julia
julia> patterns, printed_intensity, res = optimize_patterns(target, VialRayOptics(angles=range(0,2π, 501)[begin:end-1],
                     μ=nothing,
                     R_outer=6e-3,
                     R_inner=5.5e-3,
                     n_vial=1.47,
                     n_resin=1.48,
                     polarization=PolarizationRandom()
                     ), GradientBased(), LossThreshold())
```
"""
function optimize_patterns(target::AbstractArray{T}, ps::Union{VialRayOptics, ParallelRayOptics}, op::GradientBased, loss::LossThreshold) where T
    if T == Float64 && target isa CuArray
        @warn "Target seems to be Float64. For CUDA it is recommended to use a Float32 element type"
    end

    fwd, pat0 = _prepare_ray_forward(target, ps)
    # create loss evaluation and gradient function
    fg! = make_fg!(fwd, target, loss)
    
    # optimize
    # very low initialization
    # 0 fails with optim
    pat0 .= 0.001
    res = Optim.optimize(Optim.only_fg!(fg!), pat0, op.optimizer, op.options)
    
    # post processing
    patterns = NNlib.relu(res.minimizer) 
    printed_intensity = fwd(res.minimizer) 
    return patterns, printed_intensity, res
    # return permutedims(patterns, (3,2,1)), printed_intensity, res
end



function _prepare_ray_forward(target::AbstractArray{T}, ps::ParallelRayOptics) where T
    geometry = RadonParallelCircle(size(target, 1), -(size(target,1) -1)÷2:1:(size(target,1) -1)÷2)
    pat0 = radon(target, ps.angles, μ=ps.μ, geometry=geometry)
    fwd = let angles=ps.angles, μ=ps.μ, geometry=geometry
        fwd(x) = backproject(NNlib.relu.(x) ./ length(angles), angles; μ, geometry)
    end
    return fwd, pat0
end



function _prepare_ray_forward(target::AbstractArray{T}, ps::VialRayOptics) where T

    N = iseven(size(target, 1)) ? T(size(target, 1) - 1) : T(size(target, 1))
    radius_pixel = T(N ÷ 2)
    in_height = range(-N÷2, N÷2, Int(N))
    in_height_si_units = in_height ./ (radius_pixel .+ T(0.1)) .* T(ps.R_outer)
    # find the intersection with the glass vials
    # return both the entrance intersection and exit intersection
    heights = distort_rays_vial.(in_height,
	                radius_pixel,
                    radius_pixel / T(ps.R_outer) * T(ps.R_inner),
                    T(ps.n_vial),
                    T(ps.n_resin))


    weights = T.(_select_transmission_coefficient(_fresnel_weights(ps, in_height_si_units)..., ps.polarization))
    # unzip
    in_height = map(x -> x[1], heights)
    out_height = map(x -> x[2], heights)
    # RadonKA.jl
    geometry = RadonFlexibleCircle(size(target, 1), in_height, out_height, weights)

    # create forward model
    fwd = let angles=ps.angles, μ=ps.μ, geometry=geometry
        fwd(x) = backproject(NNlib.relu.(x) ./ length(angles), angles; μ, geometry)
    end

    pat0 = radon(target, ps.angles; μ=ps.μ, geometry)
    return fwd, pat0
end


"""
https://en.wikipedia.org/wiki/Fresnel_equations#Power_(intensity)_reflection_and_transmission_coefficients
"""
function _fresnel_weights(ps::VialRayOptics, in_height)
    # fix any bugs in the code below
    # air
    n₁ = 1 
    n₂ = ps.n_vial
    
    θᵢ = @. asin(in_height / ps.R_outer)
    θₜ = @. asin(n₁ / n₂ * sin(θᵢ))
    # fresnel equation for reflection
    Rs = @. abs2((n₁ * cos(θᵢ) - n₂ * cos(θₜ)) / (n₁ * cos(θᵢ) + n₂ * cos(θₜ)))
    Rp = @. abs2((n₁ * cos(θₜ) - n₂ * cos(θᵢ)) / (n₁ * cos(θₜ) + n₂ * cos(θᵢ)))
    # fresnel equation for transmission
    Ts = @. abs2(2 * n₁ * cos(θᵢ) / (n₁ * cos(θᵢ) + n₂ * cos(θₜ))) * (n₂ * cos(θₜ) / n₁ * cos(θᵢ))
    Tp = @. abs2(2 * n₁ * cos(θᵢ) / (n₁ * cos(θₜ) + n₂ * cos(θᵢ))) * (n₂ * cos(θₜ) / n₁ * cos(θᵢ))
    return 1 .- Rs, 1 .- Rp 
end

_select_transmission_coefficient(Tp, Ts, p::PolarizationParallel) = Tp
_select_transmission_coefficient(Tp, Ts, p::PolarizationPerpendicular) = Ts
_select_transmission_coefficient(Tp, Ts, p::PolarizationRandom) = (Tp .+ Ts) ./ 2


"""
    optimize_patterns(target::AbstractArray{T}, ps::ParallelRayOptics, op::OSMO) where T

Optimize patterns with the `OSMO` optimization algorithm.
This is only supported for `ParallelRayOptics`.

# Examples
```julia
julia> optimize_patterns(target, ParallelRayOptics(range(0, 2π, 401)[begin:end-1], 1 / 256), OSMO())
```
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
	sinogram = radon(img, θs; μ)
	
	if clip_sinogram
		sinogram .= max.(sinogram, 0)
	end
	
	img_recon = backproject(sinogram, θs; μ)
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

	printed = backproject(s, θs; μ)
    printed ./= maximum(printed)
    return permutedims(s, (3,2,1)), printed, losses
end



function distort_rays_vial(y::T, Rₒ::T, Rᵢ::T, nvial::T, nresin::T) where T
	y, Rₒ, Rᵢ, nvial, nresin = 	Float64.((y, Rₒ, Rᵢ, nvial, nresin))

	if iszero(y)
		return zero(T), zero(T)
	end

	α = asin(y / Rₒ)
	β = asin(sin(α) / nvial)

	x = Rₒ * cos(β) - sqrt(Rₒ^2*(cos(β)^2 - 1) + Rᵢ^2) 

	ϵ = acos((-Rₒ^2 + x^2 + Rᵢ^2) / 2 / Rᵢ / x) - Float64(π) / 2

	β_ = sign(y) * (Float64(π) / 2 - ϵ)
	γ = asin(nvial * sin(β_) / nresin)
	
	δ₁ = α - β
	δ₂ = β_ - γ
	δ_ges = δ₁ + δ₂

	y_ = abs(Rᵢ * sin(γ))
	p = sqrt(Rₒ^2 - y_^2)

	η = - (asin(p / Rₒ) - sign(y) * (Float64(π)/2 - δ_ges))
	yf = Rₒ * sin(η)
	
	yi = 2 * p * sin(δ_ges) + yf
	
	return T(yi), T(yf)
end
