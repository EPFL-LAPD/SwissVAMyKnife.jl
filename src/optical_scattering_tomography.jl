export OSTRayOptics, reconstruct_OST

"""

Type to represent a ray optical approach where refraction and reflection intensity loss at the glass vial is considered.
This is used for the reconstruction of the Optical Scattering Tomography (OST) setup
Forward model is the (attenuated) Radon transform.

- `angles` is a range or `Vector` (or `CuVector`) storing the illumination angles.
- `μ` is the absorption coefficient of the resin in units of pixels.
- So `μ=0.1` means that after ten pixels of propagation the intensity is `I(10) = I_0 * exp(-10 * 0.1)`.
- `R_outer` is the outer radius of the glass vial.
- `R_inner` is the inner radius of the glass vial.
- `camera_diameter` is the diameter of the camera along the vial radius. So this is not the height along the rotation axis!
- `n_vial` is the refractive index of the glass vial.
- `n_resin` is the refractive index of the resin.
- `polarization=PolarizationRandom()` is the polarization of the light. See [`Polarization`](@ref) for the options. 


# Examples
```jldoctest
julia> OSTRayOptics(angles=range(0,2π, 501)[begin:end-1],
                     μ=nothing,
                     R_outer=6e-3,
                     R_inner=5.5e-3,
                     camera_diameter=2 * R_outer,
                     n_vial=1.47,
                     n_resin=1.48,
                     polarization=PolarizationRandom()
                     )
```
"""
@with_kw struct OSTRayOptics{T, ToN, A, P} <: PropagationScheme
    angles::A
    μ::ToN=nothing
    R_outer::T=8e-3
    R_inner::T=7.6e-3
    n_vial::T
    n_resin::T
    camera_diameter::T=2 * R_outer
    polarization::P=PolarizationRandom()
end





function prepare_OST_geometry(target::AbstractArray{T}, ps::VialRayOptics) where T

    N = iseven(size(target, 1)) ? T(size(target, 1) - 1) : T(size(target, 1))
    radius_pixel = T(N / 2)

    in_height_N = T(round(Int, (N  * ps.DMD_diameter / (2 * ps.R_outer))÷2))
    in_height = range(-in_height_N, in_height_N, Int(2 * in_height_N + 1))
    in_height_si_units = in_height ./ (radius_pixel) .* T(ps.R_outer)
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

    return geometry 
end



function reconstruct_OST(measurements, geometry_vial; N_photons=10000, λ=1f-5, iterations=20)
    target = similar(measurements, size(measurements, 1) + 1, size(measurements, 1) + 1)
    geometry = SwissVAMyKnife.prepare_OST_geometry(target, geometry_vial)

    measurements = measurements ./ N_photons
    fg! = make_fg!(x -> radon(x, geometry_vial.angles; geometry), measurements, regularizer=reg, λ=1f-5)
    reg(x) = sum(sqrt.((circshift(x, (1,0,0)) .- x).^2 .+ (circshift(x, (0,0,1)) .- x).^2 .+ 1f-8))

    init0 = similar(target)
    fill!(init0, 1)
    res = Optim.optimize(Optim.only_fg!(fg!), init0, LBFGS(),
                                 Optim.Options(iterations = iterations,
                                               store_trace=true))
    return res.minimizer, res
end



function _OST_make_fg!(fwd_operator, measurement; λ=0.01f0, regularizer=x -> zero(eltype(x)))

	f(x) = sum(abs2, sqrt.(max.(0, fwd_operator(x)) .+ 3f0/8f0) .- sqrt.(3f0 / 8f0 .+ measurement)) + length(measurement) * λ * regularizer(x)

 	# some Optim boilerplate to get gradient and loss as fast as possible
    fg! = let f=f
        function fg!(F, G, x)
            # Zygote calculates both derivative and loss
            if G !== nothing
                y, back = Zygote.withgradient(f, x)
                # calculate gradient
                G .= back[1]
                if F !== nothing
                    return y
                end
            end
            if F !== nothing
                return f(x)
            end
        end
    end
	return fg!
end
