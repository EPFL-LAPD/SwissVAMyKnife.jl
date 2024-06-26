export OSTRayOptics, reconstruct_OST





function prepare_OST_geometry(target::AbstractArray{T}, ps::OSTRayOptics) where T

    N = iseven(size(target, 1)) ? T(size(target, 1) - 1) : T(size(target, 1))
    radius_pixel = T(N / 2)

    in_height_N = T(round(Int, (N  * ps.camera_diameter / (2 * ps.R_outer))÷2))
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

"""
    reconstruct_OST(measurements, geometry_vial; N_photons=100_000, λ=1f-5, iterations=20)

Reconstruct the polymerization of a resin inside a glass vial using the Optical Scattering Tomography (OST) setup.
The amount of photons is set to `N_photons` and the regularization parameter is set to `λ`.
The number of iterations is set to `iterations`.

"""
function reconstruct_OST(measurements, geometry_vial; N_photons=100_000, λ=1f-5, iterations=20)
    target = similar(measurements, size(measurements, 1) + 1, size(measurements, 1) + 1, size(measurements, 3))
    geometry = SwissVAMyKnife.prepare_OST_geometry(target, geometry_vial)
    reg(x) = sum(sqrt.((circshift(x, (1,0,0)) .- x).^2 .+ (circshift(x, (0,1,0)) .- x).^2 .+ 1f-8))

    measurements = measurements ./ maximum(measurements) .* N_photons
    fg! = _OST_make_fg!(x -> radon(x, geometry_vial.angles; geometry), measurements, regularizer=reg, λ=λ)

    init0 = similar(target)
    fill!(init0, 0)
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
