export optimize_patterns

export OSMO, GradientBased


struct OSMO{T} <: OptimizationScheme
    iterations::Integer
    thresholds::Tuple{T, T}

    function OSMO(; iterations=10, thresholds=(0.7f0, 0.8f0))
        return new{typeof(thresholds[1])}(iterations, thresholds)
    end
end


struct GradientBased{O, I} <: OptimizationScheme
    optimizer::O
    options::I
    
    function GradientBased(; optimizer=LBFSG(), options=Optim.options(iterations=30, store_trace=true))
        return new{typeof(optimizer), typeof(options)}(optimizer, options)
    end
end



"""
    optimize_patterns(target, angles, thresholds=(0.7f0, 0.8f0),
                      method=:radon, μ=nothing,
                      optimizer=LBFGS(), iterations=30)


"""
function optimize_patterns(target, ps::WaveOptics, op::GradientBased)
    angles = ps.angles
    μ = ps.μ
    L = ps.L
    λ = ps.λ
    z = ps.z

    sum_f = op.sum_f
    optimizer = op.optimizer
    iterations = op.iterations
    loss = op.loss
    thresholds = op.thresholds

    patterns_guess = (max.(0, radon(RadonKA.filtered_backprojection(radon(target, angles, μ), angles, μ), angles, μ)))
    
    patterns_0 = similar(target, (size(target, 1), size(target, 2), size(angles, 1)))
    fill!(patterns_0, 0)
    patterns_0[:, 2:end, :] .= permutedims(patterns_guess, (3, 1, 2))
    patterns_0 ./= maximum(patterns_0) .* 0.0001f0 
    
    x = similar(patterns_0, (size(patterns_0), 1))
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
    fg! = make_fg!(fwd2, target_permuted, thresholds; sum_f, loss)
    
    # just get the max scaling value
    patterns_0 .= 0.01f0
    
    res = Optim.optimize(Optim.only_fg!(fg!), patterns_0, optimizer, )
    
    printed = fwd2(res.minimizer)
    patterns = NNlib.relu.(res.minimizer)
    
    printed_perm = permutedims(printed, (3, 2, 1));
    patterns_perm = permutedims(patterns, (1, 3, 2));
    return patterns_perm, printed_perm, res
end



"""
"""
function make_fg!(fwd, target, loss)
    mask = similar(target, Bool, (size(target, 1), size(target, 2)))
	mask[:, :] .= rr2(eltype(target), (size(target)[1:2]..., )) .<= (size(target, 1) ÷ 2  - 1)^2

	notobject = iszero.(target)
	isobject = isone.(target)

    f = let loss=loss, fwd=fwd, notobject=notobject, isobject=isobject
        function f(x::AbstractArray{T}) where T
            return loss(fwd(x), isobject, notobject)
		end
    end

    # some Optim boilerplate to get gradient and loss as fast as possible
    fg! = let f=f
        function fg!(F, G, x) 
            # Zygote calculates both derivative and loss, therefore do everything in one step
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
