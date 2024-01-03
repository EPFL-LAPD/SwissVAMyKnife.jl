export optimize_patterns

"""
    optimize_patterns(target, angles, thresholds=(0.7f0, 0.8f0),
                      method=:radon, μ=nothing,
                      optimizer=LBFGS(), iterations=30)

Optimize some `patterns` such that they print the object `target`.
`angles` should be an array of angles in radians.
`thresholds` indicates the thresholds for the loss. 


# Keywords
* method=`:radon` indidcates a radon based method. `method=:wave` indicates a wave optical model
* `iterations=30` is the number of iterations for the optimizer
* `μ` is the absorption normalized to pixel units.


"""
function optimize_patterns(target, angles; thresholds=(0.7f0, 0.8f0), method=:radon, 
                           μ=nothing, optimizer=LBFGS(), iterations=30,
                           sum_f=abs2,
                           z = nothing, λ=405f-9, L=nothing
                )
    if method == :radon
        fwd = let angles=angles, μ=μ
            fwd(x) = iradon(abs2.(x), angles, μ)
        end

        fg! = make_fg!(fwd, target, thresholds; sum_f)
       
        # initial guess is the zero clipped filtered backprojection
        rec0 = (max.(0, radon(RadonKA.filtered_backprojection(radon(target, angles, μ), angles, μ), angles, μ)))

        #rec0 = radon(target, angles, μ)
        target2 = iradon(rec0, angles, μ)
        rec0 ./= maximum(target2)
        rec0 .= sqrt.(rec0)

        #return rec0, iradon(abs2.(rec0), angles, μ), rec0 
        #rec0 = similar(target, (size(target,1) - 1, size(angles, 1), size(target, 3)))
        # dumb initial guess
        #fill!(rec0, 1)
        res = Optim.optimize(Optim.only_fg!(fg!), rec0, optimizer, 
                             Optim.Options(iterations=iterations, store_trace=true))

        patterns = abs2.(res.minimizer) 
        patterns ./= maximum(patterns)
    
        printed_intensity = fwd(res.minimizer) 
        printed_intensity ./= maximum(printed_intensity)
        return patterns, printed_intensity, res
    elseif method == :radon_iterative
        return iterative_optimization(target, angles, μ; thresholds, iterations)
    elseif method == :wave
        patterns_guess = (max.(0, radon(RadonKA.filtered_backprojection(radon(target, angles, μ), angles, μ), angles, μ)))
        
        patterns_0 = similar(target, (size(target, 1), size(target, 2), size(angles, 1)))
        patterns_0[:, 2:end, :] .= permutedims(patterns_guess, (3, 1, 2))
        @show size(patterns_0), size(patterns_guess)

        AS, _ = Angular_Spectrum(patterns_0[:, :, 1] .+ 0im, z, λ, L, padding=false)         
        AS_abs2 = let target=target, AS=AS
	            function AS_abs2(x)
		            abs2.(AS(abs2.(x) .+ 0im)[1])
	            end
        end

        fwd2 = let AS_abs2=AS_abs2, angles=angles
            function fwd2(x)
                fwd_wave(x, AS_abs2, angles)
            end
        end
       
        fg! = make_fg!(fwd2, target, thresholds; sum_f)
        
        # just get the max scaling value
        rec0 = (max.(0, radon(RadonKA.filtered_backprojection(radon(target, angles, μ), angles, μ), angles, μ)))
        target2 = iradon(rec0, angles, μ)
        max_value = maximum(target2)
        
        # normalize patterns
        patterns_0 ./= max_value 
        patterns_0 .= sqrt.(patterns_0)
        
        res = Optim.optimize(Optim.only_fg!(fg!), patterns_0, optimizer, 
                             Optim.Options(iterations=iterations, store_trace=true))

        return abs2.(res), fwd(res), res
    else
        throw(ArgumentError("No method such as $method"))
    end
end





"""
    make_fg!(fwd, target, thresholds=(0.7, 0.8), sum_f=abs2)

`fwd` is the forward model to distribute intensity into space
`target` is an array with either 1s or 0s to mark the object
`thresholds` the thresholds we are optimizing for
`sum_f` is the summation function at the end. abs2 or abs work well
"""
function make_fg!(fwd, target, thresholds; sum_f=abs2)
    mask = similar(target, Bool, (size(target, 1), size(target, 2)))
	mask[:, :] .= rr2(eltype(target), (size(target)[1:2]..., )) .<= (size(target, 1) ÷ 2  - 1)^2

	notobject = iszero.(target)
	isobject = isone.(target)

    loss = let fwd=fwd, mask=mask, notobject=notobject, isobject=isobject, thresholds=thresholds
        function L_VAM(x::AbstractArray{T}) where T
			f = fwd(x)

            # possible speed-up to avoid max here
			f = f ./ maximum(f)
			return sum(sum_f, max.(0, T(thresholds[2]) .- view(f, isobject))) + 
                   sum(sum_f, max.(0, view(f, notobject) .- T(thresholds[1])))
		end
    end


    # some boilerplate to get gradient and loss as fast as possible
    fg! = let loss=loss
        function fg!(F, G, x) 
            # Zygote calculates both derivative and loss, therefore do everything in one step
            if G !== nothing
                # no clue why _pullback
                y, back = Zygote._pullback(loss, x)
                # calculate gradient
                G .= back(1)[2]
                if F !== nothing
                    return y
                end
            end
            if F !== nothing
                return loss(x) 
            end
        end
    end
    
    return fg!
end

