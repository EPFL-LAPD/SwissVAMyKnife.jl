export optimize_patterns
export printing_errors
export WaveOpticsProblem

export PropagationScheme, OptimizationParams
export RayOptics, WaveOptics
export OSMO, GradientBased


abstract type PropagationScheme end
abstract type OptimizationParams end

@with_kw struct WaveOptics{T2, T, A, ToN} <: PropagationScheme
    z::T2
    λ::T
    L::T
    μ::ToN
    angles::A
end

struct RayOptics{T, A} <: PropagationScheme
    angles::A
    μ::T
end


struct OSMO{I<:Integer, T} <: OptimizationParams
    iterations::I
    step_size::T
end

struct GradientBased{O, I<:Integer, F, L, T} <: OptimizationParams
    optimizer::O
    iterations::I
    sum_f::F
    loss::L
    thresholds::Tuple{T, T}
    
    function GradientBased(; optimizer=LBFSG(), iterations=30,
                             sum_f=abs2, loss=:object_space, thresholds=(0.65f0, 0.75f0))
        return new{typeof(optimizer), typeof(iterations), 
                   typeof(sum_f), typeof(loss), typeof(thresholds[1])}(optimizer, iterations, sum_f, loss, thresholds)
    end
end




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
    patterns_0 .= 0.1
    
    res = Optim.optimize(Optim.only_fg!(fg!), patterns_0, optimizer, 
                         Optim.Options(iterations=iterations, store_trace=true))
    
    printed = fwd2(res.minimizer)
    patterns = NNlib.relu.(res.minimizer)
    
    printed_perm = permutedims(printed, (3, 2, 1));
    patterns_perm = permutedims(patterns, (1, 3, 2));
    return patterns_perm, printed_perm, res
end



function optimize_patterns(target, ps::RayOptics, op::GradientBased)

end


# OSMO 
function optimize_patterns(target, ps::RayOptics, op::OSMO)

end



function optimize_patterns(target, angles; thresholds=(0.7f0, 0.8f0), method=:radon_iterative, 
                           μ=nothing, optimizer=LBFGS(), iterations=30,
                           sum_f=abs2, loss=:object_space,
                           z = nothing, λ=405f-9, L=nothing
                )
    if method == :radon
        fwd = let angles=angles, μ=μ
            fwd(x) = iradon(abs2.(x), angles, μ)
        end

        fg! = make_fg!(fwd, target, thresholds; sum_f, loss)
       
        # initial guess is the zero clipped filtered backprojection
        rec0 = (max.(0, radon(RadonKA.filtered_backprojection(radon(target, angles, μ), angles, μ), angles, μ)))

        #rec0 = radon(target, angles, μ)
        target2 = iradon(rec0, angles, μ)
        rec0 ./= maximum(target2)
        rec0 .= sqrt.(rec0)

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
    else
        throw(ArgumentError("No method such as $method"))
    end
end





"""
    make_fg!(fwd, target, thresholds=(0.7, 0.8), sum_f=abs2, loss)

`fwd` is the forward model to distribute intensity into space
`target` is an array with either 1s or 0s to mark the object
`thresholds` the thresholds we are optimizing for
`sum_f` is the summation function at the end. abs2 or abs work well
"""
function make_fg!(fwd, target, thresholds; sum_f=abs2, loss)
    mask = similar(target, Bool, (size(target, 1), size(target, 2)))
	mask[:, :] .= rr2(eltype(target), (size(target)[1:2]..., )) .<= (size(target, 1) ÷ 2  - 1)^2

	notobject = iszero.(target)
	isobject = isone.(target)

    loss_f = let sum_f=sum_f, mask=mask 
        if loss == :object_space
            @inline function loss_f3(f::AbstractArray{T}, thresholds, isobject, notobject) where T
                #f = f .* mask
                return (sum(abs2, NNlib.relu.(T(thresholds[2]) .- view(f, isobject))) + 
                        sum(abs2, NNlib.relu.(view(f, isobject) .- 1)) + 
                        sum(abs2, NNlib.relu.(view(f, notobject) .- T(thresholds[1])))
                       )
            end
        elseif loss == :leaky_relu
            @inline function loss_f2(f::AbstractArray{T}, thresholds, isobject, notobject) where T
                return (sum(relu.(T(thresholds[2]) .- view(f, isobject))) + 
                        sum(NNlib.relu.(view(f, isobject) .- 1)) + 
                        sum((NNlib.leakyrelu.(view(f, notobject) .- T(thresholds[1]), 0.01f0))))
            end
        else
            throw(ArgumentError("$loss not possible"))
        end
    end 

    loss = let fwd=fwd, mask=mask, notobject=notobject, isobject=isobject, thresholds=thresholds, loss_f=loss_f
        function L_VAM(x::AbstractArray{T}) where T
			f = fwd(x)

            #f = max.(1, f)
            # in case some parts would receive too much intensity, try to normalize
            # it to 1.
            #@show "div max"
            #m = maximum(f)
            #f = f ./ ifelse(m>1, m, 1) 
            return loss_f(f, thresholds, isobject, notobject)
		end
    end


    # some boilerplate to get gradient and loss as fast as possible
    fg! = let loss=loss
        function fg!(F, G, x) 
            # Zygote calculates both derivative and loss, therefore do everything in one step
            if G !== nothing
                # no clue why _pullback
                y, back = Zygote.withgradient(loss, x)
                # calculate gradient
                G .= back[1]#back(1)[2]
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



"""
    printing_errors(target, printed, thresholds)

"""
function printing_errors(target, printed, thresholds)
    isobject = target .≈ 1
    notobject = target .≈ 0
	mid_thresh = (thresholds[2] + thresholds[1]) / 2
	W_not = sum(printed[notobject] .> thresholds[1])
	W_not_is = sum(printed[notobject] .> mid_thresh)
	W_is = sum(printed[isobject] .< thresholds[2])
	
	N_not = sum(notobject)
	N_is = sum(isobject)
	
	voxels_object_wrong_printed = sum(abs.((printed .> mid_thresh)[isobject] .- target[isobject]))
	voxels_void_wrong_printed = sum(abs.((printed .> mid_thresh)[notobject] .- target[notobject]))

	#voxels_object_wrong_printed / N_is, W_not_is / N_not, W_not / N_not, W_is / N_is

	@info "Object pixels not printed $(round(voxels_object_wrong_printed / N_is * 100, digits=4))%"
	@info "Void pixels falsely printed $(round(voxels_void_wrong_printed / N_not * 100, digits=4))%"
end
