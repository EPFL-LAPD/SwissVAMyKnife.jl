export optimize_patterns

export OSMO, GradientBased

"""
    OSMO(; iterations=10, thresholds=(0.7f0, 0.8f0))
   
Define parameters for the OSMO optimization algorithm.

# Reference
Rackson, Charles M., et al. *Object-space optimization of tomographic reconstructions for additive manufacturing.* Additive Manufacturing 48 (2021): 102367.

"""
struct OSMO{T} <: OptimizationScheme
    iterations::Int
    thresholds::Tuple{T, T}

    function OSMO(; iterations=10, thresholds=(0.7f0, 0.8f0))
        return new{typeof(thresholds[1])}(iterations, thresholds)
    end
end

"""
    GradientBased(; optimizer=LBFSG(), options=Optim.options(iterations=30, store_trace=true))


Define type for the gradient descent based optimization scheme.
Based on Optim.jl.
"""
struct GradientBased{O, I} <: OptimizationScheme
    optimizer::O
    options::I
    
    function GradientBased(; optimizer=LBFGS(), options=Optim.Options(iterations=30, store_trace=true))
        return new{typeof(optimizer), typeof(options)}(optimizer, options)
    end
end





"""
    make_fg!(fwd, target, loss)

Define an efficient function `fg!` for the interface of Optim.jl
Internal method, do not use.
"""
function make_fg!(fwd, target, loss)
    mask = similar(target, Bool, (size(target, 1), size(target, 2)))
	mask[:, :] .= rr2(eltype(target), (size(target)[1:2]..., )) .<= (size(target, 1) ÷ 2  - 1)^2

    f = let loss=loss, fwd=fwd, target= (target .≈ 1)
        function f(x::AbstractArray{T}) where T
            return loss(fwd(x), target)
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
