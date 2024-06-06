export optimize_patterns

export OSMO, GradientBased

"""
    OSMO(; iterations=10, thresholds=(0.7f0, 0.8f0))
   
Define parameters for the OSMO optimization algorithm.
We recommend to use [`GradientBased`](@ref) instead of OSMO.

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

Define parameters for the `GradientBased` optimization algorithm.
Optim.jl is used for the optimization.
L-BFGS is the default optimizer which performs well for the optimization of the reconstruction problem.
30 iterations are used as default. The trace of the optimization is stored by default.

 # Example
```julia
julia> GradientBased(; optimizer=LBFGS(), options=Optim.Options(iterations=30, store_trace=true))
```
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
function make_fg!(fwd, target, loss, pat0)

    f = let loss=loss, fwd=fwd, target=target
        function f(x::AbstractArray{T}) where T
            return loss(fwd(x), target, x)
		end
    end

    backend = AutoTapir()
    extras = prepare_gradient(f, backend, similar(pat0)) 
    # some Optim boilerplate to get gradient and loss as fast as possible
    fg! = let f=f, backend=backend, extras=extras
        function fg!(F, G, x) 
            # Zygote calculates both derivative and loss, therefore do everything in one step
            if G !== nothing
                # y, back = Zygote.withgradient(f, x)
                y, gg = DifferentiationInterface.value_and_gradient!(f, G, backend, x, extras) 

                # calculate gradient
                # G .= back[1]
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
