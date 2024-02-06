export LossTarget, LossThreshold 

abstract type LossTarget end

"""
    LossThreshold(;sum_f=abs2, thresholds=(0.65f0, 0.75f0))

Creates a type to calculate the following loss function:

\$\$\\mathcal{L} = \\underbrace{\\sum_{v \\,\\in\\,\\text{object}} |\\text{ReLu}(T_U - I_v)|^K}_\\text{force object polymerization} + \$\$
\$\$+\\underbrace{\\sum_{v\\,\\notin\\,\\text{object}} |\\text{ReLu}(I_v - T_L) |^K}_{\\text{keep empty space unpolymerized}} +\$\$
\$\$+\\underbrace{\\sum_{v \\,\\in\\,\\text{object}} |\\text{ReLu}(I_v - 1)|^K}_{\\text{avoid overpolymerization}}\$\$

* The default `K=2` corresponds to `sum_f=abs2`.
* `(T_L, T_U) = thresholds`


```jldoctest
julia> l = LossThreshold(thresholds=(0.5, 0.7))
LossThreshold{typeof(abs2), Float64}(abs2, (0.5, 0.7))

julia> x = [1.0, 0.0, 0.55]
3-element Vector{Float64}:
 1.0
 0.0
 0.55

julia> isobject = [true, false, true]
3-element Vector{Bool}:
 1
 0
 1

julia> notobject = .!(isobject)
3-element BitVector:
 0
 1
 0

julia> l(x, isobject, notobject)
0.022499999999999975

julia> (0.7 - 0.55)^2
0.0225
```
"""
struct LossThreshold{F, T} <: LossTarget 
    sum_f::F
    thresholds::Tuple{T, T}

    function LossThreshold(; sum_f=abs2, thresholds=(0.65f0, 0.75f0))
        return new{typeof(sum_f), typeof(thresholds[1])}(sum_f, thresholds)
    end
end


"""
   
"""
@inline function (l::LossThreshold)(x::AbstractArray{T}, isobject, notobject) where T
    #return (sum(x -> l.sum_f(NNlib.relu(l.thresholds[2] - x)),  view(x, isobject)) +
    #        sum(x -> l.sum_f(NNlib.relu(x - 1)              ),  view(x, isobject)) +
    #        sum(x -> l.sum_f(NNlib.relu(x - l.thresholds[1])),  view(x, notobject)))
    return @inbounds (sum(abs2, NNlib.relu.(T(l.thresholds[2]) .- view(x, isobject))) +
            sum(abs2, NNlib.relu.(view(x, isobject) .- 1)) +
            sum(abs2, NNlib.relu.(view(x, notobject) .- T(l.thresholds[1]))))
end
