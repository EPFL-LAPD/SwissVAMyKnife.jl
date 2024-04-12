export LossTarget, LossThreshold, LossThresholdSparsity 

"""
    LossTarget

Abstract type for loss functions. 

List of implemented loss functions:
* `LossThreshold`

"""
abstract type LossTarget end

"""
    LossThreshold(;sum_f=abs2, thresholds=(0.65f0, 0.75f0))

Loss function for polymerization. 
Keeps the object voxels in the range `[T_U, 1]` and the empty space in the range `[0, T_L]`.

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

julia> target = [1, 0, 1]
3-element Vector{Int64}:
 1
 0
 1

julia> l(x, target, nothing)
0.022499999999999975

julia> (0.7 - 0.55)^2
0.022499999999999975

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
    LossThresholdSparsity(;sum_f=abs2, thresholds=(0.65f0, 0.75f0), λ=0.001f0)

Loss function for polymerization. 
Keeps the object voxels in the range `[T_U, 1]` and the empty space in the range `[0, T_L]`.
Also it avoids that the patterns are too sparse with a regularization term.

\$\$\\mathcal{L} = \\underbrace{\\sum_{v \\,\\in\\,\\text{object}} |\\text{ReLu}(T_U - I_v)|^K}_\\text{force object polymerization} + \$\$
\$\$+\\underbrace{\\sum_{v\\,\\notin\\,\\text{object}} |\\text{ReLu}(I_v - T_L) |^K}_{\\text{keep empty space unpolymerized}} +\$\$
\$\$+\\underbrace{\\sum_{v \\,\\in\\,\\text{object}} |\\text{ReLu}(I_v - 1)|^K}_{\\text{avoid overpolymerization}}\$\$
\$\$+\\underbrace{\\sum_{p \\,\\in\\,\\text{patterns}} |P_p)|^4}_{\\text{avoid sparse patterns}}\$\$

* The default `K=2` corresponds to `sum_f=abs2`.
* `(T_L, T_U) = thresholds`
* `λ` is a regularization term weight to avoid sparse patterns.
* `sparsity_sum_f` is the sum function to compute the sparsity of the patterns. Default `x -> x^4`.

"""
struct LossThresholdSparsity{F, T, F2} <: LossTarget 
    sum_f::F
    thresholds::Tuple{T, T}
    λ::T
    sparsity_sum_f::F2
    function LossThresholdSparsity(; sum_f=abs2, thresholds=(0.8f0, 0.9f0), λ=1f-9, sparsity_sum_f=x -> x^4)
        return new{typeof(sum_f), typeof(thresholds[1]), typeof(sparsity_sum_f)}(sum_f, thresholds, λ, sparsity_sum_f)
    end
end


function (l::LossThresholdSparsity)(x::AbstractArray{T}, target, patterns) where T
     return @inbounds (sum(l.sum_f.(NNlib.relu.(T(l.thresholds[2]) .- x)    .* target) .+ 
                           l.sum_f.(NNlib.relu.(x .- T(1))                  .* target) .+
                           l.sum_f.(NNlib.relu.(x .- T(l.thresholds[1]))    .* (T(1) .- target))) + 
                        T(l.λ) * sum(l.sparsity_sum_f, patterns))
end


"""
    lesson learnt from this: don't to x[isobject] where isobject would be a boolean array.
    rather express it with arithmetics. this is much faster

"""
function (l::LossThreshold)(x::AbstractArray{T}, target, patterns) where T
     return @inbounds (sum(l.sum_f.(NNlib.relu.(T(l.thresholds[2]) .- x)    .* target) .+ 
                           l.sum_f.(NNlib.relu.(x .- T(1))                  .* target) .+
                           l.sum_f.(NNlib.relu.(x .- T(l.thresholds[1]))    .* (T(1) .- target))))

    #return foldl((acc, t) -> acc[1] + abs2(NNlib.relu(T(l.thresholds[2]) - t[1]) * t[2]) + 
    #                                abs2(NNlib.relu(t[1] - T(1)) * t[2]) +
    #                                abs2(NNlib.relu(t[1] - T(l.thresholds[1])) * (T(1) - t[2])),
    #                                zip(x, target))
    #return mapreduce((x, t) -> abs2(NNlib.relu(T(l.thresholds[2]) - x) * t) + 
    #                                abs2(NNlib.relu(x - T(1)) * t) +
    #                                abs2(NNlib.relu(x - T(l.thresholds[1])) * (T(1) - t)),
    #                +, x, target)
end


"""
    custom rules for the abs2 loss function (default).
    no real speed gain but much less memory consumption 
"""
function ChainRulesCore.rrule(l::LossThreshold{typeof(abs2), TT}, x::AbstractArray{T}, target, patterns) where {T, TT}
    res = l(x, target, patterns)
    function pb(y)
        y = unthunk(y)
        g = @inbounds (2 .* y .* ((.- SwissVAMyKnife.NNlib.relu.(T(l.thresholds[2]) .- x) .* target) .+  
                                  (SwissVAMyKnife.NNlib.relu.(x .- Int(1))                .* target) .+ 
                                  (SwissVAMyKnife.NNlib.relu.(x .- T(l.thresholds[1])) .* (1 .- target))))
        return NoTangent(), g, NoTangent(), NoTangent()
    end
    return res, pb
end
