export LossTarget, Threshold 

abstract type LossTarget end

"""
    Threshold(sum_f=abs2, thresholds=(0.65f0, 0.75f0))

"""
struct Threshold{F, T} <: LossTarget 
    sum_f::F
    thresholds::Tuple{T, T}

    function Threshold(; sum_f=abs2, thresholds=(0.65f0, 0.75f0))
        return new{typeof(sum_f), typeof(thresholds[1])}(sum_f, thresholds)
    end
end


@inline function (l::Threshold)(x, isobject, notobject)
    #return (sum(x -> l.sum_f(NNlib.relu(l.thresholds[2] - x)),  view(x, isobject)) +
    #        sum(x -> l.sum_f(NNlib.relu(x - 1)              ),  view(x, isobject)) +
    #        sum(x -> l.sum_f(NNlib.relu(x - l.thresholds[1])),  view(x, notobject)))
    return (sum(abs2, NNlib.relu.(l.thresholds[2] .- view(x, isobject))) +
            sum(abs2, NNlib.relu.(view(x, isobject) .- 1)) +
            sum(abs2, NNlib.relu.(view(x, notobject) .- l.thresholds[1])))
end
