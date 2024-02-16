module SwissVAMyKnife

using WaveOpticsPropagation
using RadonKA
using DiffImageRotation
using IndexFunArrays
using ChainRulesCore
using Optim
using Zygote
using CUDA
using Plots
using FFTW
using HDF5
using FileIO
using ImageShow
using Parameters
using NNlib
using NDTools
using FourierTools
using Statistics
using Pkg
using Pkg.Artifacts
using JLD2


export PropagationScheme, OptimizationScheme

"""
    abstract type PropagationScheme end

List of possible schemes:
* [`ParallelRayOptics`](@ref)
* [`WaveOptics`](@ref)
"""
abstract type PropagationScheme end

"""
    abstract type OptimizationScheme end

List of possible schemes:
## [`GradientBased`](@ref)
Supported for all `<:PropagationScheme`.

## [`OSMO`](@ref)
Supported only for `ParallelRayOptics`
"""
abstract type OptimizationScheme end


include("utils.jl")
include("loss.jl")
include("optimization.jl")
include("wave_optics.jl")
include("ray_optics.jl")


end # module SwissVAMyKnife
