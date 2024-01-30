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

export PropagationScheme, OptimizationScheme

abstract type PropagationScheme end
abstract type OptimizationScheme end


include("utils.jl")
include("loss.jl")
include("wave_optics.jl")
include("optimization.jl")
include("ray_optics.jl")


end # module SwissVAMyKnife
