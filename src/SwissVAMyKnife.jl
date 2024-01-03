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

include("ray_optics.jl")
include("wave_optics.jl")
include("optimization.jl")


export plot_intensity 




function plot_intensity(target, object_printed, thresholds)
    plot_font = "Computer Modern"
    default(fontfamily=plot_font,
	    linewidth=2, framestyle=:box, label=nothing, grid=false)
	histogram(object_printed[target .== 0], bins=(0.0:0.01:2), xlim=(0.0, 1.0), label="dose distribution void", ylabel="voxel count", xlabel="normalized intensity",  ylim=(1, 1000000),  yscale=:log10, linewidth=1, legend=:topleft, size=(500, 350))
	histogram!(object_printed[target .== 1], bins=(0.0:0.01:1), label="dose distribution object", ylabel="voxel count", xlabel="normalized intensity",  ylim=(1, 1000000), linewidth=1,  yscale=:log10,)
	plot!([thresholds[1], thresholds[1]], [1, 10000_000], label="lower threshold", linewidth=3)
	plot!([thresholds[2], thresholds[2]], [1, 10000_000], label="upper threshold", linewidth=3)
	#plot!([chosen_threshold, chosen_threshold], [1, 30000000], label="chosen threshold", linewidth=3)
end



end # module SwissVAMyKnife
