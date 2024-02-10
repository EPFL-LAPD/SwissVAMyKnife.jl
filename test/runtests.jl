using SwissVAMyKnife

using Test
using Optim, IndexFunArrays
using ChainRulesTestUtils

@testset "Simple Optimizations" begin


    sz2 = (32, 32, 2)
    target2 = box(Float32, sz2, (17, 17, 1)) .-  box(Float32, sz2, (9, 9, 1));
    angles2 = range(0, 2π, 64)
    optimizer2 = GradientBased(optimizer=Optim.LBFGS(), options=Optim.Options(iterations=10, store_trace=true))
    geometry2 = ParallelRayOptics(angles2, nothing)


    @test target2 == (0.7 .< optimize_patterns((target2), geometry2, optimizer2, LossThreshold(thresholds=(0.65, 0.75)))[2])
    @test target2 == (0.45 .< optimize_patterns((target2), geometry2, optimizer2, LossThreshold(thresholds=(0.4, 0.5)))[2])

    geometry_vial = VialRayOptics(
    	angles=angles2,
    	μ=2/256,
    	R_outer=8e-3,
    	R_inner=7.5e-3,
    	n_vial=1.5,
    	n_resin=1.48
    )
    @test target2 == (0.7 .< optimize_patterns((target2), geometry_vial, optimizer2, LossThreshold(thresholds=(0.65, 0.75)))[2])
    @test target2 == (0.45 .< optimize_patterns((target2), geometry_vial, optimizer2, LossThreshold(thresholds=(0.4, 0.5)))[2])

end


@testset "test rrule of custom loss" begin
    l = LossThreshold(sum_f=abs2, thresholds=(0.4, 0.94))

    x = randn((4,4,4))
    target = x .> 0.5
    test_rrule(l ⊢ ChainRulesTestUtils.NoTangent(), x, target ⊢ ChainRulesTestUtils.NoTangent())
end
