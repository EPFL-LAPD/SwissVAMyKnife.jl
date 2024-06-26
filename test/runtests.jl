using SwissVAMyKnife

using Test
using Optim, IndexFunArrays
using ChainRulesTestUtils

@testset "Simple Optimizations" begin


    sz2 = (32, 32, 2)
    target2 = box(Float32, sz2, (17, 17, 1)) .-  box(Float32, sz2, (9, 9, 1));
    angles2 = range(0, 2π, 60)
    optimizer2 = GradientBased(optimizer=Optim.LBFGS(), options=Optim.Options(iterations=25, store_trace=true))
    geometry2 = ParallelRayOptics(angles=angles2, μ=nothing, DMD_diameter=16e-3)


    @test target2 == (0.7 .< optimize_patterns((target2), geometry2, optimizer2, LossThreshold(thresholds=(0.65, 0.75)))[2])
    @test target2 == (0.7 .< optimize_patterns((target2), geometry2, OSMO(iterations=50, thresholds=(0.65, 0.75)))[2])
    @test target2 == (0.45 .< optimize_patterns((target2), geometry2, optimizer2, LossThreshold(thresholds=(0.4, 0.5)))[2])


    @testset "Diffusion" begin 
        diffusion = Diffusion(1000000f0, 1f-25, 0.0001f0, 1, 1)
        pat1 = optimize_patterns((target2), geometry2, diffusion, optimizer2, LossThreshold(thresholds=(0.65, 0.75)))[1]
        pat2 = optimize_patterns((target2), geometry2, optimizer2, LossThreshold(thresholds=(0.65, 0.75)))[1] 
        @test (all(.≈(1 .+ pat1, 1 .+ pat2, rtol=0.1)))
        diffusion = Diffusion(100f-6, 1f-10, 20f0, 3, 5)
        @test target2 == (0.45 .< optimize_patterns((target2), geometry2, diffusion, optimizer2, LossThreshold(thresholds=(0.4, 0.5)))[2])
    end


    geometry_vial = VialRayOptics(
    	angles=angles2,
        μ=3/(16e-3),
    	R_outer=8e-3,
    	R_inner=7.5e-3,
    	n_vial=1.5,
    	n_resin=1.48
    )
    @test target2 == (0.7 .< optimize_patterns((target2), geometry_vial, optimizer2, LossThreshold(thresholds=(0.65, 0.75)))[2])
    @test target2 == (0.7 .< optimize_patterns((target2), geometry_vial, optimizer2, LossThreshold(thresholds=(0.65, 0.75)))[2])
    @test target2 !== (0.7 .> optimize_patterns((target2), geometry_vial, optimizer2, LossThreshold(thresholds=(0.65, 0.75)))[2])
    @test target2 == (0.45 .< optimize_patterns((target2), geometry_vial, optimizer2, LossThreshold(thresholds=(0.4, 0.5)))[2])
    @test target2 !== (0.45 .> optimize_patterns((target2), geometry_vial, optimizer2, LossThreshold(thresholds=(0.4, 0.5)))[2])
    
    @test target2 == (0.45 .< optimize_patterns((target2), geometry_vial, optimizer2, LossThresholdSparsity(thresholds=(0.4, 0.5)))[2])

    patterns, printed, res = optimize_patterns((target2), geometry_vial, optimizer2, LossThreshold(thresholds=(0.4, 0.5)))
    save_patterns(tempdir(), patterns, printed, angles2, target2; overwrite=true)
end



@testset "Simple wave optical simulation" begin

    sz2 = (24, 24, 24)
    target = box(Float32, sz2, (17, 17, 10)) .-  box(Float32, sz2, (9, 9, 8));

    n_resin = 1.5f0
    angles = range(0, π, 20)
    optimizer = GradientBased(optimizer=Optim.LBFGS(), options=Optim.Options(iterations=15, store_trace=true))


    L = 100f-6
    loss = LossThreshold(thresholds=(0.65, 0.75))

    optimizer = GradientBased(optimizer=Optim.LBFGS(), 		options=Optim.Options(iterations=20, store_trace=true))

    waveoptics = WaveOptics(
	z=(range(-L/2, L/2, size(target,1))), 
	L=L, 
	λ=405f-9 / n_resin, 
	μ=nothing, 
	angles=angles,
	)
    patterns, printed, res = optimize_patterns(target, waveoptics, optimizer, loss)
    @test target == (0.7 .< printed)
    @test target !== (0.7 .> printed) 

end

@testset "Simple wave optical simulation with sparse loss" begin

    sz2 = (24, 24, 24)
    target = box(Float32, sz2, (17, 17, 10)) .-  box(Float32, sz2, (9, 9, 8));

    n_resin = 1.5f0
    angles = range(0, π, 20)
    optimizer = GradientBased(optimizer=Optim.LBFGS(), options=Optim.Options(iterations=15, store_trace=true))


    L = 100f-6
    loss = LossThresholdSparsity(thresholds=(0.65, 0.75))

    optimizer = GradientBased(optimizer=Optim.LBFGS(), 		options=Optim.Options(iterations=20, store_trace=true))

    waveoptics = WaveOptics(
	z=(range(-L/2, L/2, size(target,1))), 
	L=L, 
	λ=405f-9 / n_resin, 
	μ=nothing, 
	angles=angles,
	)
    patterns, printed, res = optimize_patterns(target, waveoptics, optimizer, loss)
    @test target == (0.7 .< printed)
    @test target !== (0.7 .> printed) 

end


@testset "test rrule of custom loss" begin
    l = LossThreshold(sum_f=abs2, thresholds=(0.4, 0.94))
    l2 = LossThresholdSparsity(sum_f=abs2, thresholds=(0.4, 0.94), λ=0.01)
    x = randn((4,4,4))
    x2 = randn((4,4,4))
    target = x .> 0.5
    test_rrule(l ⊢ ChainRulesTestUtils.NoTangent(), x, target ⊢ ChainRulesTestUtils.NoTangent(), x ⊢ ChainRulesTestUtils.NoTangent())
    
    test_rrule(l2 ⊢ ChainRulesTestUtils.NoTangent(), x, target ⊢ ChainRulesTestUtils.NoTangent(), x2)
end


@testset "Refraction of glass and resin" begin
    @test SwissVAMyKnife.distort_rays_vial(-100.0, 200.0, 180.0, 1.3, 1.3 * 1.7)[1] ≈ (-103.5945990123854, 17.585052942618567)[1]
    @test SwissVAMyKnife.distort_rays_vial(-100.0, 200.0, 180.0, 1.3, 1.3 * 1.7)[2] ≈ (-103.5945990123854, 17.585052942618567)[2]
    @test SwissVAMyKnife.distort_rays_vial(-100.0, 200.0, 180.0, 1.37, 1.37 * 1.37)[1] ≈ (-102.26082855257938, -0.5207635370929073)[1]
    @test SwissVAMyKnife.distort_rays_vial(-100.0, 200.0, 180.0, 1.37, 1.37 * 1.37)[2] ≈ (-102.26082855257938, -0.5207635370929073)[2]
end


@testset "Test IoU" begin
    @test  calculate_IoU([1,0], [1,0]) ≈ 1.0
    @test  calculate_IoU([1,0], [0,1]) ≈ 0.0
    @test  calculate_IoU([1,0], [1,1]) ≈ 0.5
    @test  calculate_IoU([1,0], [0,0]) ≈ 0.0
end


@testset "test artifacts" begin
    @test size(load_example_target("3DBenchy_180")) == (180, 180, 180)
    @test size(load_example_target("3DBenchy_550")) == (550, 550, 550) 
end
