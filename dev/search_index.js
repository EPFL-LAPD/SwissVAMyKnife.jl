var documenterSearchIndex = {"docs":
[{"location":"ray_derivation/#Analytic-Derivation-of-Vial-Refraction","page":"Analytic Derivation of Vial Refraction","title":"Analytic Derivation of Vial Refraction","text":"","category":"section"},{"location":"ray_derivation/","page":"Analytic Derivation of Vial Refraction","title":"Analytic Derivation of Vial Refraction","text":"If we remove the index matching bath in TVAM, the glass vial affects the ray propagation because of refraction at the glass and vial interface. Our ray optical backend RadonKA.jl can handle non-parallel ray propagation if the intersection of the rays with the outer circle are calculated at entry and exit.","category":"page"},{"location":"ray_derivation/","page":"Analytic Derivation of Vial Refraction","title":"Analytic Derivation of Vial Refraction","text":"In the following we derive analytical expressions for y_i and y_f.  Given the circular shaped vial, the angle under which a ray hits the vial is alpha = arcsin(y  R_o). Consequently, because of refraction we obtain beta=arcsin(sin(alpha)  n_textvial). The orange segment x is more inconvenient to derive but the law of cosines of triangles provides us with ","category":"page"},{"location":"ray_derivation/","page":"Analytic Derivation of Vial Refraction","title":"Analytic Derivation of Vial Refraction","text":"R_i^2 = x^2 + R_o^2 - 2 cdot x cdot R_o cosbeta","category":"page"},{"location":"ray_derivation/","page":"Analytic Derivation of Vial Refraction","title":"Analytic Derivation of Vial Refraction","text":"Solving the quadratic equation, the meaningful solution is","category":"page"},{"location":"ray_derivation/","page":"Analytic Derivation of Vial Refraction","title":"Analytic Derivation of Vial Refraction","text":"x = R_o cdot cosbeta -  sqrtR_o^2 cdot (cos(beta)^2 - 1) + R_i^2","category":"page"},{"location":"ray_derivation/","page":"Analytic Derivation of Vial Refraction","title":"Analytic Derivation of Vial Refraction","text":"Again, applying law of cosine we can obtain an expression for","category":"page"},{"location":"ray_derivation/","page":"Analytic Derivation of Vial Refraction","title":"Analytic Derivation of Vial Refraction","text":"varepsilon = arccosleft(fracx^2 + R_i^2 - R_o^22 R_i xright) - fracpi2","category":"page"},{"location":"ray_derivation/","page":"Analytic Derivation of Vial Refraction","title":"Analytic Derivation of Vial Refraction","text":"Further, beta=mathrmsign(y) cdot (fracpi2- varepsilon) and because of refraction gamma=arcsin(fracn_textvial cdot sin(beta)n_textresin).  The total ray deflection angles are given by delta_1=alpha - beta and delta_2=beta-gamma resulting in delta_textges = delta_1 + delta_2. ","category":"page"},{"location":"ray_derivation/","page":"Analytic Derivation of Vial Refraction","title":"Analytic Derivation of Vial Refraction","text":"<img src=\"../assets/drawing_angle.png\" alt=\"drawing of the angular geometry\" width=\"500\"/>","category":"page"},{"location":"ray_derivation/","page":"Analytic Derivation of Vial Refraction","title":"Analytic Derivation of Vial Refraction","text":"To calculate the height y_i which describes the virtual height while entering the outer circle, we first need the distance y = R_i cdot sin(gamma). Using the Pythagorean theorem we can derive","category":"page"},{"location":"ray_derivation/","page":"Analytic Derivation of Vial Refraction","title":"Analytic Derivation of Vial Refraction","text":"p = sqrtR_o^2-y^2","category":"page"},{"location":"ray_derivation/","page":"Analytic Derivation of Vial Refraction","title":"Analytic Derivation of Vial Refraction","text":"Then, the angle eta is given by eta =- left(arcsinleft(fracyR_0right) - textsign(y) cdot left(fracpi2-delta_textgesright)right)","category":"page"},{"location":"ray_derivation/","page":"Analytic Derivation of Vial Refraction","title":"Analytic Derivation of Vial Refraction","text":"Then, the height of the ray at exiting the outer circle, is given by  y_f = R_o cdot sin(eta)","category":"page"},{"location":"ray_derivation/","page":"Analytic Derivation of Vial Refraction","title":"Analytic Derivation of Vial Refraction","text":"Because of the isosceles triangle, the height of the virtual ray entering the outer circle is given by","category":"page"},{"location":"ray_derivation/","page":"Analytic Derivation of Vial Refraction","title":"Analytic Derivation of Vial Refraction","text":"y_i =  2cdot p cdot sin(delta_textges) + y_f","category":"page"},{"location":"function_docstrings/#Geometry","page":"Function Docstrings","title":"Geometry","text":"","category":"section"},{"location":"function_docstrings/","page":"Function Docstrings","title":"Function Docstrings","text":"PropagationScheme\nParallelRayOptics\nVialRayOptics\nWaveOptics\nPolarization","category":"page"},{"location":"function_docstrings/#SwissVAMyKnife.PropagationScheme","page":"Function Docstrings","title":"SwissVAMyKnife.PropagationScheme","text":"abstract type PropagationScheme end\n\nList of possible schemes:\n\nParallelRayOptics\nWaveOptics\n\n\n\n\n\n","category":"type"},{"location":"function_docstrings/#SwissVAMyKnife.ParallelRayOptics","page":"Function Docstrings","title":"SwissVAMyKnife.ParallelRayOptics","text":"ParallelRayOptics(angles, μ, DMD_diameter)\n\nType to represent the parallel ray optical approach. This is suited for a printer with an index matching bath. This is equivalent to an inverse (attenuated) Radon transform as the forward model of the printer. \n\nangles is a range or Vector (or CuVector) storing the illumination angles.\nDMD_diameter is the diameter of the DMD along the vial radius. So this is not the height along the rotation axis!\nμ is the absorption coefficient of the resin in units of inverse meters  So μ=100.0 1/m means that after 10mm of propagation the intensity is I(10mm) = I_0 * exp(-10.0mm * 100.0/m) = I_0 * exp(-1).\n\nSee also VialRayOptics for a printer without index matching bath.\n\nExamples\n\njulia> ParallelRayOptics(range(0, 2π, 401)[begin:end-1], 1 / 256)\nParallelRayOptics{Float64, StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}}(0.0:0.015707963267948967:6.267477343911637, 0.00390625)\n\njulia> ParallelRayOptics(range(0, 2π, 401)[begin:end-1], nothing)\nParallelRayOptics{Nothing, StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}}(0.0:0.015707963267948967:6.267477343911637, nothing)\n\n\n\n\n\n","category":"type"},{"location":"function_docstrings/#SwissVAMyKnife.VialRayOptics","page":"Function Docstrings","title":"SwissVAMyKnife.VialRayOptics","text":"Type to represent a ray optical approach where refraction and reflection intensity loss at the glass vial is considered. This is equivalent to an inverse (attenuated) Radon transform as the forward model of the printer. \n\nangles is a range or Vector (or CuVector) storing the illumination angles.\nμ is the absorption coefficient of the resin in units of inverse meters  So μ=100.0 1/m means that after 10mm of propagation the intensity is I(10mm) = I_0 * exp(-10.0mm * 100.0/m) = I_0 * exp(-1).\nR_outer is the outer radius of the glass vial.\nR_inner is the inner radius of the glass vial.\nDMD_diameter is the diameter of the DMD along the vial radius. So this is not the height along the rotation axis!\nn_vial is the refractive index of the glass vial.\nn_resin is the refractive index of the resin.\npolarization=PolarizationRandom() is the polarization of the light. See Polarization for the options. \n\nExamples\n\njulia> VialRayOptics(angles=range(0,2π, 501)[begin:end-1],\n                     μ=nothing,\n                     R_outer=6e-3,\n                     R_inner=5.5e-3,\n                     DMD_diameter=2 * R_outer,\n                     n_vial=1.47,\n                     n_resin=1.48,\n                     polarization=PolarizationRandom()\n                     )\n\n\n\n\n\n","category":"type"},{"location":"function_docstrings/#SwissVAMyKnife.WaveOptics","page":"Function Docstrings","title":"SwissVAMyKnife.WaveOptics","text":"WaveOptics(;z, λ, L, μ=nothing, angles)\n\nParameters\n\nz: the different depths we propagate the field. Should be Vector or range.\nλ: wavelength in the medium. So divide by the refractive index!\nL: The side length of the array. You should satisfy L ≈ abs(z[begin]) + abs(z[end])\nμ: Absorption coefficient.\nangles: the angles we illuminate the sample. Should be Vector or range. \n\n\n\n\n\n","category":"type"},{"location":"function_docstrings/#SwissVAMyKnife.Polarization","page":"Function Docstrings","title":"SwissVAMyKnife.Polarization","text":"Polarization\n\nPolarizationParallel() describes a parallel polarization.\nPolarizationPerpendicular() describes a perpendicular polarization. \nPolarizationRandom() describes a random polarization.\n\n\n\n\n\n","category":"type"},{"location":"function_docstrings/#Optimization","page":"Function Docstrings","title":"Optimization","text":"","category":"section"},{"location":"function_docstrings/","page":"Function Docstrings","title":"Function Docstrings","text":"optimize_patterns","category":"page"},{"location":"function_docstrings/#SwissVAMyKnife.optimize_patterns","page":"Function Docstrings","title":"SwissVAMyKnife.optimize_patterns","text":"optimize_patterns(target, ps::WaveOptics, op::GradientBased, loss::Union{LossThreshold, LossThresholdSparsity})\n\nOptimize the patterns to match the target with the wave optical model. target is the target intensity. ps is the wave optical model. op is the optimization method. loss is the loss function.\n\n\n\n\n\noptimize_patterns(target, ps::{VialRayOptics, ParallelRayOptics}, op::GradientBased, loss::LossTarget)\n\nFunction to optimize a target volume. This method returns the optimized patterns, the printed intensity and the optimization result.\n\nSee VialRayOptics how to specify the geometry of the vial. See ParallelRayOptics how to specify the geometry of the vial.\n\nSee PropagationScheme for the options for the different propagation schemes. See OptimizationScheme for the options for the different optimization schemes. See LossTarget for the options for the different loss functions.\n\nExamples\n\njulia> patterns, printed_intensity, res = optimize_patterns(target, VialRayOptics(angles=range(0,2π, 501)[begin:end-1],\n                     μ=nothing,\n                     R_outer=6e-3,\n                     R_inner=5.5e-3,\n                     n_vial=1.47,\n                     n_resin=1.48,\n                     polarization=PolarizationRandom()\n                     ), GradientBased(), LossThreshold())\n\n\n\n\n\noptimize_patterns(target::AbstractArray{T}, ps::ParallelRayOptics, op::OSMO) where T\n\nOptimize patterns with the OSMO optimization algorithm. This is only supported for ParallelRayOptics.\n\nExamples\n\njulia> optimize_patterns(target, ParallelRayOptics(range(0, 2π, 401)[begin:end-1], 1 / 256), OSMO())\n\n\n\n\n\n","category":"function"},{"location":"function_docstrings/#Optimizer","page":"Function Docstrings","title":"Optimizer","text":"","category":"section"},{"location":"function_docstrings/","page":"Function Docstrings","title":"Function Docstrings","text":"OptimizationScheme\nGradientBased\nOSMO","category":"page"},{"location":"function_docstrings/#SwissVAMyKnife.OptimizationScheme","page":"Function Docstrings","title":"SwissVAMyKnife.OptimizationScheme","text":"abstract type OptimizationScheme end\n\nList of possible schemes:\n\nGradientBased\n\nSupported for all <:PropagationScheme.\n\nOSMO\n\nSupported only for ParallelRayOptics\n\n\n\n\n\n","category":"type"},{"location":"function_docstrings/#SwissVAMyKnife.GradientBased","page":"Function Docstrings","title":"SwissVAMyKnife.GradientBased","text":"GradientBased(; optimizer=LBFSG(), options=Optim.options(iterations=30, store_trace=true))\n\nDefine parameters for the GradientBased optimization algorithm. Optim.jl is used for the optimization. L-BFGS is the default optimizer which performs well for the optimization of the reconstruction problem. 30 iterations are used as default. The trace of the optimization is stored by default.\n\nExample\n\njulia> GradientBased(; optimizer=LBFGS(), options=Optim.Options(iterations=30, store_trace=true))\n\n\n\n\n\n","category":"type"},{"location":"function_docstrings/#SwissVAMyKnife.OSMO","page":"Function Docstrings","title":"SwissVAMyKnife.OSMO","text":"OSMO(; iterations=10, thresholds=(0.7f0, 0.8f0))\n\nDefine parameters for the OSMO optimization algorithm. We recommend to use GradientBased instead of OSMO.\n\nReference\n\nRackson, Charles M., et al. Object-space optimization of tomographic reconstructions for additive manufacturing. Additive Manufacturing 48 (2021): 102367.\n\n\n\n\n\n","category":"type"},{"location":"function_docstrings/#Loss-Function","page":"Function Docstrings","title":"Loss Function","text":"","category":"section"},{"location":"function_docstrings/","page":"Function Docstrings","title":"Function Docstrings","text":"LossTarget\nLossThreshold","category":"page"},{"location":"function_docstrings/#SwissVAMyKnife.LossTarget","page":"Function Docstrings","title":"SwissVAMyKnife.LossTarget","text":"LossTarget\n\nAbstract type for loss functions. \n\nList of implemented loss functions:\n\nLossThreshold\n\n\n\n\n\n","category":"type"},{"location":"function_docstrings/#SwissVAMyKnife.LossThreshold","page":"Function Docstrings","title":"SwissVAMyKnife.LossThreshold","text":"LossThreshold(;sum_f=abs2, thresholds=(0.65f0, 0.75f0))\n\nLoss function for polymerization.  Keeps the object voxels in the range [T_U, 1] and the empty space in the range [0, T_L].\n\nmathcalL = underbracesum_v intextobject textReLu(T_U - I_v)^K_textforce object polymerization + \n+underbracesum_vnotintextobject textReLu(I_v - T_L) ^K_textkeep empty space unpolymerized +\n\n+underbracesum_v intextobject textReLu(I_v - 1)^K_textavoid overpolymerization\n\nThe default K=2 corresponds to sum_f=abs2.\n(T_L, T_U) = thresholds\n\njulia> l = LossThreshold(thresholds=(0.5, 0.7))\nLossThreshold{typeof(abs2), Float64}(abs2, (0.5, 0.7))\n\njulia> x = [1.0, 0.0, 0.55]\n3-element Vector{Float64}:\n 1.0\n 0.0\n 0.55\n\njulia> target = [1, 0, 1]\n3-element Vector{Int64}:\n 1\n 0\n 1\n\njulia> l(x, target, nothing)\n0.022499999999999975\n\njulia> (0.7 - 0.55)^2\n0.022499999999999975\n\n\n\n\n\n\n","category":"type"},{"location":"background/#Background-on-Volumetric-Additive-Manufacturing","page":"Background","title":"Background on Volumetric Additive Manufacturing","text":"","category":"section"},{"location":"background/","page":"Background","title":"Background","text":"Volumetric Additive Manufacturing in general is about 3D printing arbitrary shapes and objects. This toolbox solves some of the optimization challenges around TVAM.","category":"page"},{"location":"background/#Tomographic-Volumetric-Additive-Manufacturing-(TVAM)","page":"Background","title":"Tomographic Volumetric Additive Manufacturing (TVAM)","text":"","category":"section"},{"location":"background/","page":"Background","title":"Background","text":"Tomographic Volumetric Additive Manufacturing is an emerging 3D printing technology that uses light to create 3D objects. The technology is based on the principles of tomography, which is a technique for creating 3D images of the internal structures of objects by taking a large number of 2D X-ray images from different angles and then using these images to reconstruct a 3D image of the object. In TVAM the object is not imaged, but rather the 3D object is created directly from the 2D images which are projected into a volume of photosensitive material. Of course, this would smear light into all regions of the resin and create a solid block of material. However, polymerization is inhibited by oxygen or other inhibitors. Hence, the region only polymerizes where the light intensity is high enough to overcome the polymerization threshold.","category":"page"},{"location":"background/","page":"Background","title":"Background","text":"The process is illustrated in the figure below.","category":"page"},{"location":"background/","page":"Background","title":"Background","text":"<img src=\"../assets/principle.png\" alt=\"\" width=\"600\"/>","category":"page"},{"location":"background/#Central-Question","page":"Background","title":"Central Question","text":"","category":"section"},{"location":"background/","page":"Background","title":"Background","text":"The central question in TVAM what are the light pattern such that the desired 3D object is created in the resin? In other words, void regions should receive less light than polymerized threshold.  And object regions should receive more light than the polymerized threshold.","category":"page"},{"location":"background/","page":"Background","title":"Background","text":"Mathematically, we seek to find a set of light patterns P such that the light intensity I in the resin satisfies the following constraints:","category":"page"},{"location":"background/","page":"Background","title":"Background","text":"mathcalL = underbracesum_v intextobject textReLu(T_U - I_v)^K_textforce object polymerization+underbracesum_vnotintextobject textReLu(I_v - T_L) ^K_textkeep empty space unpolymerized+","category":"page"},{"location":"background/","page":"Background","title":"Background","text":"+underbracesum_v intextobject textReLu(I_v - 1)^K_textavoid overpolymerization +underbracesum_p intextpatterns P_p^4_textavoid sparse patterns","category":"page"},{"location":"background/","page":"Background","title":"Background","text":"The propagation operator mathcalP can be either a simple ray tracing or a more complex wave propagation model.","category":"page"},{"location":"background/","page":"Background","title":"Background","text":"I_v = mathcalP P","category":"page"},{"location":"background/#Optimization","page":"Background","title":"Optimization","text":"","category":"section"},{"location":"background/","page":"Background","title":"Background","text":"The optimization problem is to find the light patterns P that minimize the loss mathcalL. This is a challenging optimization problem, because the light patterns P are high-dimensional and the loss mathcalL is non-convex and non-smooth. The optimization problem is typically solved using (projected) gradient-based optimization algorithms. In each step we set the patterns to non-negative values since light intensity cannot be negative. We use the L-BFGS algorithm, which is a quasi-Newton method that is well-suited for non-smooth optimization problems. Usually CUDA is used to accelerate the forward and backward propagation of the light patterns.","category":"page"},{"location":"#SwissVAMyKnife.jl","page":"SwissVAMyKnife.jl","title":"SwissVAMyKnife.jl","text":"","category":"section"},{"location":"","page":"SwissVAMyKnife.jl","title":"SwissVAMyKnife.jl","text":"(Image: codecov) (Image: Documentation for stable version) (Image: Documentation for development version)","category":"page"},{"location":"","page":"SwissVAMyKnife.jl","title":"SwissVAMyKnife.jl","text":"Tomographic Volumetric Additive Manufacturing is a novel 3D printing technique which is based on a tomographic principle. Light is illuminated from different angles onto a glass vial which contains a photosensitive resin. Once a voxel in the resin receives enough light, polymeristartss.","category":"page"},{"location":"","page":"SwissVAMyKnife.jl","title":"SwissVAMyKnife.jl","text":"This toolbox is developed to solve the optimization challenge around VAM. What are the required patterns on the projector such object voxels polymerize and not-object voxels stay unpolymerized.","category":"page"},{"location":"","page":"SwissVAMyKnife.jl","title":"SwissVAMyKnife.jl","text":"This package is developed in Julia Lang and features CUDA and CPU support. CUDA can accelerate reconstruction typically 10-20x times.","category":"page"},{"location":"","page":"SwissVAMyKnife.jl","title":"SwissVAMyKnife.jl","text":"It runs on Windows, Linux and macOS!","category":"page"},{"location":"","page":"SwissVAMyKnife.jl","title":"SwissVAMyKnife.jl","text":"<img src=\"assets/principle.png\" alt=\"\" width=\"500\"/>","category":"page"},{"location":"","page":"SwissVAMyKnife.jl","title":"SwissVAMyKnife.jl","text":"The general principle behind TVAM. a) a set of 2D projection patterns is propagated into space. b) shows how a slice of the pattern propagates through the volume and c) how the incoherent sum results in a total energy dose. d) the object polymerizes if it reaches an energy threshold. e) polymerization threshold results in a printed slice. f) is the intensity histogram of b). g) is the 3D view of the Benchy boat. h) is the general setup.","category":"page"},{"location":"#Features","page":"SwissVAMyKnife.jl","title":"Features","text":"","category":"section"},{"location":"","page":"SwissVAMyKnife.jl","title":"SwissVAMyKnife.jl","text":"3D parallel Radon transform for ray optical simulation \nrigorous simulation of absorption and vial refraction\nCUDA acceleration hence high performance\n3D coherent wave optical optimization (see this paper for details and limitations). Note this has been only theoretical demonstrated and experimentally not validated.","category":"page"},{"location":"","page":"SwissVAMyKnife.jl","title":"SwissVAMyKnife.jl","text":"(Image: )","category":"page"},{"location":"#Installation","page":"SwissVAMyKnife.jl","title":"Installation","text":"","category":"section"},{"location":"","page":"SwissVAMyKnife.jl","title":"SwissVAMyKnife.jl","text":"Install the most recent Julia version. Then open the REPL and run:","category":"page"},{"location":"","page":"SwissVAMyKnife.jl","title":"SwissVAMyKnife.jl","text":"julia> using Pkg\n\njulia> Pkg.add(\"SwissVAMyKnife\")","category":"page"},{"location":"#Examples","page":"SwissVAMyKnife.jl","title":"Examples","text":"","category":"section"},{"location":"","page":"SwissVAMyKnife.jl","title":"SwissVAMyKnife.jl","text":"You can also run the examples locally. Download this repository and then do the following in your REPL:","category":"page"},{"location":"","page":"SwissVAMyKnife.jl","title":"SwissVAMyKnife.jl","text":"julia> cd(\"examples/\")\n\njulia> using Pkg; Pkg.activate(\".\"); Pkg.instantiate()\n  Activating project at `~/.julia/dev/RadonKA.jl/examples`\n\njulia> using Pluto; Pluto.run()","category":"page"},{"location":"#Other-packages","page":"SwissVAMyKnife.jl","title":"Other packages","text":"","category":"section"},{"location":"","page":"SwissVAMyKnife.jl","title":"SwissVAMyKnife.jl","text":"There is the Python based VAM Toolbox and LDCD-VAM.  In terms of functionality, they also offer a ray based methods including absorption. Wave optical methods are not offered.","category":"page"},{"location":"#Development","page":"SwissVAMyKnife.jl","title":"Development","text":"","category":"section"},{"location":"","page":"SwissVAMyKnife.jl","title":"SwissVAMyKnife.jl","text":"File an issue on GitHub if you encounter any problems.","category":"page"},{"location":"#Citation","page":"SwissVAMyKnife.jl","title":"Citation","text":"","category":"section"},{"location":"","page":"SwissVAMyKnife.jl","title":"SwissVAMyKnife.jl","text":"If you use this software in academic work, please consider citing:","category":"page"},{"location":"","page":"SwissVAMyKnife.jl","title":"SwissVAMyKnife.jl","text":"@misc{wechsler2024wave,\n      title={Wave optical model for tomographic volumetric additive manufacturing},\n      author={Felix Wechsler and Carlo Gigli and Jorge Madrid-Wolff and Christophe Moser},\n      year={2024},\n      eprint={2402.06283},\n      archivePrefix={arXiv},\n      primaryClass={physics.optics}\n}","category":"page"},{"location":"real_world_application/#Real-World-Application","page":"Real World Application","title":"Real World Application","text":"","category":"section"},{"location":"real_world_application/","page":"Real World Application","title":"Real World Application","text":"In this section we explain in detail how we use this toolbox to print a real boat. ","category":"page"},{"location":"real_world_application/#Resin-Preparation","page":"Real World Application","title":"Resin Preparation","text":"","category":"section"},{"location":"real_world_application/","page":"Real World Application","title":"Real World Application","text":"For the resin we use the commercial Sartomer Arkema resin which mainly consists of Dipentaerythritol pentaaycrlate. As photo initiator we use TPO. With a refractometer we measure the refractive index n_textresin = 14849. We pour the resin into a cup. The photoinitiator is mixed into Ethanol. This is shaken until the TPO is dissolved. The ethanol with the TPO is mixed into the resin. It is mixed in a Kurabo Planetary Mixer for some minutes.  In total, we mix roughly 15mathrmmg of TPO into 40mathrmL of the resin. With a spectrometer, we determine the absorbance at our printing wavelength 405mathrmnm to be A=023471mathrmcm. That means, mu = 5404mathrmm^-1 Technically there is also absorption of the resin itself which does not contribute to the absorption but we determined it to be A=192mathrmm^-1. So we neglect this effect and assume all absorbed light is contributing to the polymerization.","category":"page"},{"location":"real_world_application/","page":"Real World Application","title":"Real World Application","text":"<img src=\"../assets/container.jpeg\" alt=\"\" width=\"300\"/>","category":"page"},{"location":"real_world_application/#Glass-Vial","page":"Real World Application","title":"Glass Vial","text":"","category":"section"},{"location":"real_world_application/","page":"Real World Application","title":"Real World Application","text":"As glass vials we use simple cylindrical glass vial which are not quite optimized for optical applications. With a measurement calliper we determine the router radius to be R_textouter = (83pm001)mathrmmm and the inner radius R_textouter = (76pm 001)mathrmmm. The refractive index is roughly n_textvial=158.","category":"page"},{"location":"real_world_application/","page":"Real World Application","title":"Real World Application","text":"<img src=\"../assets/vial.jpeg\" alt=\"\" width=\"300\"/>","category":"page"},{"location":"real_world_application/#DMD-Characterization","page":"Real World Application","title":"DMD Characterization","text":"","category":"section"},{"location":"real_world_application/","page":"Real World Application","title":"Real World Application","text":"We have a camera system which images the printing path at the center of the vial. After our 4f printing optical system, the DMD pixel pitch is 1379mathrmmu m.","category":"page"},{"location":"real_world_application/#Centering-of-DMD-with-respect-to-glass-vial","page":"Real World Application","title":"Centering of DMD with respect to glass vial","text":"","category":"section"},{"location":"real_world_application/","page":"Real World Application","title":"Real World Application","text":"Our rotation stage Zaber X-RSW60C with the drill bit holder wobbles significantly with the rotation angle.  We just fit a function f(x) = A cdot sin(x - o) + c to our measurements of the wobble.  Fortunately, we just have to shift the DMD pattern for each angle by the respective amount to correct for this left-right-movement of the vial.","category":"page"},{"location":"real_world_application/","page":"Real World Application","title":"Real World Application","text":"<img src=\"../assets/wobble.png\" alt=\"\" width=\"500\"/>","category":"page"},{"location":"real_world_application/#Selecting-a-Target","page":"Real World Application","title":"Selecting a Target","text":"","category":"section"},{"location":"real_world_application/","page":"Real World Application","title":"Real World Application","text":"In this case we optimize the 3D Benchy as a printing target.","category":"page"},{"location":"real_world_application/#Specifying-System-Parameters","page":"Real World Application","title":"Specifying System Parameters","text":"","category":"section"},{"location":"real_world_application/","page":"Real World Application","title":"Real World Application","text":"For the optimization we need to include all system parameters such as the refractive index of vial and resin and the geometry of the vial. Further, we need to input the size of the DMD with respect to the glass vial. If the DMD is smaller than the glass vial, we output patterns which are smaller than the simulated volume. If the DMD is larger than the glass vial, we simulate patterns which are exactly the size of the glass vial. It is your responsibility to pad the resulting images with zeros such that they fit to your setup. Without index matching bath this is important since a wrong scaling will ultimate result in a low quality print.","category":"page"},{"location":"real_world_application/#Specifying-Optimization-Parameters","page":"Real World Application","title":"Specifying Optimization Parameters","text":"","category":"section"},{"location":"real_world_application/#Analysing-Results","page":"Real World Application","title":"Analysing Results","text":"","category":"section"}]
}