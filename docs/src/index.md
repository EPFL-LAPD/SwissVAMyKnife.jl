# SwissVAMyKnife.jl

[Tomographic Volumetric Additive Manufacturing](https://www.nature.com/articles/s41467-020-14630-4) is a novel 3D printing technique
which is based on a tomographic principle.
Light is illuminated from different angles onto a glass vial which contains a photosensitive resin.
Once a voxel in the resin receives enough light, polymeristartss.

This toolbox is developed to solve the optimization challenge around VAM.
What are the required patterns on the projector such object voxels polymerize and not-object voxels
stay unpolymerized.

This package is developed in [Julia Lang](https://julialang.org/) and features CUDA and CPU support. CUDA can accelerate reconstruction typically 10-20x times.
It runs on Windows, Linux and MacOS

# Installation
We recommend the most recent Julia version:
```julia
julia> ]add github.com/EPFL-LAPD/SwissVAMyKnife.jl
```

# Features

* [x] several optimizers (such as L-BFGS provided by [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl))
* [x] based on automatic differentiation
* [x] CUDA acceleration
* [x] 3D parallel inverse (exponential) Radon for ray optical simulation 
* [x] 3D coherent wave optical optimization (see this paper for details and limitations). Note this has been only theoretical demonstrated and experimentally not validated.
* [x] absorption can be included
* [x] refraction of glass vial is included


# Examples


# Other packages
There is the Python based [VAM Toolbox](https://github.com/computed-axial-lithography/VAMToolbox) and [LDCD-VAM](https://github.com/facebookresearch/LDCT-VAM/). In terms of functionality, they also offer a ray based method including absorption.
Wave optical methods are not offered.

# Development
File an issue on [GitHub](https://github.com/EPFL-LAPD/SwissVAMyKnife.jl) if you encounter any problems.

