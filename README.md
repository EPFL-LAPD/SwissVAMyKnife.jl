# SwissVAMyKnife.jl
[![CI](https://github.com/EPFL-LAPD/SwissVAMyKnife.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/EPFL-LAPD/SwissVAMyKnife.jl/actions/workflows/CI.yml) [![codecov](https://codecov.io/gh/EPFL-LAPD/SwissVAMyKnife.jl/graph/badge.svg?token=JZYHT3P3B7)](https://codecov.io/gh/EPFL-LAPD/SwissVAMyKnife.jl) [![Documentation for stable version](https://img.shields.io/badge/docs-stable-blue.svg)](https://epfl-lapd.github.io/SwissVAMyKnife.jl/stable) [![Documentation for development version](https://img.shields.io/badge/docs-main-blue.svg)](https://epfl-lapd.github.io/SwissVAMyKnife.jl/dev)

[Tomographic Volumetric Additive Manufacturing](https://www.youtube.com/watch?v=ONBHkzimRbg) is a novel 3D printing technique
which is based on a tomographic principle.
Light is illuminated from different angles onto a glass vial which contains a photosensitive resin.
Once a voxel in the resin receives enough light, polymerization starts.

This toolbox is developed to solve the optimization challenge around TVAM.
What are the required patterns on the projector such object voxels polymerize and not-object voxels stay unpolymerized?

This package is written in [Julia Lang](https://julialang.org/) and features CUDA and multithreaded CPU support. CUDA can accelerate reconstruction typically 10-20x times.

It runs on Windows, Linux and macOS!

# Background

<img src="docs/src/assets/principle.png" alt="" width="900"/>
The general principle behind TVAM. a) a set of 2D projection patterns is
propagated into space. b) shows how a slice of the pattern propagates through the
volume and c) how the incoherent sum results in a total energy dose. d) the object
polymerizes if it reaches an energy threshold. e) polymerization threshold results in a
printed slice. f) is the intensity histogram of b). g) is the 3D view of the Benchy boat.
h) is the general setup.


# Features
* 3D parallel Radon transform for ray optical simulation 
* rigorous simulation of absorption and air -> vial -> resin refraction
* CUDA acceleration -> high performance
* 3D coherent wave optical optimization (see this [publication](https://opg.optica.org/oe/fulltext.cfm?uri=oe-32-8-14705&id=548744) for details and limitations). Note this has been only theoretical demonstrated and experimentally not validated.


# Installation
The installation and running of the examples is very simple. 
Install the most recent [Julia version](https://julialang.org/downloads/). Then open the REPL and run:
```julia
julia> using Pkg

julia> Pkg.add("SwissVAMyKnife")
```

# Examples
You can run the [examples](https://github.com/EPFL-LAPD/SwissVAMyKnife.jl/tree/main/examples) locally.
Download this repository and then do the following in your REPL:
```julia
julia> cd("examples/")

julia> using Pkg; Pkg.activate("."); Pkg.instantiate()
  Activating project at `~/.julia/dev/SwissVAMyKnife.jl/examples`

julia> using Pluto; Pluto.run()
```
Dependencies (including CUDA) are automatically installed!

## Overview of the examples
Here a short overview of the example:
* A simple [2D example](https://github.com/EPFL-LAPD/SwissVAMyKnife.jl/blob/main/examples/1_glass_vial_pattern_optimization.jl) to showcase abilites to model glass vial refraction and without refraction
* Here a [large 3D example](https://github.com/EPFL-LAPD/SwissVAMyKnife.jl/blob/main/examples/2_benchy_boat_optimization_ray_optics.jl) based on ray optics. It is compared with the wave optical propagator.
* A [large 3D example](https://github.com/EPFL-LAPD/SwissVAMyKnife.jl/blob/main/examples/3_wave_optics_optimization.jl) based on the wave optical optimization
* A [real example](https://github.com/EPFL-LAPD/SwissVAMyKnife.jl/blob/main/examples/4_realistic_pattern_optimization.jl) of our setup where the DMD is smaller than the glass vial.


# Other packages
There is the Python based [VAM Toolbox](https://github.com/computed-axial-lithography/VAMToolbox) and [LDCD-VAM](https://github.com/facebookresearch/LDCT-VAM/). 
In terms of functionality, they also offer a ray based methods including absorption.
Wave optical methods are not offered.

# Development
File an [issue](https://github.com/roflmaostc/RadonKA.jl/issues) on [GitHub](https://github.com/roflmaostc/RadonKA.jl) if you encounter any problems.
You can also join [my conference room](https://epfl.zoom.us/my/wechsler). Give me a minute to join!
If you need any help regarding TVAM in general, don't hesistate to contact us!

# Citation
If you use this software in academic work, please consider citing this [publication](https://opg.optica.org/oe/fulltext.cfm?uri=oe-32-8-14705&id=548744):
```bibtex
@article{Wechsler:24,
author = {Felix Wechsler and Carlo Gigli and Jorge Madrid-Wolff and Christophe Moser},
journal = {Opt. Express},
keywords = {3D printing; Computed tomography; Liquid crystal displays; Material properties; Ray tracing; Refractive index},
number = {8},
pages = {14705--14712},
publisher = {Optica Publishing Group},
title = {Wave optical model for tomographic volumetric additive manufacturing},
volume = {32},
month = {Apr},
year = {2024},
url = {https://opg.optica.org/oe/abstract.cfm?URI=oe-32-8-14705},
doi = {10.1364/OE.521322},
}
```
