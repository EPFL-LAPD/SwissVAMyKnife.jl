using SwissVAMyKnife, Documenter 

DocMeta.setdocmeta!(SwissVAMyKnife, :DocTestSetup, :(using SwissVAMyKnife); recursive=true)
makedocs(modules = [SwissVAMyKnife], 
         sitename = "SwissVAMyKnife.jl", 
         pages = Any[
            "SwissVAMyKnife.jl" => "index.md",
            "Background" => "background.md",
            "Ray Optical TVAM" => [
                "Simple Ray Optical TVAM" =>  "ray_TVAM.md",
                "Analytic Derivation of Vial Refraction" =>  "ray_derivation.md",
            ],
            #"Simple Wave Optical TVAM " =>  "wave_TVAM.md",
            "Real World Application" => "real_world_application.md",
            "Function Docstrings" => "function_docstrings.md",
         ],
         warnonly=true,
        )

deploydocs(repo = "github.com/EPFL-LAPD/SwissVAMyKnife.jl.git", devbranch="main")
