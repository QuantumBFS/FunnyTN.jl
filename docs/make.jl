using Documenter, FunnyTN

# preprocess tutorial scripts
using Literate, Pkg
tutorialpath = joinpath(@__DIR__, "src/tutorial")
for jlfile in ["tutorial.jl"]
    Literate.markdown(joinpath(tutorialpath, jlfile), tutorialpath)
end

# make documents
makedocs(
    modules = [FunnyTN, FunnyTN.Tensors, FunnyTN.TensorNetworks],
    clean = false,
    format = :html,
    sitename = "FunnyTN.jl",
    linkcheck = !("skiplinks" in ARGS),
    analytics = "UA-89508993-1",
    pages = [
        "Home" => "index.md",
        "Tutorial" => "tutorial/tutorial.md",
        "Manual" => Any[
            "man/funnytn.md",
            "man/tensors.md",
            "man/tensornetworks.md",
        ],
    ],
    html_prettyurls = !("local" in ARGS),
    html_canonical = "https://quantumbfs.github.io/FunnyTN.jl/latest/",
)

deploydocs(
    repo = "github.com/QuantumBFS/FunnyTN.jl.git",
    target = "build",
    julia = "1.0",
    deps = nothing,
    make = nothing,
)
