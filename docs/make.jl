using Documenter, FunnyTN

# TODO: use Literate to process examples
# using Literate
# preprocess tutorial scripts

# make documents
makedocs(
    modules = [FunnyTN],
    clean = false,
    format = :html,
    sitename = "FunnyTN.jl",
    linkcheck = !("skiplinks" in ARGS),
    analytics = "UA-89508993-1",
    pages = [
        "Home" => "index.md",
        "Tutorial" => "tutorial.md",
        "Manual" => "funnytn.md",
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
