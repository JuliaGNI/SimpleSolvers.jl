using SimpleSolvers
using Documenter

makedocs(;
    modules=[SimpleSolvers],
    authors="Michael Kraus",
    repo="https://github.com/JuliaGNI/SimpleSolvers.jl/blob/{commit}{path}#L{line}",
    sitename="SimpleSolvers.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JuliaGNI.github.io/SimpleSolvers.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaGNI/SimpleSolvers.jl",
)
