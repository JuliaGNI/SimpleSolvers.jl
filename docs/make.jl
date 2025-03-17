using SimpleSolvers
using Documenter
using DocumenterCitations
import Bibliography

bib = CitationBibliography(joinpath(@__DIR__, "src", "SimpleSolvers.bib"))
Bibliography.sort_bibliography!(bib.entries, :nyt)  # name-year-title

const buildpath = haskey(ENV, "CI") ? ".." : ""

makedocs(;
    plugins = [bib],
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
        "Linesearch" => ["linesearch/linesearch.md",
                         "Backtracking" => "linesearch/backtracking.md"],
        "References" => "references.md",
    ],
)

deploydocs(;
    repo   = "github.com/JuliaGNI/SimpleSolvers.jl",
    devurl = "latest",
    devbranch = "main",
)
