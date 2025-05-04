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
        size_threshold = 1048576,
    ),
    pages=[
        "Home" => "index.md",
        "Objectives" => "objectives.md",
        "Gradients" => "gradients.md",
        "Jacobians" => "jacobians.md",
        "Hessians" => "hessians.md",
        "Line Search" => ["Line Searches" => "linesearch/linesearch.md",
                         "Static" => "linesearch/static.md",
                         "The Sufficient Decrease Condition" => "linesearch/sufficient_decrease_condition.md",
                         "The Curvature Condition" => "linesearch/curvature_condition.md",
                         "Backtracking" => "linesearch/backtracking.md",
                         "Bisections" => "linesearch/bisections.md",
                         "Quadratic" => "linesearch/quadratic.md",
                         ],
        "Optimizers" => ["optimizers/optimizers.md"],
        "Updates" => "update.md",
        "Initialization" => "initialize.md",
        "References" => "references.md",
    ],
)

deploydocs(;
    repo   = "github.com/JuliaGNI/SimpleSolvers.jl",
    devurl = "latest",
    devbranch = "main",
)
