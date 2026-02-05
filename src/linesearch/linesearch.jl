const DEFAULT_LINESEARCH_NMAX = 100
const DEFAULT_LINESEARCH_RMAX = 100

"""
    Linesearch

A `struct` that stores the [`LinesearchMethod`](@ref) and [`Options`](@ref).

# Keys

- `algorithm::`[`LinesearchMethod`](@ref)
- `config::`[`Options`](@ref)

# Constructors

The following constructors can be used:

```julia
Linesearch(alg, config)
Linesearch(; algorithm, config, kwargs...)
```
"""
struct Linesearch{T,ALG<:LinesearchMethod{T},OPT<:Options{T}}
    algorithm::ALG
    config::OPT
end

"""
    solve(ls_prob, ls)
    solve(ls_prob, ls_method)

Minimize the [`LinesearchProblem`](@ref) with the [`LinesearchMethod`](@ref) `ls_method`.
"""
function solve(::LinesearchProblem{T}, ::Linesearch{T,ALG}) where {T,ALG<:LinesearchMethod{T}}
    error("Solve method missing for $(ALG).")
end

function Linesearch(T::DataType; algorithm::LinesearchMethod=Static(), options_kwargs...)
    config = Options(T; options_kwargs...)
    Linesearch{T,typeof(algorithm),typeof(config)}(algorithm, config)
end

Linesearch(; T::DataType=Float64, kwargs...) = Linesearch(T; kwargs...)

function Linesearch(algorithm::LinesearchMethod; T::DataType=Float64, options_kwargs...)
    config = Options(T; options_kwargs...)
    _algorithm = convert(T, algorithm)
    Linesearch{T,typeof(_algorithm),typeof(config)}(_algorithm, config)
end
