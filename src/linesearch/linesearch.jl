const DEFAULT_LINESEARCH_NMAX=100
const DEFAULT_LINESEARCH_RMAX=100

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
struct Linesearch{T, ALG <: LinesearchMethod, OPT <: Options{T}}
    algorithm::ALG
    config::OPT
end

"""
    solve(ls_prob, ls)
    solve(ls_prob, ls_method)

Minimize the [`LinesearchProblem`](@ref) with the [`LinesearchMethod`](@ref) `ls_method`.
"""
function solve(::LinesearchProblem, ::LT) where {LT <: Linesearch}
    error("Solve routine for ")
end

function Linesearch(T::DataType=Float64; algorithm = Static(), options_kwargs...)
    config = Options(T; options_kwargs...)
    Linesearch{T, typeof(algorithm), typeof(config)}(algorithm, config)
end