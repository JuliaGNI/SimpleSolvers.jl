"""
    LinesearchMethod

Examples include [`StaticState`](@ref), [`Backtracking`](@ref), [`Bisection`](@ref) and [`Quadratic`](@ref).
See these examples for specific information on linesearch algorithms.

# Extended help

A `LinesearchMethod` always goes together with a [`LinesearchState`](@ref) and each of those [`LinesearchState`](@ref)s has a functor implemented:

```julia
ls(obj, α)
```
where obj is a [`LinesearchProblem`](@ref) and `α` is an initial *step size*. The output of this functor is then a final step size that is used for updating the parameters.
"""
abstract type LinesearchMethod <: NonlinearMethod end

"""
    solve(ls_prob, ls)
    solve(ls_prob, ls_method)

Minimize the [`LinesearchProblem`](@ref) with the [`LinesearchMethod`](@ref) `ls_method`.
"""
function solve(::LinesearchProblem, ::LT) where {LT <: Linesearch}
    error("Solve routine for ")
end

function solve(ls_prob::LinesearchProblem, ls_method::LinesearchMethod; config::Options=Options())
    solve(ls_prob, Linesearch(ls_method, config))
end

"""
    Bisection <: LinesearchMethod

The *bisection* method.

# Constructors

```julia
Bisection()
```

# Extended help

The bisection algorithm starts with an interval and successively bisects it into smaller intervals until a root is found.
See [`bisection`](@ref).
"""
struct Bisection{T} <: LinesearchMethod 

end

"""
    Quadratic <: LinesearchMethod


The *quadratic* method. Compare this to [`BierlaireQuadratic`](@ref). The algorithm is adjusted from [kelley1995iterative](@cite).

# Constructor
```julia
Quadratic()
```
# Extended help
"""
struct Quadratic{T} <: LinesearchMethod 

end

"""
    BierlaireQuadratic <: LinesearchMethod

Algorithm taken from [bierlaire2015optimization](@cite).
"""
struct BierlaireQuadratic{T} <: LinesearchMethod 

end

"""
    Static <: LinesearchMethod

The *static* method.

# Constructors

```julia
Static(α)
```

# Keys

Keys include:
-`α`: equivalent to a step size. The default is `1`.

# Extended help
"""
struct Static{T<:Number} <: LinesearchMethod
    α::T

    Static(α::T = 1.0) where {T} = new{T}(α)
end

Base.show(io::IO, alg::Static) = print(io, "Static with α = " * string(alg.α) * ".")
Base.show(io::IO, ::Bisection) = print(io, "Bisection")
Base.show(io::IO, ::Quadratic) = print(io, "Quadratic Polynomial")
Base.show(io::IO, ::BierlaireQuadratic) = print(io, "Quadratic Polynomial (Bierlaire version).")
