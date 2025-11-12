"""
    LinesearchMethod

Examples include [`StaticState`](@ref), [`Backtracking`](@ref), [`Bisection`](@ref) and [`Quadratic`](@ref).
See these examples for specific information on linesearch algorithms.

# Extended help

A `LinesearchMethod` always goes together with a [`LinesearchState`](@ref) and each of those [`LinesearchState`](@ref)s has a functor implemented:

```julia
ls(obj, α)
```
where obj is a [`AbstractUnivariateProblem`](@ref) and `α` is an initial *step size*. The output of this functor is then a final step size that is used for updating the parameters.
"""
abstract type LinesearchMethod <: NonlinearMethod end

@doc raw"""
    Backtracking <: LinesearchMethod

The *backtracking* method.

# Constructors

```julia
Backtracking()
```

# Extended help

The backtracking algorithm starts by setting ``y_0 \gets f(0)`` and ``d_0 \gets \nabla_0f``.

The algorithm is executed by calling the functor of [`BacktrackingState`](@ref).

The following is then repeated until the stopping criterion is satisfied or `config.max_iterations` """ * """($(MAX_ITERATIONS) by default) is reached:

```julia
if value!(obj, α) ≥ y₀ + ls.ϵ * α * d₀
    α *= ls.p
else
    break
end
```
The stopping criterion as an equation can be written as:

```math
f(\alpha) < y_0 + \epsilon \alpha \nabla_0f = y_0 + \epsilon (\alpha - 0)\nabla_0f.
```
Note that if the stopping criterion is not reached, ``\alpha`` is multiplied with ``p`` and the process continues.

[Sometimes](https://en.wikipedia.org/wiki/Backtracking_line_search) the parameters ``p`` and ``\epsilon`` have different names such as ``\tau`` and ``c``.
"""
struct Backtracking <: LinesearchMethod end

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
struct Bisection <: LinesearchMethod end

"""
    Quadratic <: LinesearchMethod

The *quadratic* method. Compare this to [`BierlaireQuadratic`](@ref). The algorithm is taken from [kelley1995iterative](@cite).

# Constructors

```julia
Quadratic()
```

# Extended help
"""
struct Quadratic <: LinesearchMethod end

"""
    Quadratic2 <: LinesearchMethod

The second *quadratic* method. Compare this to [`Quadratic`](@ref).

# Extended help
"""
struct Quadratic2 <: LinesearchMethod end

"""
    BierlaireQuadratic <: LinesearchMethod

Algorithm taken from [bierlaire2015optimization](@cite).
"""
struct BierlaireQuadratic <: LinesearchMethod end

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
Base.show(io::IO, ::Backtracking) = print(io, "Backtracking")
Base.show(io::IO, ::Bisection) = print(io, "Bisection")
Base.show(io::IO, ::Quadratic) = print(io, "Quadratic Polynomial")
Base.show(io::IO, ::BierlaireQuadratic) = print(io, "Quadratic Polynomial (Bierlaire version).")