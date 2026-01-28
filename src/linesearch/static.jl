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
struct Static{T<:Number} <: LinesearchMethod{T}
    α::T

    Static(α::T = 1.0) where {T} = new{T}(α)
end

Base.show(io::IO, alg::Static) = print(io, "Static with α = " * string(alg.α) * ".")

function solve(::LinesearchProblem{T}, ls::Linesearch{T, LST}) where {T, LST<:Static{T}}
    ls.algorithm.α
end

function Base.convert(::Type{T}, algorithm::Static) where {T}
    T ≠ eltype(algorithm) || return algorithm
    Static(T(algorithm.α))
end