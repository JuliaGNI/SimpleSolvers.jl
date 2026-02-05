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
end

Static(::Type{T}=Float64; α=one(T)) where {T} = Static{T}(α)
Static(::Type{T}, ::SolverMethod) where {T} = Static(T)

function solve(::LinesearchProblem{T}, ls::Linesearch{T,LST}) where {T,LST<:Static{T}}
    ls.algorithm.α
end

Base.show(io::IO, alg::Static) = print(io, "Static with α = " * string(alg.α) * ".")

function Base.convert(::Type{T}, algorithm::Static) where {T}
    T ≠ eltype(algorithm) || return algorithm
    Static{T}(T(algorithm.α))
end

Base.isapprox(st₁::Static{T}, st₂::Static{T}; kwargs...) where {T} = isapprox(st₁.α, st₂.α; kwargs...)
