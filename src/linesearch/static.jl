"""
    Static <: LinesearchMethod

The *static* method.

# Keys

Keys include:
- `α`: equivalent to a step size. The default is `1`.

# Examples

```jldoctest; setup = :(using SimpleSolvers)
Static()

# output

Static with α = 1.0.
```
"""
struct Static{T<:Number} <: LinesearchMethod{T}
    α::T
end

Static(::Type{T}=Float64; α=one(T)) where {T} = Static{T}(α)
Static(::Type{T}, ::SolverMethod) where {T} = Static(T)

# `Static` ignores the caller's trial step and cannot fail, so it has nothing to report; the
# outcome is `LINESEARCH_UNKNOWN` rather than `LINESEARCH_DECREASED` because no merit is ever
# evaluated and hence no decrease has been established. `linesearch_warnings` passes
# `LINESEARCH_UNKNOWN` over in silence, so the derived `solve` returns `method(ls).α` and says
# nothing — which is what the hand-written `solve` this replaced did.
solve_with_status(ls::Linesearch{T,<:Static}, α::T, params=NullParameters()) where {T} =
    LinesearchStatus(method(ls).α, LINESEARCH_UNKNOWN)

Base.show(io::IO, alg::Static) = print(io, "Static with α = " * string(alg.α) * ".")

function change_precision(::Type{T}, method::Static) where {T}
    T ≠ eltype(method) || return method
    Static{T}(T(method.α))
end

Base.isapprox(st₁::Static{T}, st₂::Static{T}; kwargs...) where {T} = isapprox(st₁.α, st₂.α; kwargs...)
