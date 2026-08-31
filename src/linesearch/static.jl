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

!!! info "The caller's ceiling still binds"
    `Static` has no `αmax` field — the whole point of the method is that `α` is the caller's to
    fix — but a `params.αmax` clamps the step it hands back, since a caller that says no step
    above a given length is admissible means this one too. See
    [`SimpleSolvers.linesearch_αmax`](@ref).
"""
struct Static{T <: Number} <: LinesearchMethod{T}
    α::T
end

Static(::Type{T} = Float64; α = one(T)) where {T} = Static{T}(α)
Static(::Type{T}, ::SolverMethod) where {T} = Static(T)

# `Static` ignores the caller's trial step and cannot fail, so it has nothing to report; the
# outcome is `LINESEARCH_UNKNOWN` rather than `LINESEARCH_DECREASED` because no merit is ever
# evaluated and hence no decrease has been established. `linesearch_warnings` passes
# `LINESEARCH_UNKNOWN` over in silence, so the derived `solve` returns `method(ls).α` and says
# nothing — which is what the hand-written `solve` this replaced did.
#
# The ceiling still applies: `Static` establishes nothing about its step, which is all the more
# reason for a caller that knows its step is inadmissible above `αmax` to be able to say so. It has
# no ceiling of its own — the whole point of the method is that `α` is the caller's to fix — so
# only `params.αmax` can bind here.
function solve_with_status(ls::Linesearch{T, <:Static}, α::T, params = NullParameters()) where {T}
    LinesearchStatus(min(method(ls).α, linesearch_αmax(method(ls), params)), LINESEARCH_UNKNOWN)
end

Base.show(io::IO, alg::Static) = print(io, "Static with α = " * string(alg.α) * ".")

function change_precision(::Type{T}, method::Static) where {T}
    T ≠ eltype(method) || return method
    Static{T}(T(method.α))
end

function Base.isapprox(st₁::Static{T}, st₂::Static{T}; kwargs...) where {T}
    isapprox(st₁.α, st₂.α; kwargs...)
end
