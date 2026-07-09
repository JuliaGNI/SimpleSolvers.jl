"""
    default_precision(T)

Compute the default precision used for e.g. [`BierlaireQuadratic`](@ref).

Compare this to the [`default_tolerance`](@ref) used in [`Options`](@ref).

# Examples

```jldoctest; setup = :(using SimpleSolvers: default_precision)
julia> default_precision(Float64)
1.7763568394002505e-15
```

```jldoctest; setup = :(using SimpleSolvers: default_precision)
julia> default_precision(Float32)
9.536743f-7
```

```jldoctest; setup = :(using SimpleSolvers: default_precision)
julia> default_precision(Float16)
ERROR: No default precision defined for Float16.
[...]
```
"""
function default_precision end

function default_precision(::Type{Float32})
    8eps(Float32)
end

function default_precision(::Type{Float64})
    8eps(Float64)
end

function default_precision(::Type{T}) where {T<:AbstractFloat}
    error("No default precision defined for $(T).")
end

"""
    shift_χ_to_avoid_stalling(χ, a, b, c, ε)

Check whether `b` is closer to `a` or `c` and shift `χ` accordingly.

This is taken from [bierlaire2015optimization](@cite).
"""
function shift_χ_to_avoid_stalling(χ::T, a::T, b::T, c::T, ε::T) where {T}
    if (c - b) > (b - a)
        χ + ε / 2
    else
        χ - ε / 2
    end
end


"""
    BierlaireQuadratic <: Linesearch

Algorithm taken from [bierlaire2015optimization](@cite).
"""
struct BierlaireQuadratic{T} <: LinesearchMethod{T}
    ε::T
    ξ::T

    function BierlaireQuadratic{T}(ε::T, ξ::T) where {T}
        new{T}(ε, ξ)
    end
end

function BierlaireQuadratic(::Type{T}=Float64;
    ε=default_precision(T), # previously DEFAULT_BIERLAIRE_ε,
    ξ=default_precision(T)  # previously DEFAULT_BIERLAIRE_ξ
) where {T}
    BierlaireQuadratic{T}(ε, ξ)
end

BierlaireQuadratic(::Type{T}, ::SolverMethod) where {T} = BierlaireQuadratic(T)

function solve(ls::Linesearch{T,<:BierlaireQuadratic}, a::T, b::T, c::T, params, iteration_number::Integer) where {T}
    f = x -> problem(ls).F(x, params)
    (iteration_number != max_number_of_quadratic_linesearch_iterations(T)) ||
        ((ls.config.verbosity >= 2 && @warn "Maximum number of iterations was reached."); return b)
    # The denominator vanishes when the three points are (nearly) collinear, i.e.
    # the quadratic fit is degenerate and χ becomes Inf/NaN or falls outside the
    # bracket.  Guard on the *result* (finite and inside [a, c]) rather than a
    # magnitude threshold on the denominator, since the denominator is legitimately
    # small near convergence while still yielding a valid interior minimum; on a
    # degenerate fit fall back to a bisection step of the bracket [a, c].
    # Evaluate f once per point and reuse: the fit, the χ comparison and the
    # termination check below all need the same values (previously f(a), f(b),
    # f(c) and f(χ) were recomputed several times each).
    fa = f(a)
    fb = f(b)
    fc = f(c)
    denom = fa * (b - c) + fb * (c - a) + fc * (a - b)
    χ = T(0.5) * (fa * (b^2 - c^2) + fb * (c^2 - a^2) + fc * (a^2 - b^2)) / denom
    (isfinite(χ) && a ≤ χ ≤ c) || (χ = (a + c) / 2)
    # perform a perturbation if χ ≈ b (in order "to avoid stalling"); use a tight
    # absolute tolerance so the perturbation only fires when χ is essentially at b
    # (the former `b == χ` only caught exact equality, missing floating-point ties)
    χ = isapprox(b, χ; atol=method(ls).ε) ? shift_χ_to_avoid_stalling(χ, a, b, c, method(ls).ε) : χ
    fχ = f(χ)
    # Carry the function values of the updated triple alongside the points, so the
    # termination check needs no further evaluations.
    if χ > b
        if fχ > fb
            c, fc = χ, fχ
        else
            a, fa = b, fb
            b, fb = χ, fχ
        end
    else
        if fχ > fb
            a, fa = χ, fχ
        else
            c, fc = b, fb
            b, fb = χ, fχ
        end
    end
    !(((c - a) ≤ method(ls).ε)) || !(((fa - fb) ≤ method(ls).ε) && ((fc - fb) ≤ method(ls).ε)) || return b
    solve(ls, a, b, c, params, iteration_number + 1)
end

function solve(ls::Linesearch{T,<:BierlaireQuadratic}, α₀::T, params, iteration_number::Integer) where {T}
    # check if the minimum has already been reached
    !(l2norm(derivative(problem(ls), α₀, params)) < method(ls).ξ) || return α₀
    solve(ls, triple_point_finder(problem(ls), params, α₀)..., params, iteration_number)
end

function solve(ls::Linesearch{T,<:BierlaireQuadratic}, α₀::T, params=NullParameters()) where {T}
    # Design note (Phase 5, resolving the former "use α₀" TODO): `triple_point_finder`
    # requires the merit to be decreasing at its starting point and only searches to
    # the right, so the bracket must start at α = 0 (where φ'(0) < 0 for a descent
    # direction), not at the caller's α₀ — starting at α₀ errors whenever the optimal
    # step is smaller than α₀.  Using α₀ as the triple-point step size δ likewise
    # over-coarsens the search on stiff problems.  The step magnitude is governed by
    # `triple_point_finder`'s tuned default, not by α₀.
    solve(ls, zero(T), params, 1)
end



Base.show(io::IO, ls::BierlaireQuadratic) = print(io, "Bierlaire Quadratic with ε = " * string(ls.ε) * ", and ξ = " * string(ls.ξ) * ".")

function change_precision(::Type{T}, method::BierlaireQuadratic{AT}) where {T,AT}
    T ≠ AT || return method
    if method.ε == default_precision(AT) && method.ξ == default_precision(AT)
        BierlaireQuadratic{T}(default_precision(T), default_precision(T))
    else
        BierlaireQuadratic{T}(T(method.ε), T(method.ξ))
    end
end

function Base.isapprox(bq₁::BierlaireQuadratic{T}, bq₂::BierlaireQuadratic{T}; kwargs...) where {T}
    isapprox(bq₁.ε, bq₂.ε; kwargs...) && isapprox(bq₁.ξ, bq₂.ξ; kwargs...)
end
