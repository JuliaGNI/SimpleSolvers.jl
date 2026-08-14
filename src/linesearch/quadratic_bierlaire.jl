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
Float16(0.007812)
```
"""
default_precision(::Type{T}) where {T<:AbstractFloat} = 8eps(T)

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


@doc raw"""
    BierlaireQuadratic <: LinesearchMethod

Algorithm taken from [bierlaire2015optimization](@cite).

# Keywords

- `ε`: the bracket-width tolerance of the fit.
- `ξ`: the threshold below which ``|\varphi'(\alpha_0)|`` counts as stationary.
- `αmax`: the largest step the triple-point bracketing will try, by default
  [`DEFAULT_LINESEARCH_αmax`](@ref). Without it the bracketing doubles its increment until the
  merit stops falling, which for a nearly flat or distantly-minimised ``\varphi`` is arbitrarily
  far; see [`linesearch_αmax`](@ref), which is also how a caller imposes a smaller ceiling of its
  own.
"""
struct BierlaireQuadratic{T} <: LinesearchMethod{T}
    ε::T
    ξ::T
    αmax::T

    function BierlaireQuadratic{T}(ε::T, ξ::T, αmax::T=default_linesearch_αmax(T)) where {T}
        @assert ε > 0 "Precision ε must be positive."
        @assert ξ > 0 "Derivative threshold ξ must be positive."
        @assert αmax > 0 "The maximum step length must be positive, it is $(αmax)."
        new{T}(ε, ξ, αmax)
    end
end

function BierlaireQuadratic(::Type{T}=Float64;
    ε=default_precision(T),
    ξ=default_precision(T),
    αmax=default_linesearch_αmax(T)
) where {T}
    BierlaireQuadratic{T}(ε, ξ, αmax)
end

method_αmax(m::BierlaireQuadratic) = m.αmax

BierlaireQuadratic(::Type{T}, ::SolverMethod) where {T} = BierlaireQuadratic(T)

# Run the three-point quadratic fit on a known bracket `a < b < c`. Returns `(b, f(b), n)` — the
# merit value is carried by the loop anyway, so the caller can classify the outcome without a
# further evaluation, and `n` is the number of merit evaluations, reported as the `trials` of the
# resulting `LinesearchStatus`. Private: `solve`/`solve_with_status` is the public entry point.
function _bierlaire_fit(ls::Linesearch{T,<:BierlaireQuadratic}, a::T, b::T, c::T, params, τ::T) where {T}
    # `n` is counted next to each evaluation rather than inside `f`: a counter that `f` both
    # captures and mutates is boxed, which makes the `Int` in the returned triple inferred-`Any`
    # and allocates for every fit and every `LinesearchStatus` built from one.
    f(x) = problem(ls).F(x, params)
    ε = method(ls).ε
    # Evaluate f once per point and reuse: the fit, the χ comparison and the
    # termination check below all need the same values.  The triple updates carry
    # the values along, so each loop round costs a single new evaluation (fχ).
    # (The former recursive formulation recomputed fa, fb and fc at every
    # recursion level, discarding the carried values.)
    fa = f(a)
    fb = f(b)
    fc = f(c)
    n = 3
    # Iterate rather than recurse: the depth is bounded by the iteration
    # maximum either way, but a loop keeps the stack flat and lets the
    # triple (a, b, c) and its values persist across rounds.
    # Width of the bracket, tracked so that a non-contracting iteration can be detected. Without
    # this the loop can sit on a single point until the budget runs out — see the comment on the
    # bisection fallback below.
    width = c - a
    for _ in 1:(config(ls).linesearch_max_iterations - 1)
        # The denominator vanishes when the three points are (nearly) collinear, i.e.
        # the quadratic fit is degenerate and χ becomes Inf/NaN or falls outside the
        # bracket.  Guard on the *result* (finite and inside [a, c]) rather than a
        # magnitude threshold on the denominator, since the denominator is legitimately
        # small near convergence while still yielding a valid interior minimum; on a
        # degenerate fit fall back to a bisection step of the bracket [a, c].
        denom = fa * (b - c) + fb * (c - a) + fc * (a - b)
        χ = T(0.5) * (fa * (b^2 - c^2) + fb * (c^2 - a^2) + fc * (a^2 - b^2)) / denom
        (isfinite(χ) && a ≤ χ ≤ c) || (χ = (a + c) / 2)
        # perform a perturbation if χ ≈ b (in order "to avoid stalling"); use a tight
        # absolute tolerance so the perturbation only fires when χ is essentially at b
        # (the former `b == χ` only caught exact equality, missing floating-point ties)
        χ = isapprox(b, χ; atol=ε) ? shift_χ_to_avoid_stalling(χ, a, b, c, ε) : χ
        # `shift_χ_to_avoid_stalling` is not sufficient to guarantee progress. If χ coincides
        # with a bracket point the update below cannot narrow the bracket: for χ == b both
        # branch tests are false (it is the same point, so the merits are identical), so the
        # triple becomes `c ← b`, `b ← b`, collapsing to `c == b` with `a` untouched. The next
        # fit then has two coincident points, `den == 0`, and the `(a + c)/2` fallback plus the
        # shift map straight back onto `b` — an iteration that never contracts and therefore
        # never satisfies the convergence test, spinning to `linesearch_max_iterations`.
        # Bisecting the wider sub-interval instead makes `c - a` strictly decrease every
        # iteration, bounding the loop at O(log₂((c - a)/ε)) regardless of the merit's scale.
        if χ == a || χ == b || χ == c
            χ = (c - b) > (b - a) ? (b + c) / 2 : (a + b) / 2
        end
        fχ = f(χ)
        n += 1
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
        # The bracket width is an α-space quantity, so an absolute `ε` is dimensionally right
        # (α is a step-length fraction of order one). The merit differences are *not*: they
        # scale with φ(0), so they are compared against the shared round-off allowance τ
        # instead of against ε. Previously one absolute constant governed all three.
        ((c - a) ≤ ε) && ((fa - fb) ≤ τ) && ((fc - fb) ≤ τ) && return (b, fb, n)

        # The bisection fallback above guarantees contraction, so a width that fails to shrink
        # means the bracket is already at the resolution of the arithmetic: nothing further can
        # be resolved and continuing would only repeat the same point.
        newwidth = c - a
        newwidth < width || return (b, fb, n)
        width = newwidth
    end
    (b, fb, n)
end

"""
    solve_with_status(ls::Linesearch{T,<:BierlaireQuadratic}, α, params)

Fit successive quadratics through three bracketing points to approximate the line minimiser and
return the [`LinesearchStatus`](@ref), emitting no messages. [`solve`](@ref) is this plus the
report; see [`BierlaireQuadratic`](@ref).
"""
function solve_with_status(ls::Linesearch{T,<:BierlaireQuadratic}, α₀::T, params=NullParameters()) where {T}
    prob = problem(ls)
    # Before any merit evaluation, so that an unusable caller-supplied ceiling costs none.
    αmax = linesearch_αmax(method(ls), params)
    φ₀ = value(prob, zero(T), params)
    d₀ = derivative(prob, zero(T), params)

    # `triple_point_finder` searches rightward only and requires a decreasing merit at its
    # start, so unlike the other bracketing searches it cannot recover from an ascent anchor by
    # flipping. Checking the anchor here is what keeps it from being handed an impossible
    # problem — the α = 0 anchor is *not* guaranteed decreasing when the direction came from a
    # stale or regularized Jacobian.
    anchor = check_anchor(φ₀, d₀, α₀, αmax)
    isnothing(anchor) || return anchor

    τ = armijo_tolerance(φ₀, armijo_ulps(T))
    # Every step this function can hand back is derived from the trial step or from the bracketing,
    # and both are bounded here rather than at each of the returns below.
    α₀ = min(α₀, αmax)

    # Near-stationarity shortcut: the minimum along this direction has already been reached.
    l2norm(derivative(prob, α₀, params)) < method(ls).ξ &&
        return LinesearchStatus{T}(α₀, LINESEARCH_STATIONARY, 0, φ₀, d₀, φ₀, τ, zero(T))

    # Start triple-point bracketing at the caller's α₀ when it lies on the descent side
    # (φ′(α₀) < 0, so the minimiser is to its right), otherwise at the α = 0 anchor, which
    # `check_anchor` has established is decreasing. See issue #164.
    start = (α₀ > zero(T) && derivative(prob, α₀, params) < zero(T)) ? α₀ : zero(T)
    # A start at or beyond the ceiling leaves nothing to bracket; the ceiling is the answer, and
    # `capped_status` is what says so with the merit measured there.
    start < αmax || return capped_status(prob, params, αmax, φ₀, d₀, τ)
    # `_triple_point_core` rather than `triple_point_finder`: its concrete return type costs no
    # allocation, and this runs once per line search.
    a, b, c, bracket = _triple_point_core(prob, params, start; αmax=αmax)
    # The two failures mean opposite things and must not be conflated: `:flat` says the merit does
    # not resolve a decrease from here, so no line search can improve on this point (a floor, which
    # the outer iteration counts as a stalled step), while `:unbracketable` says there *is* a
    # decrease that could not be bracketed — reporting that as a floor would count a descending
    # merit as stagnation.
    if bracket === :flat || bracket === :unbracketable
        oc = bracket === :flat ? LINESEARCH_FLOOR : LINESEARCH_EXHAUSTED
        return LinesearchStatus{T}(α₀, oc, 0, φ₀, d₀, φ₀, τ, zero(T))
    end

    # `:capped` is neither: the merit was still falling when the bracketing reached the largest
    # step the caller allows, so that step is the best admissible one and there is no triple to
    # fit.
    bracket === :capped && return capped_status(prob, params, αmax, φ₀, d₀, τ)

    αres, φres, n = _bierlaire_fit(ls, a, b, c, params, τ)
    # The fit is confined to `[a, c] ⊆ [0, αmax]`, so this cannot bind; it is here so that the
    # α ≤ αmax half of the contract is guaranteed by this function rather than inferred from the
    # bracketing.
    if αres > αmax
        αres = αmax
        φres = value(prob, αres, params)
    end
    # The fit works inside a bracket whose left end is ≥ 0, so a non-positive result means the
    # arithmetic collapsed rather than that the anchor ascends (`check_anchor` ruled that out):
    # no positive step improves the merit as far as this search can tell, which is the floor.
    αres > zero(T) || return LinesearchStatus{T}(α₀, LINESEARCH_FLOOR, n, φ₀, d₀, φ₀, τ, zero(T))

    LinesearchStatus{T}(αres, φres ≤ φ₀ - τ ? LINESEARCH_DECREASED : LINESEARCH_FLOOR,
        n, φ₀, d₀, φres, τ, zero(T))
end



Base.show(io::IO, ls::BierlaireQuadratic) = print(io, "Bierlaire Quadratic with ε = " * string(ls.ε) * ", ξ = " * string(ls.ξ) * ", and αmax = " * string(ls.αmax) * ".")

function change_precision(::Type{T}, method::BierlaireQuadratic{AT}) where {T,AT}
    T ≠ AT || return method
    if method.ε == default_precision(AT) && method.ξ == default_precision(AT)
        BierlaireQuadratic{T}(default_precision(T), default_precision(T), convert_αmax(T, method.αmax))
    else
        BierlaireQuadratic{T}(T(method.ε), T(method.ξ), convert_αmax(T, method.αmax))
    end
end

function Base.isapprox(bq₁::BierlaireQuadratic{T}, bq₂::BierlaireQuadratic{T}; kwargs...) where {T}
    isapprox(bq₁.ε, bq₂.ε; kwargs...) && isapprox(bq₁.ξ, bq₂.ξ; kwargs...) && isapprox(bq₁.αmax, bq₂.αmax; kwargs...)
end
