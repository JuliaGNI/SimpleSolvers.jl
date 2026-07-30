@doc raw"""
    bisection(f, αmin, αmax, params, config)

Perform bisection of `f` in the interval [`αmin`, `αmax`] with [`Options`](@ref) `config`.

The algorithm is repeated until a root is found (up to tolerance `config.f_abstol` which is determined by [`default_tolerance`](@ref) by default).

!!! info
    When calling `bisection` it first checks if ``x_\mathrm{min} < x_\mathrm{max}`` and else flips the two entries.

!!! info
    You can also call `bisection` with only one `x` as input argument. It then uses [`bracket_minimum`](@ref) to find a suitable interval.

# Extended help

The bisection algorithm divides an interval into equal halves until a root is found (up to a desired accuracy).

We first initialize:
```math
\begin{aligned}
\alpha_0 \gets & \alpha_\mathrm{min}, \\
\alpha_1 \gets & \alpha_\mathrm{max},
\end{aligned}
```
and then repeat:
```math
\begin{aligned}
& \alpha \gets \frac{\alpha_0 + \alpha_1}{2}, \\
& \text{if $f(\alpha_0)f(\alpha) > 0$} \\
& \qquad \alpha_0 \gets \alpha, \\
& \text{else} \\
& \qquad \alpha_1 \gets \alpha, \\
& \text{end}
\end{aligned}
```
So the algorithm checks in each step where the sign change occurred and moves the ``\alpha_0`` or ``\alpha_1`` accordingly. The loop is terminated if `config.linesearch_max_iterations` is reached (by default """ * """$(linesearch_iterations(Float64)) for `Float64` in the [`Options`](@ref) struct, see [`linesearch_iterations`](@ref)); in that case a warning is emitted (at `verbosity ≥ 1`) and the best estimate found so far is returned.

!!! warning
    The obvious danger with using bisections is that the supplied interval can have multiple roots (or no roots). One should be careful to avoid this when fixing the interval.

!!! info
    Bisection can only locate a root if the endpoints straddle a sign change. If the
    endpoints have the same sign there is no (odd-multiplicity) root in the interval;
    this arises benignly in the line search once the derivative has flattened at a
    minimum (both endpoint values ≈ 0 with the same sign). Rather than erroring,
    `bisection` then returns the endpoint closest to a root (smallest `|f|`) and warns
    only at high verbosity.
"""
function bisection(f::Callable, αmin::T, αmax::T, params=NullParameters(), config::Options=Options(float(T))) where {T<:Number}
    α, converged, _ = _bisection_core(f, αmin, αmax, params, config)
    converged || (config.verbosity ≥ 1 && @warn "Bisection did not converge within $(config.linesearch_max_iterations) iterations; returning best estimate α = $(α).")
    α
end

# The bisection loop, returning `(α, converged, n)` so a caller can report non-convergence
# itself instead of having this function log it, and report the evaluation count `n` as the
# `trials` of a `LinesearchStatus`. The `Bisection` *line search* needs both: it reports through
# `linesearch_warnings` like every other line search, so a message emitted from here would
# duplicate it and bypass the shared verbosity policy. The public `bisection` above keeps
# warning, since it is also used standalone as a root finder.
function _bisection_core(f::Callable, αmin::T, αmax::T, params, config::Options) where {T<:Number}
    n = 0
    R = float(T)
    α₀ = R(αmin)
    α₁ = R(αmax)
    α = zero(R)

    # flip α₀ and α₁ if the former is bigger than the latter
    α₀ < α₁ || begin
        α₀, α₁ = α₁, α₀
    end

    y₀ = f(α₀, params)
    y₁ = f(α₁, params)
    n += 2
    y = zero(y₀)

    if y₀ * y₁ > zero(y₀)
        config.verbosity ≥ 2 && @warn "Bisection bracket [$(α₀), $(α₁)] shows no sign change (f = $(y₀), $(y₁)); returning the endpoint with the smallest |f|."
        return (abs(y₀) ≤ abs(y₁) ? α₀ : α₁), true, n
    end

    converged = false
    for _ in 1:config.linesearch_max_iterations
        α = (α₀ + α₁) / 2
        y = f(α, params)
        n += 1

        # break if y is close to zero.
        if ≈(y, zero(y); atol=config.f_abstol)
            converged = true
            break
        end

        if y₀ * y > 0
            α₀ = α  # Root is in the right half of [α₀,α₁].
            y₀ = y
        else
            α₁ = α  # Root is in the left half of [α₀,α₁].
            # (no need to track y₁: the loop's sign test uses only y₀.)
        end

        if isapprox(α₁ - α₀, zero(α), atol=config.x_suctol * max(abs(α₀), abs(α₁)))
            converged = true
            break
        end
    end

    α, converged, n
end

function bisection(f::Callable, α::T, params=NullParameters(), config::Options=Options(float(T))) where {T<:Number}
    R = float(T)
    lo, hi = bracket_root(β -> f(β, params), R(α))
    bisection(f, lo, hi, params, config)
end

# Disambiguates `(f, ::T, ::T, ::Options)` in favor of the interval form with default `params`.
bisection(f::Callable, αmin::T, αmax::T, config::Options) where {T<:Number} = bisection(f, αmin, αmax, NullParameters(), config)

"""
    Bisection <: LinesearchMethod

See [`bisection`](@ref) for the implementation of the algorithm.

# Extended help

When invoked with a single trial step `α` (i.e. `solve(ls, α)`), the bracket is
always *lower-anchored* at ``\\alpha = 0`` — the only point where a genuine
descent direction is guaranteed to have a decreasing merit (``\\varphi'(0) < 0``),
which one-sided rightward bracketing requires. The caller's `α` is then folded in
via one extra derivative evaluation (see issue #164):

- if ``\\varphi'(\\alpha) \\geq 0`` then `α` overshot the minimum and ``[0, \\alpha]``
  already brackets a stationary point, so it is handed straight to [`bisection`](@ref)
  with no bracketing loop;
- otherwise ``\\alpha`` still lies on the descent side, so the bracket is grown
  outward from ``0`` with the initial step seeded from ``|\\alpha|`` — clamped
  between [`DEFAULT_BRACKETING_s`](@ref) and `1` so a large `α` does not
  over-coarsen the search and a tiny `α` does not crawl — rather than the fixed
  default step.

This keeps the safe ``\\alpha = 0`` anchor while letting the caller's `α` set the
search scale and, when it overshoots, serve directly as the upper bracket bound.
"""
struct Bisection{T} <: LinesearchMethod{T} end

Bisection(::Type{T}=Float64) where {T} = Bisection{T}()
Bisection(::Type{T}, ::SolverMethod) where {T} = Bisection(T)


# Bisect the derivative on a known bracket. Private: `solve`/`solve_with_status` is the public
# entry point (this used to be a `solve` overload, which made the public name ambiguous).
_bisect_on(ls::Linesearch{T,<:Bisection}, α₀::T, α₁::T, params) where {T} =
    _bisection_core(problem(ls).D, α₀, α₁, params, config(ls))

"""
    solve(ls::Linesearch{T,<:Bisection}, α, params)

Bisect the derivative of the merit to approximate the line minimiser, report the outcome
through [`linesearch_warnings`](@ref) and return the step length. See [`Bisection`](@ref) and
[`solve_with_status`](@ref).
"""
function solve(ls::Linesearch{T,<:Bisection}, α::T, params=NullParameters()) where {T}
    status = solve_with_status(ls, α, params)
    linesearch_warnings(status, ls, params)
    steplength(status)
end

function solve_with_status(ls::Linesearch{T,<:Bisection}, α::T, params=NullParameters()) where {T}
    prob = problem(ls)
    φ₀ = value(prob, zero(T), params)
    d₀ = derivative(prob, zero(T), params)

    anchor = check_anchor(φ₀, d₀, α)
    isnothing(anchor) || return anchor

    τ = armijo_tolerance(φ₀, T(DEFAULT_ARMIJO_τ_ULPS))

    # Lower-anchor the bracket at α = 0, where a genuine descent direction has a decreasing
    # merit (φ′(0) < 0, now guaranteed by `check_anchor`). Probe the caller's trial step α (one
    # extra derivative evaluation) to decide how to fold it in; see the docstring and #164.
    αres, converged, n = if α > zero(T) && derivative(prob, α, params) ≥ zero(T)
        # α overshot the minimum: [0, α] already brackets the stationary point.
        _bisect_on(ls, zero(T), α, params)
    else
        # α is on the descent side: grow the bracket from 0, seeding the step scale from |α|
        # (clamped) instead of the fixed default.
        s = clamp(abs(α), T(DEFAULT_BRACKETING_s), one(T))
        bracket = bracket_minimum(prob, params, zero(T), s)
        # `bracket_minimum` returns `nothing` only when `nmax` steps found no bracket in either
        # direction, i.e. for a merit that keeps decreasing. There is then no interval to bisect —
        # but that is a failure to *report*, not a round-off floor: calling it a floor would make
        # the outer iteration count a descending merit as stagnation.
        isnothing(bracket) && return LinesearchStatus{T}(α, LINESEARCH_EXHAUSTED, 0, φ₀, d₀, φ₀, τ, zero(T))
        _bisect_on(ls, bracket..., params)
    end
    # Non-convergence of the bisection is reported through the status rather than logged here,
    # so that `linesearch_warnings` remains the only place a line search emits messages.
    converged || return LinesearchStatus{T}(αres > zero(T) ? αres : α, LINESEARCH_EXHAUSTED, n, φ₀, d₀, value(prob, αres > zero(T) ? αres : α, params), τ, zero(T))

    # `bracket_minimum` flips direction when the merit rises to the right of its start, so the
    # bracket — and hence the bisected result — can lie left of zero. A negative step is not a
    # meaningful step length along a direction (see the α > 0 contract), so retry once from the
    # α = 0 anchor, which `check_anchor` has established is decreasing.
    if αres ≤ zero(T)
        bracket = bracket_minimum(prob, params, zero(T), T(DEFAULT_BRACKETING_s))
        if !isnothing(bracket)
            αres, _, nretry = _bisect_on(ls, bracket..., params)
            n += nretry
        end
    end
    # Still non-positive: no positive step improves the merit as far as this search can tell.
    # That is the floor, not a non-descent anchor — `check_anchor` established above that the
    # anchor *does* descend.
    αres > zero(T) || return LinesearchStatus{T}(α, LINESEARCH_FLOOR, n, φ₀, d₀, φ₀, τ, zero(T))

    φres = value(prob, αres, params)
    LinesearchStatus{T}(αres, φres ≤ φ₀ - τ ? LINESEARCH_DECREASED : LINESEARCH_FLOOR,
        n, φ₀, d₀, φres, τ, zero(T))
end

Base.show(io::IO, ::Bisection) = print(io, "Bisection")

function change_precision(::Type{T}, method::Bisection) where {T}
    T ≠ eltype(method) || return method
    Bisection(T)
end

Base.isapprox(::Bisection{T}, ::Bisection{T}; kwargs...) where {T} = true
