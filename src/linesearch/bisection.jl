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
    endpoints have the same sign there is no (odd-multiplicity) root in the interval.
    Rather than erroring — a line search must not abort the enclosing solve — `bisection`
    returns the endpoint closest to a root (smallest `|f|`) and *reports* the failure:
    `_bisection_core` distinguishes it from a located root with
    `BISECTION_NOBRACKET`, and `bisection` warns accordingly.
"""
function bisection(f::Callable, αmin::T, αmax::T, params=NullParameters(), config::Options=Options(float(T))) where {T<:Number}
    α, outcome, _ = _bisection_core(f, αmin, αmax, params, config)
    # Each failure gets the wording that fits it. The two are not interchangeable: a spent budget
    # says "the interval does contain a root, I did not narrow it far enough", a failed bracket says
    # "there is no root in this interval at all", and only the first is fixed by a larger budget.
    outcome === BISECTION_EXHAUSTED && report_bisection_nonconvergence(α, config)
    outcome === BISECTION_NOBRACKET && report_bisection_nobracket(αmin, αmax, α, config)
    α
end

# Behind barriers because `bisection` and `_bisection_core` are specialized on the merit closure `f`
# — see `report_linesearch_status`.
@noinline function report_bisection_nonconvergence(α::Number, config::Options)
    verbosity(config) ≥ 1 && @warn "Bisection did not converge within $(config.linesearch_max_iterations) iterations; returning best estimate α = $(α)."
    nothing
end

@noinline function report_bisection_nobracket(αmin::Number, αmax::Number, α::Number, config::Options)
    lo, hi = minmax(αmin, αmax)
    verbosity(config) ≥ 1 && @warn "Bisection bracket [$(lo), $(hi)] shows no sign change, so it contains no root of odd multiplicity and no bisection can locate one in it. Returning the endpoint with the smallest |f|, α = $(α)."
    nothing
end

@doc raw"""
    BisectionOutcome

Why `_bisection_core` stopped. This is what lets a caller tell *"found the root"* from
*"gave up"* — a distinction a `Bool` cannot carry, and whose absence made an unbracketable
derivative look like a located line minimiser (see [`Bisection`](@ref)).

- `BISECTION_CONVERGED`: a root was located, either because ``|f(\alpha)| \leq`` `f_abstol` or
  because the bracket collapsed to `x_suctol`.
- `BISECTION_NOBRACKET`: the endpoint values share a sign, so the interval contains no root of
  odd multiplicity and bisection cannot start. The endpoint with the smallest ``|f|`` is returned,
  but it is *not* a root — this is a failure to report, not a result.
- `BISECTION_EXHAUSTED`: the `linesearch_max_iterations` budget of [`Options`](@ref) was spent
  with the interval still straddling a sign change. Unlike `BISECTION_NOBRACKET` there *is* a root
  in the interval, so a larger budget would find it; the best estimate so far is returned.

This is internal: it is the return type of a private function, like the `Symbol` that
`_triple_point_core` reports. Callers of a *line search* see a
[`LinesearchOutcome`](@ref) instead.
"""
@enum BisectionOutcome::Int8 begin
    BISECTION_CONVERGED
    BISECTION_NOBRACKET
    BISECTION_EXHAUSTED
end

# The bisection loop, returning `(α, outcome, n)` so a caller can report the failure itself
# instead of having this function log it, and report the evaluation count `n` as the `trials` of
# a `LinesearchStatus`. The `Bisection` *line search* needs all three: it reports through
# `linesearch_warnings` like every other line search, so a message emitted from here would
# duplicate it and bypass the shared verbosity policy. The public `bisection` above keeps
# warning, since it is also used standalone as a root finder.
#
# The outcome is a `BisectionOutcome` and not a `Bool` because "no sign change" used to be folded
# into `converged = true` — the endpoint with the smallest |f| was returned and *claimed* as a
# root. That claim then propagated into `LINESEARCH_FLOOR`, which asserts that no step can decrease
# the merit; a failed bracket establishes nothing of the kind. See `solve_with_status` below.
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
        return (abs(y₀) ≤ abs(y₁) ? α₀ : α₁), BISECTION_NOBRACKET, n
    end

    outcome = BISECTION_EXHAUSTED
    for _ in 1:config.linesearch_max_iterations
        α = (α₀ + α₁) / 2
        y = f(α, params)
        n += 1

        # break if y is close to zero.
        if ≈(y, zero(y); atol=config.f_abstol)
            outcome = BISECTION_CONVERGED
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
            outcome = BISECTION_CONVERGED
            break
        end
    end

    α, outcome, n
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

# Keywords

- `αmax`: the largest step the bracketing will try, by default
  [`DEFAULT_LINESEARCH_αmax`](@ref). See [`linesearch_αmax`](@ref), which is also how a caller
  imposes a smaller ceiling of its own.

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

## A bracket that fails is never a floor

Bisection drives on the *sign* of ``\\varphi'``, so it can only work on an interval whose
endpoints straddle a sign change. When [`bracket_minimum`](@ref) hands it an interval that
brackets a minimum in *value* but over which ``\\varphi'`` keeps its sign — a non-smooth or
noisy merit, or a derivative inconsistent with it — there is nothing to bisect and
`_bisection_core` reports `BISECTION_NOBRACKET`.

That case used to be folded into "converged", so the endpoint with the smallest ``|\\varphi'|``
was *claimed* as the line minimiser and, when it did not improve the merit, classified as
`LINESEARCH_FLOOR` — which asserts that **no** line search can make progress along this
direction and makes the outer iteration count the step towards `max_stalls`. A failed bracket
establishes nothing of the kind. So the outcome is now classified by the merit alone:
`LINESEARCH_DECREASED` when the returned step still beats ``\\varphi(0)`` by more than
``\\tau``, and `LINESEARCH_EXHAUSTED` when it does not. `LINESEARCH_FLOOR` is reachable only
from a bisection that actually converged.
"""
struct Bisection{T} <: LinesearchMethod{T}
    αmax::T

    function Bisection{T}(αmax::T=T(DEFAULT_LINESEARCH_αmax)) where {T}
        @assert αmax > 0 "The maximum step length must be positive, it is $(αmax)."
        new{T}(αmax)
    end
end

Bisection(::Type{T}=Float64; αmax=T(DEFAULT_LINESEARCH_αmax)) where {T} = Bisection{T}(T(αmax))
Bisection(::Type{T}, ::SolverMethod) where {T} = Bisection(T)

method_αmax(m::Bisection) = m.αmax


# Bisect the derivative on a known bracket. Private: `solve`/`solve_with_status` is the public
# entry point (this used to be a `solve` overload, which made the public name ambiguous).
_bisect_on(ls::Linesearch{T,<:Bisection}, α₀::T, α₁::T, params) where {T} =
    _bisection_core(problem(ls).D, α₀, α₁, params, config(ls))

"""
    solve_with_status(ls::Linesearch{T,<:Bisection}, α, params)

Bisect the derivative of the merit to approximate the line minimiser and return the
[`LinesearchStatus`](@ref), emitting no messages. [`solve`](@ref) is this plus the report; see
[`Bisection`](@ref).
"""
function solve_with_status(ls::Linesearch{T,<:Bisection}, α::T, params=NullParameters()) where {T}
    prob = problem(ls)
    # Before any merit evaluation, so that an unusable caller-supplied ceiling costs none.
    αmax = linesearch_αmax(method(ls), params)
    φ₀ = value(prob, zero(T), params)
    d₀ = derivative(prob, zero(T), params)

    anchor = check_anchor(φ₀, d₀, α, αmax)
    isnothing(anchor) || return anchor

    τ = armijo_tolerance(φ₀, armijo_ulps(T))

    # Every step this function can hand back is derived from the trial step or from the bracketing,
    # and both are bounded here rather than at each of the five returns below.
    α = min(α, αmax)

    # Lower-anchor the bracket at α = 0, where a genuine descent direction has a decreasing
    # merit (φ′(0) < 0, now guaranteed by `check_anchor`). Probe the caller's trial step α (one
    # extra derivative evaluation) to decide how to fold it in; see the docstring and #164.
    αres, bres, n = if α > zero(T) && derivative(prob, α, params) ≥ zero(T)
        # α overshot the minimum: [0, α] already brackets the stationary point.
        _bisect_on(ls, zero(T), α, params)
    else
        # α is on the descent side: grow the bracket from 0, seeding the step scale from |α|
        # (clamped) instead of the fixed default.
        s = clamp(abs(α), T(DEFAULT_BRACKETING_s), one(T))
        lo, hi, bstatus = _bracket_minimum_core(prob, params, zero(T), s, T(DEFAULT_BRACKETING_k), DEFAULT_BRACKETING_nmax, αmax)
        # The bracket ends at the ceiling with the merit still falling across it, so the minimiser
        # lies beyond the largest step the caller allows and that step is the best admissible one.
        # There is nothing to bisect: the derivative keeps its sign on `[0, αmax]` by construction,
        # so bisection would report `BISECTION_NOBRACKET` and hand back an endpoint chosen by
        # `|φ′|` rather than by the merit.
        bstatus === :capped && return capped_status(prob, params, αmax, φ₀, d₀, τ)
        # `_bracket_minimum_core` reports `:unbracketable` only when `nmax` steps found no bracket
        # in either direction, i.e. for a merit that keeps decreasing. There is then no interval to
        # bisect — but that is a failure to *report*, not a round-off floor: calling it a floor
        # would make the outer iteration count a descending merit as stagnation.
        # The merit *is* evaluated at the step handed back, at the cost of one evaluation on a path
        # that has already failed. Filling `φ` with `φ₀` instead would be a claim that the merit did
        # not change, which is what `linesearch_exhausted_reason` would then interpolate into its
        # message — and for a merit that descends forever it is exactly wrong. (The `check_anchor`
        # returns do fill `φ` with `φ₀`, deliberately: the whole point of an ascent or stationary
        # anchor is that no trial step is worth an evaluation.)
        bstatus === :unbracketable && return LinesearchStatus{T}(α, LINESEARCH_EXHAUSTED, 0, φ₀, d₀, value(prob, α, params), τ, zero(T))
        _bisect_on(ls, lo, hi, params)
    end
    # A spent budget is reported through the status rather than logged here, so that
    # `linesearch_warnings` remains the only place a line search emits messages. Note that this
    # is *only* the budget case: a failed bracket carries on below, because the endpoint it
    # returns may still improve the merit and there is no reason to throw that away.
    bres === BISECTION_EXHAUSTED &&
        return LinesearchStatus{T}(αres > zero(T) ? αres : α, LINESEARCH_EXHAUSTED, n, φ₀, d₀, value(prob, αres > zero(T) ? αres : α, params), τ, zero(T))

    # `bracket_minimum` flips direction when the merit rises to the right of its start, so the
    # bracket — and hence the bisected result — can lie left of zero. A negative step is not a
    # meaningful step length along a direction (see the α > 0 contract), so retry once from the
    # α = 0 anchor, which `check_anchor` has established is decreasing.
    if αres ≤ zero(T)
        bracket = bracket_minimum(prob, params, zero(T), T(DEFAULT_BRACKETING_s), T(DEFAULT_BRACKETING_k), DEFAULT_BRACKETING_nmax, αmax)
        if !isnothing(bracket)
            αres, bretry, nretry = _bisect_on(ls, bracket..., params)
            n += nretry
            # The retry's own verdict counts too: a retry that could not bracket either must not
            # be allowed to claim the floor below, for exactly the reason the first one may not.
            # Note that this keeps the *worse* of the two verdicts rather than replacing the first
            # with the retry's, even though `αres` now comes wholly from the retry. That is
            # deliberate. The two disagree only when a failed bracket is followed by a converged
            # retry, which needs φ′ to be inconsistent with φ (a derivative that agrees with the
            # merit changes sign inside a bracket `bracket_minimum` found, so the first bisection
            # would have converged) — and `LINESEARCH_EXHAUSTED`, whose reported reason is exactly
            # "φ′(0) is inconsistent with the merit", is then the better diagnosis of the two.
            bretry === BISECTION_CONVERGED || (bres = bretry)
        end
    end

    # A bisection that could not bracket has located nothing, so the only honest thing left to say
    # about its endpoint is what the *merit* says about it. `LINESEARCH_FLOOR` — "no step can
    # decrease φ, so no line search can help here" — is a claim only a converged bisection has
    # earned; a failed bracket that did not improve the merit is `LINESEARCH_EXHAUSTED`, which says
    # no acceptable step was found while leaving open that one exists.
    bracketed = bres === BISECTION_CONVERGED
    nodecrease = bracketed ? LINESEARCH_FLOOR : LINESEARCH_EXHAUSTED

    # Still non-positive: no positive step improves the merit as far as this search can tell.
    # That is the floor, not a non-descent anchor — `check_anchor` established above that the
    # anchor *does* descend. As on the no-bracket path above, `φ` is the merit at the step handed
    # back and not `φ₀`: the caller's `α` is what is returned here, and nothing has measured the
    # merit there.
    αres > zero(T) || return LinesearchStatus{T}(α, nodecrease, n, φ₀, d₀, value(prob, α, params), τ, zero(T))

    # The bracketing stops at the ceiling and bisection only narrows the interval it is given, so
    # this cannot bind; it is here so that the α ≤ αmax half of the contract is guaranteed by this
    # function rather than inferred from the helpers that feed it.
    αres = min(αres, αmax)
    φres = value(prob, αres, params)
    LinesearchStatus{T}(αres, φres ≤ φ₀ - τ ? LINESEARCH_DECREASED : nodecrease,
        n, φ₀, d₀, φres, τ, zero(T))
end

Base.show(io::IO, ls::Bisection) = print(io, "Bisection with αmax = $(ls.αmax).")

function change_precision(::Type{T}, method::Bisection) where {T}
    T ≠ eltype(method) || return method
    Bisection{T}(T(method.αmax))
end

Base.isapprox(b₁::Bisection{T}, b₂::Bisection{T}; kwargs...) where {T} = isapprox(b₁.αmax, b₂.αmax; kwargs...)
