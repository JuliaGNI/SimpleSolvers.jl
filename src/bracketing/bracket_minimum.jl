"""
    const DEFAULT_BRACKETING_s

Gives the default initial width of the interval (the bracket). Used for [`bracket_minimum`](@ref), [`bracket_minimum_with_fixed_point`](@ref) and [`bracket_root`](@ref).
"""
const DEFAULT_BRACKETING_s = 1E-2

"""
    const DEFAULT_BRACKETING_k

Gives the default ratio by which the bracket is increased if bracketing was not successful. See [`bracket_minimum`](@ref).
"""
const DEFAULT_BRACKETING_k = 2.0

"Default constant. Number of maximum iterations for [`bracket_minimum`](@ref), [`bracket_minimum_with_fixed_point`](@ref) and [`bracket_root`](@ref)."
const DEFAULT_BRACKETING_nmax = 100

"""
    BracketingCriterion

Abstract type for the criteria used while bracketing. It determines when a bracket has been found.
The two concrete subtypes are [`BracketMinimumCriterion`](@ref) (used by [`bracket_minimum`](@ref)) and
[`BracketRootCriterion`](@ref) (used by [`bracket_root`](@ref)).
"""
abstract type BracketingCriterion end

"""
    BracketMinimumCriterion <: BracketingCriterion

The criterion used for [`bracket_minimum`](@ref). It checks whether ``y(c)`` is greater than or equal to ``y(b)`` (i.e. checks whether we are passed the minimum).
Compare this with [`BracketRootCriterion`](@ref).

# Functor

```jldoctest; setup = :(using SimpleSolvers: BracketMinimumCriterion)
bc = BracketMinimumCriterion()

yc = .1
yb = .2

bc(yb, yc)

# output

false
```
"""
struct BracketMinimumCriterion <: BracketingCriterion end

"""
    BracketRootCriterion <: BracketingCriterion

The criterion used for [`bracket_root`](@ref). It checks whether there is a sign change between ``b`` and ``c`` (i.e. checks whether there is a root between those two points).
Compare this with [`BracketMinimumCriterion`](@ref).

# Functor

```jldoctest; setup = :(using SimpleSolvers: BracketRootCriterion)
bc = BracketRootCriterion()

yc = .1
yb = -.2

bc(yb, yc)

# output

true
```
"""
struct BracketRootCriterion <: BracketingCriterion end
(::BracketMinimumCriterion)(yb::T, yc::T) where {T<:Number} = yc ≥ yb
(::BracketRootCriterion)(yb::T, yc::T) where {T<:Number} = yc * yb ≤ zero(T)

"""
    bracket(f, x, bc, s, k, nmax, αmax)

Grow a bracket outward from `x` (in steps scaled by `k`, starting from `s`) until
the [`BracketingCriterion`](@ref) `bc` is satisfied. Used by [`bracket_minimum`](@ref)
and [`bracket_root`](@ref).

# Extended help

Before entering the main loop we check whether the criterion is already satisfied
just to the *left* of `a` (at `a - s`). This early exit is only valid for the
[`BracketRootCriterion`](@ref), where it corresponds to a sign change in
`(a - s, b)`. For the [`BracketMinimumCriterion`](@ref) it would instead bracket a
maximum rather than a minimum, so it is deliberately skipped.

Returns `nothing` when no bracket is found within `nmax` steps. A line search must be able to
report an unbracketable merit rather than abort the enclosing solve, so this is a `nothing`
rather than an error; see [`bracket_minimum`](@ref).

`αmax` bounds how far to the *right* the search may probe (see [`linesearch_αmax`](@ref)); a
bracket truncated by it is returned like any other, and only [`_bracket_core`](@ref) distinguishes
the two.
"""
function bracket(f::Callable, x::T, bc::BracketingCriterion, s::T=T(DEFAULT_BRACKETING_s), k::T=T(DEFAULT_BRACKETING_k), nmax::Integer=DEFAULT_BRACKETING_nmax, αmax::T=T(Inf)) where {T<:Number}
    lo, hi, _, status = _bracket_core(f, x, bc, s, k, nmax, αmax)
    status === :unbracketable ? nothing : (lo, hi)
end

"""
    _bracket_core(f, x, bc, s, k, nmax, αmax)

The loop of [`bracket`](@ref), returning `(lo, hi, n, status)` with `n` the number of evaluations
of `f` it spent and `status` one of `:ok`, `:capped` or `:unbracketable`. Splitting the loop from
the reporting is what `bisection`/`_bisection_core` and
[`triple_point_finder`](@ref)/`_triple_point_core` already do, and here it is what lets a
caller tell a bracket that *ends* at the ceiling `αmax` from one that satisfied the criterion:
the first says the turning point lies beyond the largest step the caller allows, so the answer is
`αmax` itself, and the interval is not worth fitting anything to.

`n` is reported for the same reason the status is: it is the cost of the bracketing, and for the
searches that bracket it is most of the cost of the whole line search. Without it the `trials` of
their [`LinesearchStatus`](@ref) could only ever be a lower bound — vacuous on the path where the
bracketing *is* the search, which is exactly the path a ceiling produces.

Private: [`bracket`](@ref) and [`bracket_minimum`](@ref) are the public entry points.
"""
function _bracket_core(f::Callable, x::T, bc::BracketingCriterion, s::T, k::T, nmax::Integer, αmax::T) where {T<:Number}
    a = x
    n = 0

    # The *first* probe is bounded too, not just the loop's. A ceiling smaller than the initial
    # step `s` would otherwise have the merit evaluated outside the range the caller called
    # admissible before the loop ever tests the bound — one evaluation, but on a problem where a
    # step beyond `αmax` is meaningless it is exactly the evaluation the ceiling exists to avoid.
    # `s` is left alone: it is the growth scale, and clamping it would also shrink the loop's steps.
    b = s > zero(T) ? min(a + s, αmax) : a + s
    yb = f(b)
    n += 1

    if bc isa BracketRootCriterion
        n += 1
        bc(f(a - s), yb) && return (a - s, b, n, :ok)
    end

    for _ in 1:nmax
        c = b + s
        # The ceiling bounds a *step length*, so it applies only while the search runs to the
        # right. A negative `s` means the caller flipped the search (see `bracket_minimum`), and
        # the α > 0 contract of the line-search layer is what handles that side.
        if s > zero(T) && c ≥ αmax
            # Probe *at* the ceiling rather than past it: it is the last point the caller allows,
            # and evaluating it is what lets the search report the merit at the step it returns.
            c = αmax
            c > a || return (a, a, n, :capped)
            # …unless `b` is already *at* the ceiling, which it is whenever the first probe was
            # clamped to it (`s ≥ αmax - x`, so any caller ceiling at or below the initial step).
            # There is then nothing further to the right to probe, and probing anyway compares the
            # point with itself: `yc == yb` satisfies `BracketMinimumCriterion` (`yc ≥ yb`), so the
            # truncation would be reported as `:ok` — a turning point that is one point counted
            # twice — and the caller would bisect a derivative, or fit a polynomial, on an interval
            # over which the merit only falls.
            c > b || return (a, b, n, :capped)
            yc = f(c)
            n += 1
            return (a, c, n, bc(yb, yc) ? :ok : :capped)
        end
        yc = f(c)
        n += 1
        if bc(yb, yc)
            interval = a < c ? (a, c) : (c, a)
            return (interval..., n, :ok)
        end
        a = b
        b = c
        yb = yc
        s *= k
    end
    (a, b, n, :unbracketable)
end

@doc raw"""
    bracket_minimum(f, x)

Move a bracket successively in the search direction (starting at `x`) and increase its size until a local minimum of `f` is found.

This is used in [`bisection`](@ref)s when only one `x` is given (and not an entire interval).

This bracketing algorithm is taken from [kochenderfer2019algorithms](@cite). Also compare it to [`bracket_minimum_with_fixed_point`](@ref).

# Arguments

- `f`: the function to be bracketed,
- `x`: the starting point,
- `s`: by default [`DEFAULT_BRACKETING_s`](@ref),
- `k`: by default [`DEFAULT_BRACKETING_k`](@ref),
- `nmax`: by default [`DEFAULT_BRACKETING_nmax`](@ref),
- `αmax`: how far to the *right* the search may probe, by default `Inf` (see
  [`linesearch_αmax`](@ref)). A bracket truncated by it is returned like any other, so a caller
  that has to tell the two apart uses `_bracket_minimum_core`.

# Extended help

For bracketing we need two constants ``s`` and ``k`` (see [`DEFAULT_BRACKETING_s`](@ref) and [`DEFAULT_BRACKETING_k`](@ref)).

Before we start the algorithm we *initialize* it, i.e. we check that we indeed have a descent direction:
```math
\begin{aligned}
& a \gets x, \\
& b \gets a + s, \\
& \mathrm{if} \quad f(b) > f(a)\\
& \qquad\text{Flip $a$ and $b$ and set $s\gets-s$.}\\
& \mathrm{end}
\end{aligned}
```

The algorithm then successively computes:
```math
c \gets b + s,
```

and then checks whether ``f(c) \geq f(b)`` (also see [`BracketMinimumCriterion`](@ref)). If this is true it returns ``(a, c)`` or ``(c, a)``, depending on whether ``a<c`` or ``c<a`` respectively.
If this is not satisfied ``a,`` ``b`` and ``s`` are updated:
```math
\begin{aligned}
a \gets & b, \\
b \gets & c, \\
s \gets & sk,
\end{aligned}
```
and the algorithm is continued. If we have not found a bracket after ``n_\mathrm{max}`` iterations (see [`DEFAULT_BRACKETING_nmax`](@ref)) the algorithm terminates and returns `nothing`.
The interval that is returned by `bracket_minimum` is then typically used as a starting point for [`bisection`](@ref).

!!! warning "Returns `nothing` on failure"
    A line search must be able to *report* a merit it cannot bracket rather than abort the
    enclosing solve, so an unbracketable `f` yields `nothing` rather than an error. Callers must
    handle it — see [`solve_with_status`](@ref) and [`LinesearchOutcome`](@ref).

!!! info
    The function [`bracket_root`](@ref) is equivalent to `bracket_minimum` with the only difference that the criterion we check for is:
    ```math
    f(c)f(b) < 0,
    ```
    i.e. that a sign change in the function occurs. Also see [`BracketRootCriterion`](@ref).

"""
function bracket_minimum(f::Callable, x::T, s::T, k::T=T(DEFAULT_BRACKETING_k), nmax::Integer=DEFAULT_BRACKETING_nmax, αmax::T=T(Inf)) where {T<:Number}
    lo, hi, _, status = _bracket_minimum_core(f, x, s, k, nmax, αmax)
    status === :unbracketable ? nothing : (lo, hi)
end

"""
    _bracket_minimum_core(f, x, s, k, nmax, αmax)

[`bracket_minimum`](@ref) with the `(lo, hi, n, status)` return of [`_bracket_core`](@ref), so that
a caller can tell a bracket truncated at the ceiling `αmax` from one that found a turning point and
can charge its cost `n` to the `trials` it reports. Private; see [`_bracket_core`](@ref).
"""
function _bracket_minimum_core(f::Callable, x::T, s::T, k::T, nmax::Integer, αmax::T) where {T<:Number}
    a = x
    ya = f(a)
    n = 1

    # This probe decides the *direction* and is taken before `_bracket_core` sees the problem at
    # all, so it needs the same bound `_bracket_core`'s own first probe carries — otherwise a
    # ceiling below `s` is stepped over here and the bound in the loop never gets the chance.
    b = s > zero(T) ? min(a + s, αmax) : a + s
    yb = f(b)
    n += 1

    # flip a & b if necessary
    if yb > ya
        a, b = b, a
        ya, yb = yb, ya
        s = -s
    elseif s > zero(T) && b ≥ αmax && b > a
        # The direction probe was clamped to the ceiling *and* the merit fell to it, so there is
        # nothing further to the right that the caller allows: the bracket ends here. Returning
        # now rather than handing this to `_bracket_core` is what keeps the ceiling from being
        # evaluated twice — this function and `_bracket_core` both probe `a + s`, and clamped to
        # the same ceiling that is the same point.
        return (a, b, n, :capped)
    end

    lo, hi, ncore, status = _bracket_core(f, a, BracketMinimumCriterion(), s, k, nmax, αmax)
    (lo, hi, n + ncore, status)
end

function bracket_minimum(f::Callable, x::T; s::T=T(DEFAULT_BRACKETING_s), k::T=T(DEFAULT_BRACKETING_k), nmax::Integer=DEFAULT_BRACKETING_nmax, αmax::T=T(Inf)) where {T<:Number}
    bracket_minimum(f, x, s, k, nmax, αmax)
end

function bracket_minimum(prob::LinesearchProblem{T}, params, x::T, s::T, k::T=T(DEFAULT_BRACKETING_k), nmax::Integer=DEFAULT_BRACKETING_nmax, αmax::T=T(Inf)) where {T<:Number}
    bracket_minimum(x -> value(prob, x, params), x, s, k, nmax, αmax)
end

function bracket_minimum(prob::LinesearchProblem{T}, params, x::T; s::T=T(DEFAULT_BRACKETING_s), k::T=T(DEFAULT_BRACKETING_k), nmax::Integer=DEFAULT_BRACKETING_nmax, αmax::T=T(Inf)) where {T<:Number}
    bracket_minimum(prob, params, x, s, k, nmax, αmax)
end

_bracket_minimum_core(prob::LinesearchProblem{T}, params, x::T, s::T, k::T=T(DEFAULT_BRACKETING_k), nmax::Integer=DEFAULT_BRACKETING_nmax, αmax::T=T(Inf)) where {T<:Number} =
    _bracket_minimum_core(x -> value(prob, x, params), x, s, k, nmax, αmax)

@doc raw"""
    bracket_minimum_with_fixed_point(f, x, s, k, nmax, αmax)

Find a bracket while keeping the left side (i.e. `x`) fixed.

The algorithm is similar to [`bracket_minimum`](@ref) (also based on [`DEFAULT_BRACKETING_s`](@ref) and [`DEFAULT_BRACKETING_k`](@ref)) with the difference that for the latter the left side is also moving.

The function `bracket_minimum_with_fixed_point` is used as a starting point for [`Quadratic`](@ref) (adapted from [kelley1995iterative](@cite)): that line search fits the polynomial centred at the bracket's left endpoint ``a`` (which may differ from the input `x` after the bracketer’s initial direction flip),
```math
p(\alpha) = f(a) + f'(a)(\alpha - a) + p_2(\alpha - a)^2,
```
so interpolating ``f`` at the right endpoint ``b`` fixes the coefficient ``p_2`` as
```math
p_2 = \frac{f(b) - f(a) - f'(a)(b - a)}{(b - a)^2},
```
where ``(a, b)`` is the bracket returned by ``\mathtt{bracket\_minimum\_with\_fixed\_point}``. The right end `b` is
grown outward (with the left end `a` held fixed) until `f` stops decreasing, i.e.
until the *turning point* `f(b) ≥ f(b_\mathrm{prev})` is reached, so that a minimum
is bracketed in `(a, b)`. (The earlier variant compared against the fixed anchor
`f(a)` instead, which failed to bracket a minimum whose right tail stays below
`f(a)`.) The [`Quadratic`](@ref) caller guards the fitted curvature (`p_2 ≤ 0`
falls back to a bisection step), so `f(b) > f(a)` is no longer required.

Returns the bracket *together with the function values at its endpoints*,
`(a, b, f(a), f(b))` with `a < b`.  The values are already computed during
bracketing, so the caller (the [`Quadratic`](@ref) line search) does not have
to re-evaluate `f` at the endpoints.

Returns `nothing` if no bracket is found within `nmax` steps — a line search must be able to
report an unbracketable merit rather than abort the enclosing solve.

`αmax` bounds how far to the right the search may probe; see [`linesearch_αmax`](@ref) and
[`_bracket_minimum_with_fixed_point_core`](@ref), which is what tells a truncated bracket from a
genuine one.
"""
function bracket_minimum_with_fixed_point(f::Callable, x::T, s::T, k::T=T(DEFAULT_BRACKETING_k), nmax::Integer=DEFAULT_BRACKETING_nmax, αmax::T=T(Inf)) where {T<:Number}
    a, b, ya, yb, _, status = _bracket_minimum_with_fixed_point_core(f, x, s, k, nmax, αmax)
    status === :unbracketable ? nothing : (a, b, ya, yb)
end

"""
    _bracket_minimum_with_fixed_point_core(f, x, s, k, nmax, αmax)

[`bracket_minimum_with_fixed_point`](@ref) with the number of evaluations of `f` it spent and the
bracket's `status` appended — `:ok`, `:capped` or `:unbracketable`, as [`_bracket_core`](@ref)
reports them. Private; the split exists so that
[`Quadratic`](@ref) can tell a bracket that ends at the ceiling from one that found a turning
point, and hand back the ceiling instead of fitting a polynomial to an interval over which the
merit only falls.
"""
function _bracket_minimum_with_fixed_point_core(f::Callable, x::T, s::T, k::T, nmax::Integer, αmax::T) where {T<:Number}
    a = x
    # Bounded like the loop's probes below, and for the reason given in `_bracket_core`: a ceiling
    # smaller than `s` must not be stepped over before the bound is first tested.
    b = s > zero(T) ? min(a + s, αmax) : a + s

    ya = f(a)
    yb = f(b)
    n = 2

    # flip a & b if necessary
    if yb > ya
        a, b = b, a
        ya, yb = yb, ya
        s = -s
    end

    bc = BracketMinimumCriterion()

    # Track the previous point so we can stop at the *turning point* (where `f`
    # stops decreasing) rather than only when `f` climbs back above the fixed
    # anchor `f(a)`.  A minimum whose right tail stays below `f(a)` (e.g. a merit
    # that dips and then only asymptotes back up) would otherwise never satisfy
    # `f(b) ≥ f(a)` and the routine would exhaust `nmax` and error, even though a
    # minimum was plainly bracketed.  This makes the fixed-point bracketer detect
    # the minimum just like the moving-anchor `bracket_minimum`.
    ybprev = yb
    for _ in 1:nmax
        bnext = b + s
        # As in `_bracket_core`: the ceiling bounds a step length, so it binds only while the
        # search runs rightward, and the ceiling itself is probed rather than skipped over.
        if s > zero(T) && bnext ≥ αmax
            αmax > a || return (a, a, ya, ya, n, :capped)
            # `b` is already at the ceiling whenever the first probe was clamped to it, and then
            # `yb` (which equals `ybprev` here — every iteration ends by copying one into the
            # other) *is* the merit at `αmax`. Re-evaluating it would tie with itself, and a tie
            # reads as a turning point; see the same guard in `_bracket_core`.
            αmax > b || return (a, b, ya, yb, n, :capped)
            b = αmax
            yb = f(b)
            n += 1
            return (a, b, ya, yb, n, bc(ybprev, yb) ? :ok : :capped)
        end
        b = bnext
        yb = f(b)
        n += 1
        if bc(ybprev, yb)
            # return the endpoints (sorted) along with their function values
            return a < b ? (a, b, ya, yb, n, :ok) : (b, a, yb, ya, n, :ok)
        end
        ybprev = yb
        s *= k
    end

    (a, b, ya, yb, n, :unbracketable)
end

function bracket_minimum_with_fixed_point(f::Callable, x::T; s::T=T(DEFAULT_BRACKETING_s), k::T=T(DEFAULT_BRACKETING_k), nmax::Integer=DEFAULT_BRACKETING_nmax, αmax::T=T(Inf)) where {T<:Number}
    bracket_minimum_with_fixed_point(f, x, s, k, nmax, αmax)
end

function bracket_minimum_with_fixed_point(prob::LinesearchProblem{T}, params, x::T, s::T, k::T=T(DEFAULT_BRACKETING_k), nmax::Integer=DEFAULT_BRACKETING_nmax, αmax::T=T(Inf)) where {T<:Number}
    bracket_minimum_with_fixed_point(x -> value(prob, x, params), x, s, k, nmax, αmax)
end

function bracket_minimum_with_fixed_point(prob::LinesearchProblem{T}, params, x::T; s::T=T(DEFAULT_BRACKETING_s), k::T=T(DEFAULT_BRACKETING_k), nmax::Integer=DEFAULT_BRACKETING_nmax, αmax::T=T(Inf)) where {T<:Number}
    bracket_minimum_with_fixed_point(prob, params, x, s, k, nmax, αmax)
end

_bracket_minimum_with_fixed_point_core(prob::LinesearchProblem{T}, params, x::T, s::T, k::T=T(DEFAULT_BRACKETING_k), nmax::Integer=DEFAULT_BRACKETING_nmax, αmax::T=T(Inf)) where {T<:Number} =
    _bracket_minimum_with_fixed_point_core(x -> value(prob, x, params), x, s, k, nmax, αmax)

"""
    bracket_root(f, x)

Make a bracket for the function based on `x` (for root finding).

This is largely equivalent to [`bracket_minimum`](@ref). See the end of that docstring for more information.

!!! info
    Here we use [`BracketRootCriterion`](@ref) instead of [`BracketMinimumCriterion`](@ref).
"""
function bracket_root(f::Callable, x::T; s::T=T(DEFAULT_BRACKETING_s), k::T=T(DEFAULT_BRACKETING_k), nmax::Integer=DEFAULT_BRACKETING_nmax)::Tuple{T,T} where {T<:Number}
    bracket(f, x, BracketRootCriterion(), s, k, nmax)
end

function bracket_root(prob::LinesearchProblem{T}, params, x::T; kwargs...) where {T<:Number}
    bracket_root(β -> value(prob, β, params), x; kwargs...)
end
