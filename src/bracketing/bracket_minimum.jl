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
    bracket(f, x, bc, s, k, nmax)

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
"""
function bracket(f::Callable, x::T, bc::BracketingCriterion, s::T=T(DEFAULT_BRACKETING_s), k::T=T(DEFAULT_BRACKETING_k), nmax::Integer=DEFAULT_BRACKETING_nmax) where {T<:Number}
    a = x

    b = a + s
    yb = f(b)

    if bc isa BracketRootCriterion && bc(f(a - s), yb)
        return (a - s, b)
    end

    for _ in 1:nmax
        c = b + s
        yc = f(c)
        if bc(yb, yc)
            interval = a < c ? (a, c) : (c, a)
            return interval
        end
        a = b
        b = c
        yb = yc
        s *= k
    end
    nothing
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
- `nmax`: by default [`DEFAULT_BRACKETING_nmax`](@ref).

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
function bracket_minimum(f::Callable, x::T, s::T, k::T=T(DEFAULT_BRACKETING_k), nmax::Integer=DEFAULT_BRACKETING_nmax) where {T<:Number}
    a = x
    ya = f(a)

    b = a + s
    yb = f(b)

    # flip a & b if necessary
    if yb > ya
        a, b = b, a
        ya, yb = yb, ya
        s = -s
    end

    bracket(f, a, BracketMinimumCriterion(), s, k, nmax)
end

function bracket_minimum(f::Callable, x::T; s::T=T(DEFAULT_BRACKETING_s), k::T=T(DEFAULT_BRACKETING_k), nmax::Integer=DEFAULT_BRACKETING_nmax) where {T<:Number}
    bracket_minimum(f, x, s, k, nmax)
end

function bracket_minimum(prob::LinesearchProblem{T}, params, x::T, s::T, k::T=T(DEFAULT_BRACKETING_k), nmax::Integer=DEFAULT_BRACKETING_nmax) where {T<:Number}
    bracket_minimum(x -> value(prob, x, params), x, s, k, nmax)
end

function bracket_minimum(prob::LinesearchProblem{T}, params, x::T; s::T=T(DEFAULT_BRACKETING_s), k::T=T(DEFAULT_BRACKETING_k), nmax::Integer=DEFAULT_BRACKETING_nmax) where {T<:Number}
    bracket_minimum(prob, params, x, s, k, nmax)
end

@doc raw"""
    bracket_minimum_with_fixed_point(f, x, s, k, nmax)

Find a bracket while keeping the left side (i.e. `x`) fixed.

The algorithm is similar to [`bracket_minimum`](@ref) (also based on [`DEFAULT_BRACKETING_s`](@ref) and [`DEFAULT_BRACKETING_k`](@ref)) with the difference that for the latter the left side is also moving.

The function `bracket_minimum_with_fixed_point` is used as a starting point for [`Quadratic`](@ref) (adapted from [kelley1995iterative](@cite)): that line search fits the polynomial centred at the bracket's left endpoint ``a``,
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
"""
function bracket_minimum_with_fixed_point(f::Callable, x::T, s::T, k::T=T(DEFAULT_BRACKETING_k), nmax::Integer=DEFAULT_BRACKETING_nmax) where {T<:Number}
    a = x
    b = a + s

    ya = f(a)
    yb = f(b)

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
        b = b + s
        yb = f(b)
        if bc(ybprev, yb)
            # return the endpoints (sorted) along with their function values
            return a < b ? (a, b, ya, yb) : (b, a, yb, ya)
        end
        ybprev = yb
        s *= k
    end

    nothing
end

function bracket_minimum_with_fixed_point(f::Callable, x::T; s::T=T(DEFAULT_BRACKETING_s), k::T=T(DEFAULT_BRACKETING_k), nmax::Integer=DEFAULT_BRACKETING_nmax) where {T<:Number}
    bracket_minimum_with_fixed_point(f, x, s, k, nmax)
end

function bracket_minimum_with_fixed_point(prob::LinesearchProblem{T}, params, x::T, s::T, k::T=T(DEFAULT_BRACKETING_k), nmax::Integer=DEFAULT_BRACKETING_nmax) where {T<:Number}
    bracket_minimum_with_fixed_point(x -> value(prob, x, params), x, s, k, nmax)
end

function bracket_minimum_with_fixed_point(prob::LinesearchProblem{T}, params, x::T; s::T=T(DEFAULT_BRACKETING_s), k::T=T(DEFAULT_BRACKETING_k), nmax::Integer=DEFAULT_BRACKETING_nmax) where {T<:Number}
    bracket_minimum_with_fixed_point(prob, params, x, s, k, nmax)
end

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
