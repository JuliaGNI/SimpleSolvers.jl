"""
    const DEFAULT_BRACKETING_s

Gives the default width of the interval (the bracket). See [`bracket_minimum`](@ref).
"""
const DEFAULT_BRACKETING_s = 1E-2

"""
    const DEFAULT_BRACKETING_k

Gives the default ratio by which the bracket is increased if bracketing was not successful. See [`bracket_minimum`](@ref).
"""
const DEFAULT_BRACKETING_k = 2.0

"Default constant"
const DEFAULT_BRACKETING_nmax=100

abstract type BracketingCriterion end
"""
    BracketMinimumCriterion <: BracketingCriterion

The criterion used for [`bracket_minimum`](@ref).

# Functor

```julia
bc(yb, yc)
```
This checks whether `yc` is bigger than `yb`, i.e. whether `c` is *past the minimum*.
"""
struct BracketMinimumCriterion <: BracketingCriterion end
struct BracketRootCriterion <: BracketingCriterion end
(::BracketMinimumCriterion)(yb::T, yc::T) where {T<:Number} = yc ≥ yb
(::BracketRootCriterion)(yb::T, yc::T) where {T<:Number} = yc * yb ≤ zero(T)

function bracket(f::Callable, x::T, bc::BracketingCriterion; s::T=T(DEFAULT_BRACKETING_s), k::T=T(DEFAULT_BRACKETING_k), nmax::Integer=DEFAULT_BRACKETING_nmax)::Tuple{T, T} where {T <: Number}    
    a = x
    ya = f(a)

    b = a + s
    yb = f(b)

    # check if condition is already satisfied
    if bc(f(a - s), yb)
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
        ya = yb
        b = c
        yb = yc
        s *= k
    end
    error("Unable to bracket f starting at x = $x.")
end

@doc raw"""
    bracket_minimum(f, x)

Move a bracket successively in the search direction (starting at `x`) and increase its size until a local minimum of `f` is found. 
This is used for performing [`Bisection`](@ref)s when only one `x` is given (and not an entire interval). 
This bracketing algorithm is taken from [kochenderfer2019algorithms](@cite). Also compare it to [`bracket_minimum_with_fixed_point`](@ref).

# Keyword arguments

- `s::`[`DEFAULT_BRACKETING_s`](@ref)
- `k::`[`DEFAULT_BRACKETING_k`](@ref)
- `nmax::`[`DEFAULT_BRACKETING_nmax`](@ref)

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

and then checks whether ``f(c) > f(b)``. If this is true it returns ``(a, c)`` or ``(c, a)``, depending on whether ``a<c`` or ``c<a`` respectively.
If this is not satisfied ``a,`` ``b`` and ``s`` are updated:
```math
\begin{aligned}
a \gets & b, \\
b \gets & c, \\
s \gets & sk, 
\end{aligned}
```
and the algorithm is continued. If we have not found a sign chance after ``n_\mathrm{max}`` iterations (see [`DEFAULT_BRACKETING_nmax`](@ref)) the algorithm is terminated and returns an error.
The interval that is returned by `bracket_minimum` is then typically used as a starting point for [`bisection`](@ref).

!!! info
    The function `bracket_root` is equivalent to `bracket_minimum` with the only difference that the criterion we check for is:
    ```math
    f(c)f(b) < 0,
    ```
    i.e. that a sign change in the function occurs.

See [`bracket_root`](@ref).
"""
function bracket_minimum(f::Callable, x::T=0.0; s::T=T(DEFAULT_BRACKETING_s), k::T=T(DEFAULT_BRACKETING_k), nmax::Integer=DEFAULT_BRACKETING_nmax) where {T <: Number}
    
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

    bracket(f, a, BracketMinimumCriterion(); s=s, k=k, nmax=nmax)
end

function bracket_minimum(obj::AbstractOptimizerProblem{T}, x::T=zero(T); kwargs...) where {T <: Number}
    bracket_minimum(obj.F, x; kwargs...)
end

@doc raw"""
    bracket_minimum_with_fixed_point(f, x)

Find a bracket while keeping the left side (i.e. `x`) fixed. 
The algorithm is similar to [`bracket_minimum`](@ref) (also based on [`DEFAULT_BRACKETING_s`](@ref) and [`DEFAULT_BRACKETING_k`](@ref)) with the difference that for the latter the left side is also moving.

The function `bracket_minimum_with_fixed_point` is used as a starting point for [`Quadratic`](@ref) (taken from [kelley1995iterative](@cite)), as the minimum of the polynomial approximation is:
```math
p_2 = \frac{f(b) - f(a) - f'(0)b}{b^2},
```
where ``b = \mathtt{bracket\_minimum\_with\_fixed\_point}(a)``. We check that ``f(b) > f(a)`` in order to ensure that the curvature of the polynomial (i.e. ``p_2`` is positive) and we have a minimum.
"""
function bracket_minimum_with_fixed_point(f::Callable, d::Callable, x::T=0.0; s::T=T(DEFAULT_BRACKETING_s), k::T=T(DEFAULT_BRACKETING_k), nmax::Integer=DEFAULT_BRACKETING_nmax) where {T <: Number}
    
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

    da = d(a)

    bc = BracketRootCriterion()

    # check if condition is already satisfied
    if bc(da, d(b))
        return (a, b)
    end

    for _ in 1:nmax
        b = b + s
        yb = f(b)
        if bc(da, d(b))
            interval = a < b ? (a, b) : (b, a)
            return interval
        end
        s *= k
    end
    error("Unable to bracket f starting at x = $x.")
end

function bracket_minimum_with_fixed_point(obj::AbstractOptimizerProblem{T}, x::T=zero(T); kwargs...) where {T <: Number}
    bracket_minimum_with_fixed_point(obj.F, obj.D, x; kwargs...)
end

"""
    bracket_root(f, x)

Make a bracket for the function based on `x` (for root finding).

This is largely equivalent to [`bracket_minimum`](@ref). See the end of that docstring for more information.
"""
function bracket_root(f::Callable, x::T=0.0; s::T=T(DEFAULT_BRACKETING_s), k::T=T(DEFAULT_BRACKETING_k), nmax::Integer=DEFAULT_BRACKETING_nmax)::Tuple{T, T} where {T <: Number}
    bracket(f, x, BracketRootCriterion(); s=s, k=k, nmax=nmax)
end
