"""
    const DEFAULT_WOLFE_αmax

Default upper bound on the step length for the bracketing phase of
[`StrongWolfe`](@ref). Its value is `65536.0`.
"""
const DEFAULT_WOLFE_αmax = 65536.0

@doc raw"""
    StrongWolfe{T} <: LinesearchMethod

A line search that finds a step ``\alpha`` satisfying the **strong Wolfe
conditions**

```math
\begin{aligned}
f(\alpha) &\leq f(0) + c_1\,\alpha\,f'(0), &\text{(sufficient decrease / Armijo)}\\
|f'(\alpha)| &\leq c_2\,|f'(0)|, &\text{(strong curvature)}
\end{aligned}
```

with ``0 < c_1 < c_2 < 1``.  It implements the bracketing line search of
[nocedal2006numerical; Alg. 3.5 and 3.6 (`zoom`)](@cite): a *bracketing* phase
grows the step until an interval containing an acceptable point is found, then a
*zoom* phase shrinks that interval (by bisection) until the strong Wolfe
conditions hold.

Unlike [`Backtracking`](@ref) — which enforces only sufficient decrease, since
the curvature condition cannot be honoured by shrinking alone — `StrongWolfe`
actually enforces the curvature condition, at the cost of evaluating the
derivative at each trial step.  Use it when curvature control is genuinely
required; [`Backtracking`](@ref) is cheaper otherwise.

# Keys

- `c₁` (default [`DEFAULT_WOLFE_c₁`](@ref)): the Armijo constant ``c_1``.
- `c₂` (default [`DEFAULT_WOLFE_c₂`](@ref)): the curvature constant ``c_2``. We require ``c_1 < c_2 < 1``.
- `αmax` (default [`DEFAULT_WOLFE_αmax`](@ref)): the largest step the bracketing phase will try.

!!! info
    The strong Wolfe conditions require a *descent direction* (``f'(0) < 0``).  If
    the line search problem is not decreasing at ``\alpha = 0`` the method cannot
    make progress; it then returns the caller's initial step and (at
    `verbosity ≥ 1`) warns.
"""
struct StrongWolfe{T} <: LinesearchMethod{T}
    c₁::T
    c₂::T
    αmax::T

    function StrongWolfe{T}(c₁::T, c₂::T, αmax::T) where {T}
        @assert 0 < c₁ < c₂ < 1 "The Wolfe constants need to satisfy 0 < c₁ < c₂ < 1, they are c₁ = $(c₁), c₂ = $(c₂)."
        @assert αmax > 0 "The maximum step length must be positive, it is $(αmax)."
        new{T}(c₁, c₂, αmax)
    end
end

function StrongWolfe(::Type{T}=Float64;
    c₁=T(DEFAULT_WOLFE_c₁),
    c₂=T(DEFAULT_WOLFE_c₂),
    αmax=T(DEFAULT_WOLFE_αmax)
) where {T}
    StrongWolfe{T}(c₁, c₂, αmax)
end

StrongWolfe(::Type{T}, ::SolverMethod) where {T} = StrongWolfe(T)

Base.show(io::IO, ls::StrongWolfe) = print(io, "StrongWolfe with c₁ = $(ls.c₁), c₂ = $(ls.c₂) and αmax = $(ls.αmax).")

function change_precision(::Type{T}, method::StrongWolfe) where {T}
    T ≠ eltype(method) || return method
    StrongWolfe{T}(T(method.c₁), T(method.c₂), T(method.αmax))
end

function Base.isapprox(w₁::StrongWolfe{T}, w₂::StrongWolfe{T}; kwargs...) where {T}
    isapprox(w₁.c₁, w₂.c₁; kwargs...) && isapprox(w₁.c₂, w₂.c₂; kwargs...) && isapprox(w₁.αmax, w₂.αmax; kwargs...)
end

# The `zoom` subroutine (Nocedal & Wright, Alg. 3.6).  On entry [αlo, αhi] brackets
# an acceptable step, with αlo the lower-φ endpoint that already satisfies the
# sufficient-decrease condition.  We interpolate by bisection (robust, no
# curvature assumptions) and maintain the bracket invariant.  On iteration
# exhaustion we return αlo, which by construction satisfies sufficient decrease —
# the method never returns a step worse than Armijo.  The two strong Wolfe
# conditions are checked through the shared `sdc`/`cc` condition objects.
function _wolfe_zoom(ls::Linesearch{T}, φ, dφ, sdc::SufficientDecreaseCondition{T}, cc::CurvatureCondition{T}, αlo::T, αhi::T, φlo::T) where {T}
    αj = αlo
    for _ in 1:config(ls).max_iterations
        αj = (αlo + αhi) / 2
        φj = φ(αj)
        if !sdc(αj) || φj ≥ φlo
            αhi = αj
        else
            cc(αj) && return αj
            dj = dφ(αj)
            if dj * (αhi - αlo) ≥ zero(T)
                αhi = αlo
            end
            αlo = αj
            φlo = φj
        end
        isapprox(αhi, αlo; atol=eps(T)) && break
    end
    αlo
end

"""
    solve(ls::Linesearch{T,<:StrongWolfe}, α, params)

Run the strong-Wolfe bracketing line search starting from the trial step `α`.
See [`StrongWolfe`](@ref).
"""
function solve(ls::Linesearch{T,<:StrongWolfe}, α::T, params=NullParameters()) where {T}
    m = method(ls)
    c₁ = m.c₁
    c₂ = m.c₂
    αmax = m.αmax
    prob = problem(ls)

    # One-slot memoisation of the merit φ and its derivative φ′: the bracketing /
    # zoom logic and the composed Wolfe conditions (`sdc`/`cc`) query the same
    # trial `α`, so caching the last evaluation avoids recomputing the (expensive)
    # merit and derivative.
    # `NaN` never equals a real query, so the first call at any `α` always evaluates.
    φα = Ref(T(NaN)); φv = Ref(T(NaN))
    dα = Ref(T(NaN)); dv = Ref(T(NaN))
    function φ(a)
        a == φα[] || (φα[] = a; φv[] = value(prob, a, params))
        φv[]
    end
    function dφ(a)
        a == dα[] || (dα[] = a; dv[] = derivative(prob, a, params))
        dv[]
    end

    φ0 = φ(zero(T))
    d0 = dφ(zero(T))

    # The strong Wolfe conditions are only meaningful for a descent direction.
    if d0 ≥ zero(T)
        config(ls).verbosity ≥ 1 && @warn "StrongWolfe: φ'(0) = $(d0) ≥ 0 is not a descent direction; returning α = $(α)."
        return α
    end

    # The strong Wolfe conditions are the Armijo (sufficient decrease) condition
    # plus the strong curvature condition.
    sdc = SufficientDecreaseCondition(c₁, φ0, d0, φ)
    cc = CurvatureCondition(c₂, d0, dφ, Val(:Strong))

    αprev = zero(T)
    φprev = φ0
    αi = clamp(α > zero(T) ? α : one(T), eps(T), αmax)
    αvalid = αi   # last trial step that satisfied sufficient decrease

    for i in 1:config(ls).max_iterations
        φi = φ(αi)
        if !sdc(αi) || (i > 1 && φi ≥ φprev)
            return _wolfe_zoom(ls, φ, dφ, sdc, cc, αprev, αi, φprev)
        end
        # αi satisfies sufficient decrease from here on
        cc(αi) && return αi
        di = dφ(αi)
        if di ≥ zero(T)
            return _wolfe_zoom(ls, φ, dφ, sdc, cc, αi, αprev, φi)
        end
        αvalid = αi
        # ascend toward αmax; stop expanding once the cap is reached
        αi == αmax && break
        αprev = αi
        φprev = φi
        αi = min(T(2) * αi, αmax)
    end

    # Return the last trial step that satisfied sufficient decrease — never the
    # freshly-doubled, unchecked `αi` from the expansion, and never the zero step.
    config(ls).verbosity ≥ 1 && @warn "StrongWolfe: no step satisfying the strong Wolfe conditions found within (0, $(αmax)]; returning the last sufficient-decrease step α = $(αvalid)."
    αvalid
end
