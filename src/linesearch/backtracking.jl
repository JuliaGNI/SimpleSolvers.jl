using Printf
@doc raw"""
    const DEFAULT_ARMIJO_α₀

The default starting value for ``\alpha`` used in [`Backtracking`](@ref).
Its value is """ * """$(DEFAULT_ARMIJO_α₀).
"""
const DEFAULT_ARMIJO_α₀ = 1.0

"""
    const DEFAULT_ARMIJO_p

Constant used in [`Backtracking`](@ref).
Its value is $(DEFAULT_ARMIJO_p)

This is the default for the constant ``p`` by which `α` is decreased if the [`SufficientDecreaseCondition`](@ref) and the [`CurvatureCondition`](@ref) are not satisfied.
"""
const DEFAULT_ARMIJO_p = 0.5

@doc raw"""
    const DEFAULT_WOLFE_c₁

A constant ``c_1`` that is used in the [`SufficientDecreaseCondition`](@ref) (the
Armijo condition):

```math
f(\alpha) \leq f(\alpha_0) + c_1 \alpha f'(\alpha_0).
```
"""
const DEFAULT_WOLFE_c₁ = 1E-4

@doc raw"""
    const DEFAULT_WOLFE_c₂

The constant used in the second Wolfe condition (the [`CurvatureCondition`](@ref)). According to [nocedal2006numerical,kochenderfer2019algorithms](@cite) we should have
```math
c_2 \in (c_1, 1),
```
where ``c_1`` is the constant specified by [`DEFAULT_WOLFE_c₁`](@ref).

Furthermore [nocedal2006numerical](@cite) recommend ``c_2 = 0.9``; in [kochenderfer2019algorithms](@cite) the authors write: "it is common to set ``c_2=0.1`` when approximate line search is used with the conjugate gradient method and to 0.9 when used with Newton's method."
We use ``c_2 = 0.9`` as default.
"""
const DEFAULT_WOLFE_c₂ = 0.9

@doc raw"""
    const BACKTRACKING_SHRINK_MIN

Lower bound on the factor by which a rejected step is shrunk by the interpolation in
[`Backtracking`](@ref): the new trial step is confined to
``[\mathrm{BACKTRACKING\_SHRINK\_MIN}\cdot\alpha, p\alpha]`` (see
[nocedal2006numerical; §3.5](@cite), [dennis1996numerical; Alg. A6.3.1](@cite)).
Its value is """ * """$(BACKTRACKING_SHRINK_MIN).
"""
const BACKTRACKING_SHRINK_MIN = 0.1

@doc raw"""
    Backtracking <: LinesearchMethod

# Keys

The keys are:
- `α₀`=""" * string(DEFAULT_ARMIJO_α₀) * raw""": the initial step size ``\alpha``. This is decreased iteratively by a factor ``p`` until the [`SufficientDecreaseCondition`](@ref) is satisfied.
- `c₁`=""" * string(DEFAULT_WOLFE_c₁) * raw""": the constant ``c_1`` in the [`SufficientDecreaseCondition`](@ref) (Armijo condition). Also see [`DEFAULT_WOLFE_c₁`](@ref).
- `c₂`=""" * string(DEFAULT_WOLFE_c₂) * raw""": the constant on whose basis the [`CurvatureCondition`](@ref) is tested. We should have ``c_2\in(c_1, 1).`` The closer this constant is to 1, the easier it is to satisfy the [`CurvatureCondition`](@ref).
- `p`=""" * string(DEFAULT_ARMIJO_p) * raw""": an *upper bound* on the factor by which ``\alpha`` is decreased in every step until the stopping criterion is satisfied. The actual factor is chosen by interpolation and confined to ``[`` [`BACKTRACKING_SHRINK_MIN`](@ref) ``\cdot\alpha, p\alpha]``, so the trial sequence is never longer than the plain ``\alpha \gets p\alpha`` ladder.
- `τ_ulps`=[`armijo_ulps`](@ref)`(T, c₁)` (""" * string(DEFAULT_ARMIJO_τ_ULPS) * raw""" in `Float64` and `Float32`, less in `Float16`): the round-off resolution of the merit, in units in the last place of ``\varphi(0)``. It slackens the [`SufficientDecreaseCondition`](@ref) (never past ``\varphi(0)``), fixes ``\alpha_\mathrm{min}``, and separates a genuine decrease from one within the noise. A value larger than [`armijo_ulps`](@ref)`(T, c₁)` is capped to it, since above that ``\tau`` would swamp the decrease the condition demands. See [`DEFAULT_ARMIJO_τ_ULPS`](@ref).

# Implementation

The algorithm starts by setting
```math
\begin{aligned}
\varphi_0 &\gets \varphi(0),\\
d_0 &\gets \varphi'(0),
\end{aligned}
```
where ``\varphi`` is of type [`LinesearchProblem`](@ref). Unless ``\varphi_0`` and ``d_0`` are
finite with ``d_0 < 0`` the search is abandoned at once — no ``\alpha`` can satisfy the
[`SufficientDecreaseCondition`](@ref) along a direction that is not decreasing, so shrinking
``\alpha`` would only waste merit evaluations to find that out.

Otherwise it sets the round-off resolution ``\tau`` ([`armijo_tolerance`](@ref)) and the
smallest informative step ``\alpha_\mathrm{min}`` ([`backtracking_αmin`](@ref)), and shrinks
the trial step by [`backtracking_interpolation`](@ref) until one of the following happens:
1. the [`SufficientDecreaseCondition`](@ref) is satisfied — the step is accepted, and reported
   as a genuine decrease only if ``\varphi(\alpha) \leq \varphi_0 - \tau``;
2. two consecutive trials return ``\varphi(\alpha) = \varphi_0`` bit-exactly — the trial point
   no longer differs from the base point in floating point, so no smaller step can either;
3. ``\alpha \leq \alpha_\mathrm{min}`` — a smaller step could only be judged by rounding;
4. the `linesearch_max_iterations` budget of [`Options`](@ref) is spent.

Cases 2–4 are distinguished in the returned [`LinesearchStatus`](@ref): a merit that does not
vary by more than ``\tau`` has reached its round-off floor (`LINESEARCH_FLOOR`, benign and not
improvable by any line search), whereas one that does vary contradicts ``d_0 < 0``
(`LINESEARCH_EXHAUSTED`, a genuine inconsistency). See [`LinesearchOutcome`](@ref).

The [`CurvatureCondition`](@ref) is not used to terminate the iteration — it cannot be
honoured by shrinking alone — it is only checked afterwards to emit a warning (see
[`curvature_diagnostic`](@ref)).

# Extended help

[Sometimes](https://en.wikipedia.org/wiki/Backtracking_line_search) the parameters ``p`` and ``c_1`` have different names such as ``\tau`` and ``c``. Note that our ``\tau`` is something else entirely (the round-off resolution above).

``\alpha_\mathrm{min}`` is a factor ``2\,`` `τ_ulps` above the step ``\alpha^*`` at which the
condition degenerates into a test decided by rounding — *provided*
[`backtracking_αmin`](@ref)'s upper clamp at ``\sqrt{\mathrm{eps}(T)}`` is inactive, which it is
for a merit of ordinary steepness in double precision. Where the clamp binds (a very flat merit,
or any merit in `Float16`) the search does trial steps below ``\alpha^*``. That is deliberate and
harmless: the ``\min`` in the [`SufficientDecreaseCondition`](@ref) means the test there reduces
to ``\varphi(\alpha) \leq \varphi_0``, i.e. plain monotonicity, and such an accept is reported as
`LINESEARCH_FLOOR` rather than as a decrease.
"""
struct Backtracking{T} <: LinesearchMethod{T}
    α₀::T
    c₁::T
    c₂::T
    p::T
    τ_ulps::T

    function Backtracking{T}(α₀::T, c₁::T, c₂::T, p::T, τ_ulps::T=armijo_ulps(T, c₁)) where {T}
        @assert 0 < p < 1 "The shrinking parameter needs to satisfy 0 < p < 1, it is $(p)."
        @assert 0 < c₁ < c₂ < 1 "The Wolfe constants need to satisfy 0 < c₁ < c₂ < 1, they are c₁ = $(c₁), c₂ = $(c₂)."
        @assert τ_ulps ≥ 0 "The round-off resolution needs to be nonnegative, it is $(τ_ulps) ulps."
        # Capped here rather than only in the keyword constructor, so that *every* path into a
        # `Backtracking{T}` — including `change_precision`, which converts a method built for a
        # different `T` — gets a resolution the element type can support. See `armijo_ulps`.
        new{T}(α₀, c₁, c₂, p, min(τ_ulps, armijo_ulps(T, c₁)))
    end
end

function Backtracking(::Type{T}=Float64;
    α₀=T(DEFAULT_ARMIJO_α₀),
    c₁=T(DEFAULT_WOLFE_c₁),
    c₂=T(DEFAULT_WOLFE_c₂),
    p=T(DEFAULT_ARMIJO_p),
    τ_ulps=armijo_ulps(T, c₁)
) where {T}
    Backtracking{T}(α₀, c₁, c₂, p, τ_ulps)
end

Backtracking(::Type{T}, ::SolverMethod) where {T} = Backtracking(T)


@doc raw"""
    backtracking_αmin(c₁, d₀, τ)

The smallest step length for which the [`SufficientDecreaseCondition`](@ref) can still be
decided by the merit rather than by rounding:

```math
\alpha_\mathrm{min} = \frac{\tau}{c_1|\varphi'(0)|} .
```

Below ``\alpha_\mathrm{min}`` the demanded decrease ``c_1\alpha|\varphi'(0)|`` is smaller than
the round-off resolution ``\tau``, so a trial step carries no information. Writing
``\tau = n\cdot\mathrm{ulp}(\varphi(0))`` (see [`armijo_tolerance`](@ref)) gives

```math
\alpha_\mathrm{min} = 2n\,\alpha^*, \qquad
\alpha^* = \frac{\mathrm{ulp}(\varphi(0))}{2c_1|\varphi'(0)|},
```

where ``\alpha^*`` is the step below which ``\mathrm{fl}(\varphi(0) + c_1\alpha\varphi'(0))``
rounds back up to ``\varphi(0)`` and the condition degenerates to
``\varphi(\alpha) \leq \varphi(0)``.

The result is clamped to ``[\mathrm{eps}(T), \sqrt{\mathrm{eps}(T)}]``: the lower bound is the
historical negligible-step floor, and the upper bound makes sure that a nearly flat but
genuine merit (very small ``|\varphi'(0)|``) is still searched — an unclamped
``\alpha_\mathrm{min}`` grows without bound as ``|\varphi'(0)| \to 0`` and would stop the search
before it began.

!!! note "The factor ``2n`` holds only while the upper clamp is inactive"
    ``\alpha_\mathrm{min} = 2n\,\alpha^*`` puts the search a factor ``2n`` clear of the region
    where the condition is decided by rounding, but the ``\sqrt{\mathrm{eps}(T)}`` clamp can pull
    it *below* ``\alpha^*``: that happens for
    ``|\varphi'(0)| < \mathrm{ulp}(\varphi(0)) / (2c_1\sqrt{\mathrm{eps}(T)})``, i.e. below
    ``7\cdot10^{-5}`` for `Float64` with ``\varphi(0) = 1``, below ``1.7`` for `Float32`, and
    essentially always for `Float16`. Trial steps below ``\alpha^*`` are then taken, and that is
    safe rather than merely tolerated: the ``\min`` in the [`SufficientDecreaseCondition`](@ref)
    reduces the test there to ``\varphi(\alpha) \leq \varphi(0)``, so it can accept a
    non-increase but never an increase, and such an accept is classified `LINESEARCH_FLOOR`.
"""
function backtracking_αmin(c₁::T, d₀::T, τ::T) where {T}
    αmin = τ / (c₁ * abs(d₀))
    isfinite(αmin) || (αmin = sqrt(eps(T)))
    clamp(αmin, eps(T), sqrt(eps(T)))
end

@doc raw"""
    backtracking_interpolation(φ₀, d₀, α, φα, αp, φp, p)

The next trial step of the safeguarded polynomial backtracking used by [`Backtracking`](@ref)
(see [nocedal2006numerical; §3.5](@cite), [dennis1996numerical; Alg. A6.3.1](@cite)).

`α`/`φα` is the trial step that was just rejected and `αp`/`φp` the one rejected before it
(`αp` is `NaN` on the first backtrack). The model interpolates ``\varphi(0)``,
``\varphi'(0)`` and the rejected value(s) — a quadratic on the first backtrack, a cubic
afterwards — and its minimiser is clamped to
``[`` [`BACKTRACKING_SHRINK_MIN`](@ref) ``\cdot\alpha, p\alpha]``.

The clamp is what makes this safe: an unclamped interpolant can return ``\alpha`` itself (no
progress at all), collapse to numerically zero, or be meaningless because the merit values it
is built from are rounding noise. Because the upper bound is ``p``, the trial sequence is
pointwise never longer than the plain ``\alpha \gets p\alpha`` ladder.
"""
function backtracking_interpolation(φ₀::T, d₀::T, α::T, φα::T, αp::T, φp::T, p::T) where {T}
    αₙ = T(NaN)
    if isfinite(φα)
        if isnan(αp)
            # quadratic model through (0, φ₀), φ'(0) = d₀ and (α, φα)
            den = 2 * (φα - φ₀ - d₀ * α)
            den > zero(T) && (αₙ = -d₀ * α^2 / den)
        elseif isfinite(φp) && α ≠ αp
            # cubic model through (0, φ₀), φ'(0) = d₀, (αp, φp) and (α, φα)
            r₁ = φα - φ₀ - d₀ * α
            r₂ = φp - φ₀ - d₀ * αp
            den = α^2 * αp^2 * (α - αp)
            a = (αp^2 * r₁ - α^2 * r₂) / den
            b = (-αp^3 * r₁ + α^3 * r₂) / den
            disc = b^2 - 3 * a * d₀
            if disc ≥ zero(T)
                αₙ = iszero(a) ? -d₀ / (2 * b) : (-b + sqrt(disc)) / (3 * a)
            end
        end
    end
    (isfinite(αₙ) && αₙ > zero(T)) || (αₙ = p * α)
    clamp(αₙ, T(BACKTRACKING_SHRINK_MIN) * α, p * α)
end

"""
    solve(ls::Linesearch{T,<:Backtracking}, α, params)

Run the backtracking line search from the trial step `α`, report the outcome through
[`linesearch_warnings`](@ref) and return the accepted step length.

Use [`solve_with_status`](@ref) to obtain the [`LinesearchStatus`](@ref) instead: a caller
that has to tell "I found a decreasing step" from "the merit is at its round-off floor and
nothing can decrease it" cannot do so from the step length alone.
"""
function solve(ls::Linesearch{T,<:Backtracking}, α::T, params=NullParameters()) where {T}
    status = solve_with_status(ls, α, params)
    linesearch_warnings(status, ls, params)
    steplength(status)
end

function solve_with_status(ls::Linesearch{T,<:Backtracking}, α::T, params=NullParameters()) where {T}
    m = method(ls)
    f(a) = value(problem(ls), a, params)
    d(a) = derivative(problem(ls), a, params)

    # note that we anchor at α = 0 here as this is the base point of the linesearch problem.
    φ₀ = f(zero(T))
    d₀ = d(zero(T))

    # No α can satisfy the SufficientDecreaseCondition unless the merit is finite and actually
    # decreasing at the anchor, so report that instead of shrinking α fifty times to find out.
    anchor = check_anchor(φ₀, d₀, α)
    isnothing(anchor) || return anchor

    τ = armijo_tolerance(φ₀, m.τ_ulps)
    αmin = backtracking_αmin(m.c₁, d₀, τ)
    sdc = SufficientDecreaseCondition(m.c₁, φ₀, d₀, f; τ=τ)

    αₐ = α       # last trial step that was actually evaluated
    φₐ = φ₀      # the merit there
    αp = T(NaN)  # previous trial step and merit, for the cubic model
    φp = T(NaN)
    frozen = 0   # consecutive trials whose merit is bit-identical to φ(0)
    n = 0        # trial steps at which the merit was evaluated
    reachedαmin = false  # the ladder ran down to the αmin floor rather than out of budget

    for _ in 1:config(ls).linesearch_max_iterations
        αₐ = α
        φₐ = f(α)
        n += 1

        # Accept as soon as the (round-off tolerant) sufficient decrease condition holds.
        # Whether that is a *genuine* decrease or merely a tie at the merit's round-off floor
        # is decided by φₐ: only a decrease exceeding the resolution τ counts. The returned
        # step is the same either way, the reported outcome is not — which is precisely what
        # the former exact test could not express, since its accepts in the region below
        # α* = ulp(φ₀)/(2c₁|d₀|) were ties produced by rounding yet reported as successes.
        # The condition itself can never accept an increase (see `SufficientDecreaseCondition`),
        # so a `LINESEARCH_FLOOR` accept here is always a non-increasing step.
        if sdc(α, φₐ)
            oc = φₐ ≤ φ₀ - τ ? LINESEARCH_DECREASED : LINESEARCH_FLOOR
            return LinesearchStatus{T}(α, oc, n, φ₀, d₀, φₐ, τ, αmin)
        end

        # A merit that is bit-identical to φ(0) means the trial point no longer differs from
        # the base point in floating point (α below the x-scale eps‖x‖/‖δx‖), or that the
        # merit is flat to the last bit: every smaller α is frozen too, so shrinking cannot
        # help. Two consecutive frozen trials guard against an accidental tie. This is the
        # scalar surrogate for the x-scale minimum step — a line search only ever sees φ.
        φₐ == φ₀ ? (frozen += 1) : (frozen = 0)
        frozen ≥ 2 && return LinesearchStatus{T}(α, LINESEARCH_FLOOR, n, φ₀, d₀, φₐ, τ, αmin)

        if α ≤ αmin
            reachedαmin = true
            break
        end

        αₙ = backtracking_interpolation(φ₀, d₀, α, φₐ, αp, φp, m.p)
        αp, φp = α, φₐ
        α = max(αₙ, αmin)  # a trial exactly at αmin is always taken before giving up
    end

    # Nothing satisfied the sufficient decrease condition. Distinguish the two very different
    # reasons: if the ladder got all the way down to the smallest informative step and the merit
    # there differs from φ(0) by no more than the round-off resolution, the merit has reached its
    # round-off floor and *no* step can decrease it — the outer iteration is stuck no matter what
    # the line search returns. Otherwise the merit genuinely fails to decrease even at the
    # smallest informative step, which contradicts φ'(0) < 0.
    #
    # `reachedαmin` is tracked explicitly rather than inferred from `α > αmin`, which
    # misclassifies the one case where the budget runs out on the very iteration that sets
    # `α = αmin`: the ladder was then cut short, but the comparison says otherwise.
    oc = (reachedαmin && abs(φₐ - φ₀) ≤ τ) ? LINESEARCH_FLOOR : LINESEARCH_EXHAUSTED
    LinesearchStatus{T}(αₐ, oc, n, φ₀, d₀, φₐ, τ, αmin)
end

# The message is `@noinline` and takes nothing but a number, for the reason spelled out on
# `report_linesearch_status`: `curvature_diagnostic` is specialized on the `Linesearch`, hence on
# the closure types of its `LinesearchProblem`, and it is called from `linesearch_warnings` — so a
# message inlined into it is re-inferred and re-codegen'd once per solver, exactly what the
# barrier there exists to avoid.
@noinline function report_curvature_violation(α::Number)
    @warn "Backtracking line search: accepted step α = $(α) satisfies the sufficient decrease but not the curvature condition."
    nothing
end

# The curvature condition cannot be enforced by shrinking alone, so `Backtracking` only
# reports it — and only for a step that was genuinely accepted, because `derivative` costs a
# full Jacobian for the line search problem of a nonlinear solver.
function curvature_diagnostic(status::LinesearchStatus{T}, ls::Linesearch{T,<:Backtracking}, params) where {T}
    issufficient(status) || return nothing
    d(a) = derivative(problem(ls), a, params)
    cc = CurvatureCondition(method(ls).c₂, status.d₀, d, Val(:Standard))
    cc(steplength(status)) || report_curvature_violation(steplength(status))
    nothing
end

Base.show(io::IO, ls::Backtracking) = print(io, "Backtracking with α₀ = $(ls.α₀) c₁ = $(ls.c₁), c₂ = $(ls.c₂), p = $(ls.p) and τ_ulps = $(ls.τ_ulps).")

function change_precision(::Type{T}, method::Backtracking) where {T}
    T ≠ eltype(method) || return method
    Backtracking{T}(T(method.α₀), T(method.c₁), T(method.c₂), T(method.p), T(method.τ_ulps))
end

function Base.isapprox(bt₁::Backtracking{T}, bt₂::Backtracking{T}; kwargs...) where {T}
    isapprox(bt₁.α₀, bt₂.α₀; kwargs...) && isapprox(bt₁.c₁, bt₂.c₁; kwargs...) && isapprox(bt₁.c₂, bt₂.c₂; kwargs...) && isapprox(bt₁.p, bt₂.p; kwargs...) && isapprox(bt₁.τ_ulps, bt₂.τ_ulps; kwargs...)
end
