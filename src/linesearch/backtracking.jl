using Printf
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
    const BACKTRACKING_GROW_MIN

Lower bound on the factor by which the expansion phase of [`Backtracking`](@ref) lengthens an
accepted step: unless the model minimiser lies at least
``\mathrm{BACKTRACKING\_GROW\_MIN}\cdot\alpha``, the trial step is kept and no further merit
evaluation is spent (see [`backtracking_extrapolation`](@ref)).
Its value is """ * """$(BACKTRACKING_GROW_MIN).

This is the counterpart of [`BACKTRACKING_SHRINK_MIN`](@ref) on the growing side, and it is what
makes the expansion phase free for a well-scaled direction: a Newton or BFGS step is already at
its model minimum, so the test fails and the search returns at once.
"""
const BACKTRACKING_GROW_MIN = 2.0

@doc raw"""
    const DEFAULT_BACKTRACKING_q

The default *upper* bound on the factor by which the expansion phase of [`Backtracking`](@ref)
lengthens the step in one round, i.e. the counterpart of ``p`` on the growing side.
Its value is """ * """$(DEFAULT_BACKTRACKING_q).
"""
const DEFAULT_BACKTRACKING_q = 10.0

"""
    const DEFAULT_BACKTRACKING_NEXPAND

The default cap on the number of expansion trials of [`Backtracking`](@ref); each one costs a
merit evaluation. Its value is $(DEFAULT_BACKTRACKING_NEXPAND).
"""
const DEFAULT_BACKTRACKING_NEXPAND = 3

@doc raw"""
    Backtracking <: LinesearchMethod

# Keys

The trial step ``\alpha`` is *not* a key: it is the argument of [`solve`](@ref), which is its only
source. (A `Backtracking` used to carry an `α₀` field for it, which the algorithm never read — see
issue #174.)

The keys are:
- `c₁`=""" * string(DEFAULT_WOLFE_c₁) * raw""": the constant ``c_1`` in the [`SufficientDecreaseCondition`](@ref) (Armijo condition). Also see [`DEFAULT_WOLFE_c₁`](@ref).
- `c₂`=""" * string(DEFAULT_WOLFE_c₂) * raw""": the constant on whose basis the [`CurvatureCondition`](@ref) is tested. We should have ``c_2\in(c_1, 1).`` The closer this constant is to 1, the easier it is to satisfy the [`CurvatureCondition`](@ref).
- `p`=""" * string(DEFAULT_ARMIJO_p) * raw""": an *upper bound* on the factor by which ``\alpha`` is decreased in every step until the stopping criterion is satisfied. The actual factor is chosen by interpolation and confined to ``[`` [`BACKTRACKING_SHRINK_MIN`](@ref) ``\cdot\alpha, p\alpha]``, so the trial sequence is never longer than the plain ``\alpha \gets p\alpha`` ladder.
- `expand`=`false`: whether the search may *lengthen* the trial step (the expansion phase described below). Off by default, so a `Backtracking` is the classical one-sided algorithm unless it is asked for. Setting it requires the merit to be *evaluable* — finite or not, but not throwing — out to ``q^{\mathrm{nexpand}}\alpha``, since that is the largest step the phase can try.
- `q`=""" * string(DEFAULT_BACKTRACKING_q) * raw""": an *upper bound* on the factor by which ``\alpha`` is increased in one expansion round — the counterpart of ``p``. Only used when `expand` is set. See [`DEFAULT_BACKTRACKING_q`](@ref).
- `nexpand`=""" * string(DEFAULT_BACKTRACKING_NEXPAND) * raw""": the cap on the number of expansion trials, each of which costs one merit evaluation. It bounds the phase from within the `linesearch_max_iterations` of [`Options`](@ref) rather than beside it: whichever of the two is smaller applies, so the whole search still spends at most `linesearch_max_iterations` merit evaluations. Only used when `expand` is set. See [`DEFAULT_BACKTRACKING_NEXPAND`](@ref).
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
[`curvature_diagnostic`](@ref)). With `expand` set it *is* satisfied more often, because the
expansion phase moves the accepted step toward the line minimiser, so the diagnostic fires less.

## The expansion phase

With `expand = true` and the *first* trial step accepted, the search may also lengthen it, by
[`backtracking_extrapolation`](@ref), while each longer trial still satisfies the
[`SufficientDecreaseCondition`](@ref) *and* strictly improves the merit; at most `nexpand` such
trials are made and the best step seen is returned. A shrunken step is never expanded again:
once the ladder has backtracked, the longer steps are already known to fail.

This is the one place where the search leaves the interval ``[0, \alpha]`` the caller offered.
The largest step it can try is ``q^{\mathrm{nexpand}}\alpha`` — a thousand times the trial step
on the defaults — and a trial whose merit is not finite is simply rejected, at the cost of the
one evaluation. A merit that *throws* outside its domain is the caller's to guard, and it is one
reason the phase is opt-in.

The trials it spends come out of the `linesearch_max_iterations` budget of [`Options`](@ref),
not out of a second budget beside it, so termination case 4 above still bounds the whole search:
the phase makes at most `nexpand` trials *and* at most as many as that budget has left.

This is what makes the search two-sided. A shrink-only search returns the trial step it was
given whenever that step is acceptable, so on a direction whose natural scale is *larger* than
the trial step it pins ``\alpha`` at that ceiling on every iteration and the outer solve crawls —
by two orders of magnitude in the DFP case of issue #174, where the direction wanted
``\alpha \approx 11`` throughout.

The phase costs nothing where it can gain nothing. The model it extrapolates from is the same
quadratic that [`backtracking_interpolation`](@ref) uses on the way down, built from
``\varphi(0)``, ``\varphi'(0)`` and the merit at the trial step — all three already known — so
the decision whether to expand at all is free, and a direction that is already scaled like a
Newton step (which is at its model minimum at ``\alpha = 1``) fails the test and returns without
a further merit evaluation. That matters because for the merit of a [`NonlinearSolver`](@ref) an
evaluation is a full residual evaluation, the most expensive single operation of a solver step,
which is also why the phase does not test the [`CurvatureCondition`](@ref) to decide when to stop
growing: that would cost a full [`Jacobian`](@ref) per trial. Use [`StrongWolfe`](@ref), which
brackets on the derivative, where curvature control is genuinely required.

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
    c₁::T
    c₂::T
    p::T
    q::T
    τ_ulps::T
    expand::Bool
    nexpand::Int

    function Backtracking{T}(c₁::T, c₂::T, p::T, q::T, τ_ulps::T=armijo_ulps(T, c₁),
        expand::Bool=false, nexpand::Int=DEFAULT_BACKTRACKING_NEXPAND) where {T}
        @assert 0 < p < 1 "The shrinking parameter needs to satisfy 0 < p < 1, it is $(p)."
        @assert q > 1 "The expansion parameter needs to satisfy q > 1, it is $(q)."
        @assert 0 < c₁ < c₂ < 1 "The Wolfe constants need to satisfy 0 < c₁ < c₂ < 1, they are c₁ = $(c₁), c₂ = $(c₂)."
        @assert τ_ulps ≥ 0 "The round-off resolution needs to be nonnegative, it is $(τ_ulps) ulps."
        # `≥ 1` rather than `≥ 0`: `expand` is the one and only way to switch the phase off, so
        # `nexpand = 0` would be a second, silent encoding of "disabled" that `show` and
        # `isapprox` would then have to agree about.
        @assert nexpand ≥ 1 "The expansion budget needs to be at least one trial, it is $(nexpand). Pass expand = false to switch the phase off."
        # Capped here rather than only in the keyword constructor, so that *every* path into a
        # `Backtracking{T}` — including `change_precision`, which converts a method built for a
        # different `T` — gets a resolution the element type can support. See `armijo_ulps`.
        new{T}(c₁, c₂, p, q, min(τ_ulps, armijo_ulps(T, c₁)), expand, nexpand)
    end
end

function Backtracking(::Type{T}=Float64;
    c₁=T(DEFAULT_WOLFE_c₁),
    c₂=T(DEFAULT_WOLFE_c₂),
    p=T(DEFAULT_ARMIJO_p),
    q=T(DEFAULT_BACKTRACKING_q),
    τ_ulps=armijo_ulps(T, c₁),
    expand=false,
    nexpand=DEFAULT_BACKTRACKING_NEXPAND
) where {T}
    Backtracking{T}(c₁, c₂, p, q, τ_ulps, expand, nexpand)
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

@doc raw"""
    backtracking_extrapolation(φ₀, d₀, α, φα, q)

The next trial step of the expansion phase of [`Backtracking`](@ref), or `α` itself to say that
the step should not be lengthened.

`α`/`φα` is the step that was just *accepted*. The model is the same quadratic through
``\varphi(0)``, ``\varphi'(0)`` and ``\varphi(\alpha)`` that [`backtracking_interpolation`](@ref)
uses on the first backtrack, and its minimiser is

```math
\alpha^\star = \frac{-\varphi'(0)\,\alpha^2}{2\big(\varphi(\alpha) - \varphi(0) - \varphi'(0)\alpha\big)} ,
```

clamped from above to ``q\alpha``. A denominator that is negative or zero means the model is not
convex — the merit fell at least as fast as its tangent, so it is still dropping steeply — and
the step grows by the full factor ``q``. A denominator that is not *finite* is a different thing
entirely, namely no model at all, and returns `α`.

Everything the model needs has already been evaluated, so the decision costs no merit
evaluation. That is what the lower bound [`BACKTRACKING_GROW_MIN`](@ref) is for: unless the step
the search would actually try is at least that multiple of ``\alpha``, `α` is returned unchanged
and the search stops without spending one. The test is on the *clamped* step rather than on
``\alpha^\star`` itself, which matters only for ``q <`` [`BACKTRACKING_GROW_MIN`](@ref): there
the clamp, not the model, is what decides, and a convex model must not be allowed to buy a
growth by ``q`` that a non-convex one is refused. A direction scaled like a Newton step has
``\alpha^\star \approx \alpha`` at ``\alpha = 1`` and therefore pays nothing at all, and a merit
sitting at its round-off floor (``\varphi(\alpha) \approx \varphi(0)``) gives
``\alpha^\star \approx \alpha/2``, so the model declines to expand into rounding noise without
needing a special case for it.
"""
function backtracking_extrapolation(φ₀::T, d₀::T, α::T, φα::T, q::T) where {T}
    den = 2 * (φα - φ₀ - d₀ * α)
    # A non-finite denominator is *not* the non-convex case: it means there is no model at all,
    # so it must not fall through to the `q * α` branch, which would grow the step on the
    # strength of a `NaN`. (`backtracking_interpolation` guards its model the same way.)
    isfinite(den) || return α
    αₙ = den > zero(T) ? -d₀ * α^2 / den : q * α
    isfinite(αₙ) || return α
    # The gate is applied to the *clamped* step — the one that would actually be tried — rather
    # than to the raw minimiser. For `q ≥ BACKTRACKING_GROW_MIN`, which every default is, the two
    # agree: the clamp cannot pull a step that passed the gate back below it. For a smaller `q`
    # they do not, and gating the raw minimiser would then let a convex model spend a merit
    # evaluation to grow by only `q`, while a non-convex one — which asks for exactly `q * α` —
    # was refused that same growth outright.
    αₙ = min(αₙ, q * α)
    αₙ ≥ T(BACKTRACKING_GROW_MIN) * α || return α
    αₙ
end

# The expansion phase of `solve_with_status` below: lengthen the accepted step `α` while
# `backtracking_extrapolation` asks for it and each longer trial both satisfies the sufficient
# decrease condition and strictly improves the merit, and return the best pair seen together with
# the updated evaluation count. A rejected expansion therefore costs exactly one merit evaluation
# and nothing else — the previous best is still what is returned.
#
# `n` is passed and returned rather than captured: a counter captured by this function and mutated
# by its caller would be boxed, which makes the `trials` of the status built from it inferred-`Any`
# (the same reason `_wolfe_zoom`'s caller passes its `n`).
function backtracking_expand(f, sdc::SufficientDecreaseCondition{T}, φ₀::T, d₀::T, α::T, φα::T, n::Int, q::T, nexpand::Int) where {T}
    for _ in 1:nexpand
        αₙ = backtracking_extrapolation(φ₀, d₀, α, φα, q)
        αₙ > α || break
        φₙ = f(αₙ)
        n += 1
        (isfinite(φₙ) && φₙ < φα && sdc(αₙ, φₙ)) || break
        α, φα = αₙ, φₙ
    end
    (α, φα, n)
end

"""
    solve_with_status(ls::Linesearch{T,<:Backtracking}, α, params)

Run the backtracking line search from the trial step `α` and return the
[`LinesearchStatus`](@ref), emitting no messages. [`solve`](@ref) is this plus the report; see
[`Backtracking`](@ref).
"""
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

    for i in 1:config(ls).linesearch_max_iterations
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
            # The step may also be too *short*: a shrink-only search hands back whatever trial
            # step it was given as soon as that step is acceptable, which pins α at the caller's
            # ceiling on a direction whose natural scale is larger (issue #174). Only the first
            # trial is expanded — once the ladder has backtracked, the longer steps below have
            # already been rejected — and only when asked, `expand` being off by default.
            # The expansion draws on the *same* `linesearch_max_iterations` budget as the ladder,
            # so `trials` keeps meaning "merit evaluations this line search spent" and the bound
            # `Options` documents keeps holding. The cap binds only for a budget of `nexpand + 1`
            # or less; at the default of 60 against 3 it never does.
            if m.expand && i == 1
                budget = min(m.nexpand, config(ls).linesearch_max_iterations - n)
                α, φₐ, n = backtracking_expand(f, sdc, φ₀, d₀, α, φₐ, n, m.q, budget)
            end
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

# Behind a barrier because `curvature_diagnostic` below is specialized on the `Linesearch`, hence on
# its problem's closure types — see `report_linesearch_status`.
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

Base.show(io::IO, ls::Backtracking) = print(io, "Backtracking with c₁ = $(ls.c₁), c₂ = $(ls.c₂), p = $(ls.p) and τ_ulps = $(ls.τ_ulps)$(ls.expand ? ", expanding by at most q = $(ls.q) in at most $(ls.nexpand) trial(s)" : ", shrinking only").")

function change_precision(::Type{T}, method::Backtracking) where {T}
    T ≠ eltype(method) || return method
    Backtracking{T}(T(method.c₁), T(method.c₂), T(method.p), T(method.q), T(method.τ_ulps), method.expand, method.nexpand)
end

function Base.isapprox(bt₁::Backtracking{T}, bt₂::Backtracking{T}; kwargs...) where {T}
    # `expand` and `nexpand` are compared exactly: they select *which* algorithm runs and how many
    # times, and neither is a quantity an approximate comparison means anything for.
    bt₁.expand == bt₂.expand && bt₁.nexpand == bt₂.nexpand &&
        isapprox(bt₁.c₁, bt₂.c₁; kwargs...) && isapprox(bt₁.c₂, bt₂.c₂; kwargs...) && isapprox(bt₁.p, bt₂.p; kwargs...) && isapprox(bt₁.q, bt₂.q; kwargs...) && isapprox(bt₁.τ_ulps, bt₂.τ_ulps; kwargs...)
end
