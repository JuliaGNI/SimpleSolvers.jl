"Default initial trust-region radius for the [`DogLegSolver`](@ref); the default of the `dogleg_initial_radius` field of [`Options`](@ref), which the solver actually reads."
const INITIAL_Δ = 1.0
"Default maximum trust-region radius (``\\hat\\Delta`` in [nocedal2006numerical; Alg. 4.1](@cite)) for the [`DogLegSolver`](@ref); the radius is never expanded beyond this. The default of the `dogleg_max_radius` field of [`Options`](@ref), which the solver actually reads."
const DOGLEG_Δ_MAX = 1E2
"Factor by which the trust-region radius is shrunk on a poor step (``\\rho < 1/4``)."
const DOGLEG_Δ_SHRINK = 0.25
"Factor by which the trust-region radius is expanded on a very good step (``\\rho > 3/4`` at the boundary)."
const DOGLEG_Δ_EXPAND = 2.0
"Lower ρ threshold below which the trust-region radius is shrunk."
const DOGLEG_ρ_LOW = 0.25
"Upper ρ threshold above which the trust-region radius may be expanded."
const DOGLEG_ρ_HIGH = 0.75
"Minimum ρ (actual/predicted reduction) for a step to be accepted (``\\eta`` in [nocedal2006numerical; Alg. 4.1](@cite))."
const DOGLEG_η = 1E-4

"""
    DogLeg(refactorize=1)

*Powell's dogleg method* [powell1970new](@cite).

Like [`Newton`](@ref), the `refactorize` parameter determines after how many
steps the [`Jacobian`](@ref) is re-evaluated and refactored (see [`factorize!`](@ref)).
The default `refactorize = 1` re-evaluates and refactorizes the Jacobian on every step;
`refactorize > 1` reuses the Jacobian (and its factorization) in between, giving a
quasi-Newton-style dogleg method.
"""
struct DogLeg <: NonlinearSolverMethod
    refactorize::Int

    DogLeg(refactorize::Integer=1) = new(refactorize)
end

"""
    DogLegSolver

The [`NonlinearSolver`](@ref) for the [`DogLeg`](@ref) method.
"""
const DogLegSolver{T} = NonlinearSolver{T,DogLeg}

@doc raw"""
    directions!(s, x, params, iteration=1)

Compute [`direction₁`](@ref) and [`direction₂`](@ref) for the [`DogLegSolver`](@ref).

This is equivalent to [`direction!`](@ref) for the [`NewtonSolver`](@ref).

# Examples

```jldoctest; setup = :(using SimpleSolvers; using SimpleSolvers: NullParameters, directions!, direction₁, direction₂, cache, l2norm)
julia> J = [0 1; -1 0];

julia> f(y, x, params) = y .= cos.(J * x .- 2.) .^ 2 / l2norm(sin.(x) .- 1.);

julia> x = zeros(2); y = similar(x); s = DogLegSolver(x, y; F = f);

julia> directions!(s, x, NullParameters());

julia> direction₁(cache(s))
2-element Vector{Float64}:
 -0.25513686072399455
  0.1601152321012896

julia> direction₂(cache(s))
2-element Vector{Float64}:
 -0.22882877718014286
  0.22882877718014288
```

# Extended help

The Gauss-Newton direction (i.e. [`direction₂`](@ref)) is computed the usual way:

```math
\mathbf{d}_2 = -\mathbf{J}^{-1} \mathbf{r}
```

where ``\mathbf{J}`` is the Jacobian matrix and ``\mathbf{r}`` is the residual vector. The steepest descent direction (taken from [nocedal2006numerical; Equation (11.46)](@cite)) is different:
```math
\mathbf{d}_1 = -\frac{||\mathbf{J}^T\mathbf{r}||^2}{\mathbf{r}^T(\mathbf{J}\mathbf{J}^T)(\mathbf{J}\mathbf{J}^T)\mathbf{r}}\mathbf{J}^T\mathbf{r}.
```

The [`DogLegSolver`](@ref) then interpolates between these two directions (this interpolation is piecewise linear).

As for the (quasi-)[`NewtonSolver`](@ref), the [`Jacobian`](@ref) is only re-evaluated and refactored every `refactorize` iterations (see [`DogLeg`](@ref)), and always on a fresh state or the first step (`iteration ≤ 1`), or when `force_refactorize = true` (used by [`solver_step!`](@ref) to recover from a collapsed trust-region radius). In between, the stale Jacobian and its factorization are reused for both directions. The default `refactorize = 1` refactorizes on every step.
"""
function directions!(s::DogLegSolver{T}, x::AbstractVector{T}, params, iteration=1; force_refactorize::Bool=false) where {T}
    # the Newton direction
    value!(rhs(linearproblem(s)), nonlinearproblem(s), x, params)
    rhs(linearproblem(s)) .*= -1
    if (force_refactorize || mod(iteration, method(s).refactorize) == 0 || iteration ≤ 1)
        jacobian!(s, x, params)
        matrix(linearproblem(s)) .= jacobianmatrix(s)
        idxs = diagind(matrix(linearproblem(s)))
        @view(matrix(linearproblem(s))[idxs]) .+= config(s).regularization_factor
        factorize!(linearsolver(s), linearproblem(s))
    end
    ldiv!(direction₂(cache(s)), linearsolver(s), rhs(linearproblem(s)))

    # the steepest descent direction
    # direction₁ ← Jᵀ·rhs = -JᵀF; fac₁ = ‖JᵀF‖²
    mul!(direction₁(cache(s)), transpose(jacobianmatrix(s)), rhs(linearproblem(s)))
    fac₁ = L2norm(direction₁(cache(s)))
    if fac₁ < eps(T)
        # ‖JᵀF‖² ≈ 0: the iterate is stationary (e.g. the exact root). The Cauchy
        # scaling below would divide by ‖J·JᵀF‖² = 0 and produce NaN, so we set
        # the steepest-descent direction to zero and let the convergence check
        # handle it.
        direction₁(cache(s)) .= zero(T)
    else
        mul!(cache(s).y₂, jacobianmatrix(s), direction₁(cache(s)))
        mul!(cache(s).y₃, transpose(jacobianmatrix(s)), cache(s).y₂)
        fac₂ = direction₁(cache(s)) ⋅ cache(s).y₃
        direction₁(cache(s)) .*= fac₁
        direction₁(cache(s)) ./= fac₂
    end

    direction₁(cache(s)), direction₂(cache(s))
end

"""
    dogleg_direction!(cache, Δ)

Compute the (piecewise-linear) dogleg step for trust-region radius `Δ` from the
steepest-descent direction [`direction₁`](@ref) and the Newton direction
[`direction₂`](@ref) (both already stored in `cache`), writing the result into
[`direction`](@ref)`(cache)`.

`direction₁` and `direction₂` do **not** depend on `Δ`, so this may be called
repeatedly while shrinking `Δ` without recomputing (and refactorizing) the
Jacobian.
"""
function dogleg_direction!(cache::DogLegCache{T}, Δ::T) where {T}
    # Each branch broadcasts straight into `direction(cache)` rather than building
    # a temporary and copying it in.
    if l2norm(direction₂(cache)) ≤ Δ
        direction(cache) .= direction₂(cache)
    elseif l2norm(direction₁(cache)) > Δ
        direction(cache) .= direction₁(cache) .* (Δ / l2norm(direction₁(cache)))
    else
        direction_difference(cache) .= direction₂(cache) .- direction₁(cache)
        d₁d₂d₁ = direction₁(cache) ⋅ direction_difference(cache)
        # expression under the square root (nonnegative on this branch, where
        # ‖direction₁‖ ≤ Δ, but clamped at zero to guard against rounding)
        eusr = d₁d₂d₁^2 - L2norm(direction_difference(cache)) * (L2norm(direction₁(cache)) - Δ^2)
        τ₂ = (-d₁d₂d₁ + √(max(eusr, zero(T)))) / L2norm(direction_difference(cache)) + 1
        # τ₂ should lie in [1, 2]; clamp it (with the interval closed) rather than
        # erroring out on a value that is slightly outside due to rounding.
        τ = clamp(τ₂, one(T), T(2))
        direction(cache) .= direction₁(cache) .+ (τ - 1) .* direction_difference(cache)
    end
    direction(cache)
end

function initialize!(s::DogLegSolver, x::AbstractVector)
    # The cache reset (`initialize!(::DogLegCache, …)`) restores the radius to the
    # constant default; override it with the configured `dogleg_initial_radius` so a
    # (re)used solver starts each solve from the caller's chosen radius.
    initialize!(cache(s), x)
    trust_radius!(cache(s), config(s).dogleg_initial_radius)
end

function solver_step!(x::AbstractVector{T}, s::DogLegSolver{T}, state::NonlinearSolverState{T}, params) where {T}
    # If the carried trust-region radius collapsed on the previous step, that step
    # made no progress (in quasi-Newton mode this happens when a *stale* Jacobian's
    # steepest-descent direction is not a descent direction for ‖F‖²).  Recover by
    # resetting the radius and forcing a fresh Jacobian this step: a freshly evaluated
    # steepest-descent direction is guaranteed to reduce the merit for a small enough
    # step, so the trust-region loop below can accept before the radius underflows
    # again.  Without this, `while Δ > eps(T)` would never run, the iterate would
    # freeze, and the solve would silently spin to `max_iterations`.
    Δ = trust_radius(cache(s))
    force_refresh = Δ ≤ eps(T)
    force_refresh && (Δ = config(s).dogleg_initial_radius)

    directions!(s, x, params, iteration_number(state); force_refactorize=force_refresh)
    any(isnan, direction₁(cache(s))) && throw(NonlinearSolverException("NaN detected in direction₁ vector"))
    any(isnan, direction₂(cache(s))) && throw(NonlinearSolverException("NaN detected in direction₂ vector"))

    # Trust-region step with a ρ-based radius update (Nocedal & Wright, Alg. 4.1).
    # The radius Δ is *carried across outer solver steps* in the cache (rather than
    # reset to a fixed value every step): a good step expands it, a poor step
    # shrinks it.  Because d₁ and d₂ do not depend on Δ, shrinking Δ on a rejected
    # step only recomputes the cheap dogleg interpolation via `dogleg_direction!` —
    # the Jacobian evaluation and factorization (`directions!`) run exactly once
    # per solver step.  Termination on the Δ floor is independent of `verbosity`.
    φ₀ = L2norm(value(state))          # current merit φ(x) = ‖F(x)‖²
    accepted = false
    while Δ > eps(T)
        dogleg_direction!(cache(s), Δ)
        d = direction(cache(s))
        pₙ = l2norm(d)

        # trial iterate x + d and its merit φ(x + d)
        compute_new_iterate!(solution(cache(s)), solution(state), one(T), d)
        value!(value(cache(s)), nonlinearproblem(s), solution(cache(s)), params)
        φ = L2norm(value(cache(s)))

        # An undefined merit (`F` evaluated outside its domain, e.g. `log`/`sqrt`
        # of a negative trial iterate) is treated exactly like a rejected step:
        # shrink the radius and retry with a shorter step along the *same* dogleg
        # path.  (Rescaling d₁ or d₂ themselves would
        # destroy the ‖d₁‖ ≤ ‖d₂‖ relation the dogleg interpolation assumes; a NaN
        # merit must also not reach the ρ update below, where every comparison
        # with NaN is false and the loop would spin forever at constant Δ.)
        if isnan(φ)
            verbosity(config(s)) ≥ 2 && @warn "DogLeg: undefined merit (NaN) at the trial step; shrinking the trust-region radius."
            Δ *= T(DOGLEG_Δ_SHRINK)
            continue
        end

        # A (numerically) zero step means the model predicts no further progress:
        # accept it and let the convergence check decide.  This is the exact-root /
        # stationary-point case where d₁ = d₂ = 0 (see [`directions!`](@ref)).
        if pₙ ≤ eps(T)
            x .= solution(cache(s))
            accepted = true
            break
        end

        # Predicted reduction from the Gauss-Newton model m(d) = ‖F + J·d‖²:
        #   pred = ‖F‖² − ‖F + J·d‖²   (cache.y₂ = J·d is reused as scratch).
        mul!(cache(s).y₂, jacobianmatrix(s), d)
        cache(s).y₂ .+= value(state)
        pred = φ₀ - L2norm(cache(s).y₂)
        ared = φ₀ - φ
        # ρ = actual / predicted reduction.  Guard the degenerate pred ≤ 0 case
        # (model predicts no decrease): accept only if the merit actually dropped.
        ρ = pred > eps(T) ? ared / pred : (ared > zero(T) ? one(T) : zero(T))

        # Radius update (before the accept test, so a shrink applies to the retry).
        if ρ < T(DOGLEG_ρ_LOW)
            Δ *= T(DOGLEG_Δ_SHRINK)
        elseif ρ > T(DOGLEG_ρ_HIGH) && isapprox(pₙ, Δ; rtol=sqrt(eps(T)))
            # very good step sitting on the trust-region boundary ⇒ grow the radius
            Δ = min(T(DOGLEG_Δ_EXPAND) * Δ, config(s).dogleg_max_radius)
        end

        if ρ > T(DOGLEG_η)
            x .= solution(cache(s))
            accepted = true
            break
        end
        # rejected (ρ ≤ η < 1/4 ⇒ Δ was just shrunk): retry with the smaller radius.
    end

    trust_radius!(cache(s), Δ)      # carry the updated radius to the next step

    if !accepted
        # The trust-region radius underflowed without an acceptable step.  Take the
        # last (smallest-Δ) trial — unless its merit is undefined (NaN), in which
        # case keep the current iterate: a stalled step, which the residual-gated
        # convergence test reports as non-converged.
        verbosity(config(s)) ≥ 1 && @warn "DogLeg trust-region radius Δ underflowed without an acceptable step (iterations: $(iteration_number(state)))."
        any(isnan, value(cache(s))) || (x .= solution(cache(s)))
    end

    x
end

function DogLegSolver(x::AT, nlp::NLST, ls::LST, linearsolver::LSoT, linesearch::LiSeT, cache::CT; jacobian::Jacobian=JacobianAutodiff(nlp.F, x), refactorize::Integer=1, options_kwargs...) where {T,AT<:AbstractVector{T},NLST,LST,LSoT,LiSeT,CT}
    config = Options(T; options_kwargs...)

    if refactorize > 1 && typeof(method(linesearch)) <: Static
        config.verbosity ≥ 1 && (@warn "Static line search will not work with refactorize = $(refactorize). Setting refactorize = 1.")
        refactorize = 1
    end

    NonlinearSolver(x, nlp, ls, linearsolver, linesearch, cache, config; method=DogLeg(refactorize), jacobian=jacobian)
end

function DogLegSolver(x::AbstractVector{T}, F::Callable, y::AbstractVector{T}; linear_solver_method=LU(), (DF!)=missing, linesearch=Backtracking(T), jacobian=missing, kwargs...) where {T}
    nlp = NonlinearProblem(F, DF!, x, y)
    # Build the default autodiff Jacobian lazily, so we don't allocate ForwardDiff
    # configs when either a Jacobian or a `DF!` is supplied.
    jacobian = ismissing(DF!) ?
               (ismissing(jacobian) ? JacobianAutodiff(F, x, y) : jacobian) :
               JacobianFunction{T}(F, DF!)
    cache = DogLegCache(x, y)
    linearproblem = LinearProblem(alloc_j(x, y))
    linearsolver = LinearSolver(linear_solver_method, y)
    ls = Linesearch(linesearch_problem(nlp, jacobian, cache), linesearch)
    DogLegSolver(x, nlp, linearproblem, linearsolver, ls, cache; jacobian=jacobian, kwargs...)
end

# Same pattern as the NewtonSolver/PicardSolver convenience form: `F` as a
# `missing`-defaulted keyword with a friendly error, and both vectors sharing
# an element type.
function DogLegSolver(x::AT, y::AT; F=missing, kwargs...) where {T,AT<:AbstractVector{T}}
    !ismissing(F) || error("You have to provide an F.")
    DogLegSolver(x, F, y; kwargs...)
end
NonlinearSolver(method::DogLeg, x...; kwargs...) = DogLegSolver(x...; refactorize=method.refactorize, kwargs...)
