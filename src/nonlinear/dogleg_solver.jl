const INITIAL_Δ = 1.0
const DEFAULT_Δ_REDUCTION = 0.5

"""
    DogLegSolver

The [`NonlinearSolver`](@ref) for the [`DogLeg`](@ref) method.
"""
const DogLegSolver{T} = NonlinearSolver{T,DogLeg}

@doc raw"""
    directions!(s, x, params)

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
"""
function directions!(s::DogLegSolver{T}, x::AbstractVector{T}, params) where {T}
    # the Newton direction
    value!(rhs(linearproblem(s)), nonlinearproblem(s), x, params)
    rhs(linearproblem(s)) .*= -1
    jacobian!(s, x, params)
    matrix(linearproblem(s)) .= jacobianmatrix(s)
    idxs = diagind(matrix(linearproblem(s)))
    @view(matrix(linearproblem(s))[idxs]) .+= config(s).regularization_factor
    factorize!(linearsolver(s), linearproblem(s))
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

function solver_step!(x::AbstractVector{T}, s::DogLegSolver{T}, state::NonlinearSolverState{T}, params; Δ::T=T(INITIAL_Δ)) where {T}
    directions!(s, x, params)
    any(isnan, direction₁(cache(s))) && throw(NonlinearSolverException("NaN detected in direction₁ vector"))
    any(isnan, direction₂(cache(s))) && throw(NonlinearSolverException("NaN detected in direction₂ vector"))

    # The following loop checks if the RHS contains any NaNs.
    # If so, the direction vector is reduced by a factor of NAN_FACTOR.
    for _ in 1:config(s).nan_max_iterations
        solution(cache(s)) .= x .+ direction₁(cache(s))
        value!(value(cache(s)), nonlinearproblem(s), solution(cache(s)), params)
        if any(isnan, value(cache(s)))
            (s.config.verbosity ≥ 2 && @warn "NaN detected in nonlinear solver. Reducing length of direction₁ vector.")
            direction₁(cache(s)) .*= T(config(s).nan_factor)
        else
            break
        end
    end
    for _ in 1:config(s).nan_max_iterations
        solution(cache(s)) .= x .+ direction₂(cache(s))
        value!(value(cache(s)), nonlinearproblem(s), solution(cache(s)), params)
        if any(isnan, value(cache(s)))
            (s.config.verbosity ≥ 2 && @warn "NaN detected in nonlinear solver. Reducing length of direction₂ vector.")
            direction₂(cache(s)) .*= T(config(s).nan_factor)
        else
            break
        end
    end

    # Shrink the trust-region radius Δ until the dogleg step satisfies the
    # sufficient-decrease condition.  This is an in-place loop (not recursion), so
    # `directions!` — and hence the Jacobian evaluation and factorization — runs
    # exactly once per solver step, regardless of how many times Δ is shrunk.
    # The termination on the Δ floor is independent of `verbosity`.
    while Δ > eps(T)
        dogleg_direction!(cache(s), Δ)
        compute_new_iterate!(solution(cache(s)), solution(state), one(T), direction(cache(s)))
        value!(value(cache(s)), nonlinearproblem(s), solution(cache(s)), params)
        # sufficient-decrease for the merit φ(x) = ‖F(x)‖² = L2norm(F): the model
        # decrease is c₁·∇φ(x)ᵀd with ∇φ = 2·JᵀF, i.e. c₁·2·Fᵀ·J·d, which is
        # negative for a descent direction.
        if L2norm(value(cache(s))) ≤ L2norm(value(state)) + DEFAULT_WOLFE_c₁ * 2 * dot(value(state), jacobianmatrix(s), direction(cache(s)))
            x .= solution(cache(s))
            return x
        end
        Δ *= T(DEFAULT_Δ_REDUCTION)
    end

    # The trust-region radius underflowed without achieving sufficient decrease.
    # Take the last (smallest-Δ) step; the convergence check will decide whether
    # this is an acceptable stationary point.
    verbosity(config(s)) ≥ 1 && @warn "DogLeg trust-region radius Δ underflowed without sufficient decrease (iterations: $(iteration_number(state)))."
    x .= solution(cache(s))

    x
end

function DogLegSolver(x::AT, nlp::NLST, ls::LST, linearsolver::LSoT, linesearch::LiSeT, cache::CT; jacobian::Jacobian=JacobianAutodiff(nlp.F, x), options_kwargs...) where {T,AT<:AbstractVector{T},NLST,LST,LSoT,LiSeT,CT}
    config = Options(T; options_kwargs...)

    NonlinearSolver(x, nlp, ls, linearsolver, linesearch, cache, config; method=DogLeg(), jacobian=jacobian)
end

# Note: the `DogLeg` method has no `refactorize` option — it re-evaluates and
# refactorizes the Jacobian on every step — so (unlike `NewtonSolver`) this
# constructor does not accept a `refactorize` keyword; passing one is rejected
# rather than silently ignored.
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

DogLegSolver(x::AbstractVector, y::AbstractVector; F::Callable, kwargs...) = DogLegSolver(x, F, y; kwargs...)
NonlinearSolver(::DogLeg, x...; kwargs...) = DogLegSolver(x...; kwargs...)
