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

# Extended help

The Gauss-Newton direction (i.e. [`direction₂`](@ref)) is computed using the following formula:

```math
\mathbf{d}_2 = (\mathbf{J}^T \mathbf{J})^{-1} \mathbf{r}
```

where ``\mathbf{J}`` is the Jacobian matrix and ``\mathbf{r}`` is the residual vector:

```math
\mathbf{r} = -\mathbf{J}(\mathbf{x})^T\mathbf{f}(\mathbf{x})
```
"""
function directions!(s::DogLegSolver{T}, x::AbstractVector{T}, params) where {T}
    # first we update the rhs of the linearproblem
    value!(rhs(linearproblem(s)), nonlinearproblem(s), x, params)
    rhs(linearproblem(s)) .*= -1
    jacobian!(s, x, params)
    matrix(linearproblem(s)) .= jacobianmatrix(s)
    matrix(linearproblem(s)) .+= config(s).regularization_factor .* I(length(x))
    factorize!(linearsolver(s), linearproblem(s))
    ldiv!(direction₁(cache(s)), linearsolver(s), rhs(linearproblem(s)))

    mul!(rhs(linearproblem(s)), transpose(jacobianmatrix(s)), copy(rhs(linearproblem(s))))
    mul!(matrix(linearproblem(s)), transpose(jacobianmatrix(s)), jacobianmatrix(s))
    matrix(linearproblem(s)) .+= config(s).regularization_factor .* I(length(x))
    factorize!(linearsolver(s), linearproblem(s))
    ldiv!(direction₂(cache(s)), linearsolver(s), rhs(linearproblem(s)))

    direction₁(cache(s)), direction₂(cache(s))
end

function solver_step!(x::AbstractVector{T}, s::DogLegSolver{T}, state::NonlinearSolverState{T}, params; Δ::T=T(INITIAL_Δ)) where {T}
    Δ > eps(T) || (@warn "Δ must be greater than zero. Iteration stops."; return x)
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

    direction(cache(s)) .= if l2norm(direction₂(cache(s))) ≤ Δ
        direction₂(cache(s))
    elseif l2norm(direction₁(cache(s))) > Δ
        direction₁(cache(s)) / l2norm(direction₁(cache(s))) * Δ
    else
        d_diff = direction₂(cache(s)) - direction₁(cache(s))
        d₁d₂d₁ = direction₁(cache(s)) ⋅ d_diff
        s = -d₁d₂d₁ / L2norm(d_diff) + √(d₁d₂d₁^2 / L2norm(d_diff)^2 - (L2norm(direction₁(cache(s))) - Δ^2))
        direction₁(cache(s)) + s * (direction₂(cache(s)) - direction₁(cache(s)))
    end
    # println(direction₁(cache(s)))
    # println(direction₂(cache(s)))
    # println(direction(cache(s)))
    # println("")

    compute_new_iterate!(solution(cache(s)), one(T), direction(cache(s)))
    value!(value(cache(s)), nonlinearproblem(s), solution(cache(s)), params)
    if L2norm(value(cache(s))) ≤ L2norm(value(state)) + DEFAULT_WOLFE_c₁ * dot(direction(cache(s)), jacobianmatrix(s), value(state))
        x .= solution(cache(s))
    else
        solver_step!(x, s, state, params; Δ=Δ * T(DEFAULT_Δ_REDUCTION))
    end

    x
end

function DogLegSolver(x::AT, nlp::NLST, ls::LST, linearsolver::LSoT, linesearch::LiSeT, cache::CT; jacobian::Jacobian=JacobianAutodiff(nlp.F, x), options_kwargs...) where {T,AT<:AbstractVector{T},NLST,LST,LSoT,LiSeT,CT}
    config = Options(T; options_kwargs...)

    NonlinearSolver(x, nlp, ls, linearsolver, linesearch, cache, config; method=DogLeg(), jacobian=jacobian)
end

function DogLegSolver(x::AbstractVector{T}, F::Callable, y::AbstractVector{T}; linear_solver_method=LU(), (DF!)=missing, linesearch=Backtracking(T), jacobian=JacobianAutodiff(F, x, y), refactorize=1, kwargs...) where {T}
    nlp = NonlinearProblem(F, DF!, x, y)
    jacobian = ismissing(DF!) ? jacobian : JacobianFunction{T}(F, DF!)
    cache = DogLegCache(x, y)
    linearproblem = LinearProblem(alloc_j(x, y))
    linearsolver = LinearSolver(linear_solver_method, y)
    ls = Linesearch(linesearch_problem(nlp, jacobian, cache), linesearch)
    DogLegSolver(x, nlp, linearproblem, linearsolver, ls, cache; jacobian=jacobian, kwargs...)
end
