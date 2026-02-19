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

    lmul!(rhs(linearproblem(s)), transpose(jacobianmatrix(s)), rhs(linearproblem(s)))
    mul!(matrix(linearproblem(s)), transpose(jacobianmatrix(s)), jacobianmatrix(s))
    matrix(linearproblem(s)) .+= config(s).regularization_factor .* I(length(x))
    factorize!(linearsolver(s), linearproblem(s))
    ldiv!(direction₂(cache(s)), linearsolver(s), rhs(linearproblem(s)))

    direction₁(cache(s)), direction₂(cache(s))
end

function solver_step!(x::AbstractVector{T}, s::DogLegSolver{T}, state::NonlinearSolverState{T}, params) where {T}
    directions!(s, x, params)
    any(isnan, direction(cache(s))) && throw(NonlinearSolverException("NaN detected in direction vector"))

    # The following loop checks if the RHS contains any NaNs.
    # If so, the direction vector is reduced by a factor of NAN_FACTOR.
    for _ in 1:config(s).nan_max_iterations
        solution(cache(s)) .= x .+ direction(cache(s))
        value!(value(cache(s)), nonlinearproblem(s), solution(cache(s)), params)
        if any(isnan, value(cache(s)))
            (s.config.verbosity ≥ 2 && @warn "NaN detected in nonlinear solver. Reducing length of direction vector.")
            direction(cache(s)) .*= T(config(s).nan_factor)
        else
            break
        end
    end

    α = solve(linesearch(s), one(T), (x=x, parameters=params))
    compute_new_iterate!(x, α, direction(cache(s)))

    x
end
