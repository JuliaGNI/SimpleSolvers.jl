const DogLegSolver{T} = NonlinearSolver{T, DogLeg}

"""
    directions!(s, x, params)

Compute [`direction₁`](@ref) and [`direction₂`](@ref) for the [`DogLegSolver`](@ref).
This is equivalent to [`direction!`](@ref) for the [`NewtonSolver`](@ref).
"""
function directions!(s::DogLegSolver{T}, x::AbstractVector{T}, params) where {T}
    direction₁(cache(s)) .= Nan(T)
    direction₂(cache(s)) .= Nan(T)
end

# function direction!(d::AbstractVector{T}, x::AbstractVector{T}, s::Union{NewtonSolver{T},QuasiNewtonSolver{T}}, params, iteration) where {T}
#     # first we update the rhs of the linearproblem
#     value!(rhs(linearproblem(s)), nonlinearproblem(s), x, params)
#     rhs(linearproblem(s)) .*= -1
#     jacobian!(s, x, params)
#     matrix(linearproblem(s)) .= jacobianmatrix(s)
#     matrix(linearproblem(s)) .+= config(s).regularization_factor .* I(length(x))
#     factorize!(linearsolver(s), linearproblem(s))
#     ldiv!(d, linearsolver(s), rhs(linearproblem(s)))
# end

function solver_step!(x::AbstractVector{T}, s::DogLegSolver{T}, state::NonlinearSolverState{T}, params) where {T}
    direction!(s, x, params, iteration_number(state))
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