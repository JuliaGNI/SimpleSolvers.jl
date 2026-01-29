using Printf

"""
    NonlinearSolver

A `struct` that comprises *Newton solvers* (see [`NewtonMethod`](@ref)) and the *fixed point iterator* (see [`PicardMethod`](@ref)).

# Constructors

```julia
NonlinearSolver(x, nlp, ls, linearsolver, linesearch, cache; method)
```

The `NonlinearSolver` can be called with an [`NonlinearProblem`](@ref) or with a `Callable`. Note however that the latter will probably be deprecated in the future. See [`NewtonSolver`](@ref) for examples (as well as [`NonlinearSolverStatus`](@ref)).

It's arguments are:
- `nlp::`[`NonlinearProblem`](@ref): the system that has to be solved. This can be accessed by calling [`nonlinearproblem`](@ref),
- `ls::`[`LinearProblem`](@ref),
- `linearsolver::`[`LinearSolver`](@ref): the linear solver is used to compute the [`direction`](@ref) of the solver step (see [`solver_step!`](@ref)). This can be accessed by calling [`linearsolver`](@ref),
- `linesearch::`[`Linesearch`](@ref)
- `cache::`[`NonlinearSolverCache`](@ref)
- `config::`[`Options`](@ref)
- `status::`[`NonlinearSolverStatus`](@ref):
"""
struct NonlinearSolver{T,MT<:NonlinearSolverMethod,AT,NLST<:NonlinearProblem{T},LST<:AbstractLinearProblem,JT<:Jacobian{T},LSoT<:AbstractLinearSolver,LiSeT<:Linesearch{T},CT<:NonlinearSolverCache{T}} <: AbstractSolver
    nonlinearproblem::NLST
    linearproblem::LST
    jacobian::JT
    linearsolver::LSoT
    linesearch::LiSeT
    method::MT

    cache::CT
    config::Options{T}

    function NonlinearSolver(x::AT, nlp::NLST, ls::LST, linearsolver::LSoT, linesearch::LiSeT, cache::CT, config::Options{T}; method::MT=NewtonMethod(), jacobian::JT=JacobianAutodiff(nlp.F, x), options_kwargs...) where {T,AT<:AbstractVector{T},MT<:NonlinearSolverMethod,JT<:Jacobian,NLST,LST,LSoT,LiSeT,CT}
        new{T,MT,AT,NLST,LST,JT,LSoT,LiSeT,CT}(nlp, ls, jacobian, linearsolver, linesearch, method, cache, config)
    end
end

cache(s::NonlinearSolver) = s.cache
config(s::NonlinearSolver) = s.config
method(s::NonlinearSolver) = s.method

linearproblem(s::NonlinearSolver) = s.linearproblem
linesearch(s::NonlinearSolver) = s.linesearch
Jacobian(s::NonlinearSolver) = s.jacobian

solver_step!(s::NonlinearSolver) = error("solver_step! not implemented for $(typeof(s))")

function initialize!(s::NonlinearSolver, x::AbstractVector)
    initialize!(cache(s), x)

    s
end


"""
    nonlinearproblem(solver)

Return the [`NonlinearProblem`](@ref) contained in the [`NonlinearSolver`](@ref). Compare this to [`linearsolver`](@ref).
"""
nonlinearproblem(s::NonlinearSolver) = s.nonlinearproblem

jacobian!(s::NonlinearSolver{T}, x::AbstractVector{T}, params) where {T} = Jacobian(s)(jacobian(cache(s)), x, params)

"""
    jacobian(solver::NewtonSolver)

Return the evaluated Jacobian (a Matrix) stored in the [`NonlinearProblem`](@ref) of `solver`.

Also see [`jacobian(::NonlinearProblem)`](@ref) and [`Jacobian(::NonlinearProblem)`](@ref).
"""
jacobian(solver::NonlinearSolver) = jacobian(cache(solver))

"""
    linearsolver(solver)

Return the linear part (i.e. a [`LinearSolver`](@ref)) of an [`NewtonSolver`](@ref).

# Examples

```jldoctest; setup = :(using SimpleSolvers; using SimpleSolvers: linearsolver)
x = rand(3)
y = rand(3)
F(x) = tanh.(x)
F!(y, x, params) = y .= F(x)
s = NewtonSolver(x, y; F = F!)
linearsolver(s)

# output

LinearSolver{Float64, LU{Missing}, SimpleSolvers.LUSolverCache{Float64, StaticArraysCore.MMatrix{3, 3, Float64, 9}}}(LU{Missing}(missing, true), SimpleSolvers.LUSolverCache{Float64, StaticArraysCore.MMatrix{3, 3, Float64, 9}}([0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0], [0, 0, 0], [0, 0, 0], 0))
```
"""
linearsolver(solver::NonlinearSolver) = solver.linearsolver


struct NonlinearSolverException <: Exception
    msg::String
end

Base.showerror(io::IO, e::NonlinearSolverException) = print(io, "Nonlinear Solver Exception: ", e.msg, "!")

"""
    solver_step!(x, s, state, params)

Solve the problem stored in an instance `s` of [`NonlinearSolver`](@ref).
"""
function solver_step!(x::AbstractVector{T}, s::NonlinearSolver{T}, state::NonlinearSolverState{T}, params) where {T}
    direction!(s, x, params, iteration_number(state))
    # The following loop checks if the RHS contains any NaNs.
    # If so, the direction vector is reduced by a factor of LINESEARCH_NAN_FACTOR.
    for _ in 1:linesearch(s).config.linesearch_nan_max_iterations
        solution(cache(s)) .= x .+ direction(cache(s))
        value!(value(cache(s)), nonlinearproblem(s), solution(cache(s)), params)
        if any(isnan, value(cache(s)))
            (s.config.verbosity ≥ 2 && @warn "NaN detected in nonlinear solver. Reducing length of direction vector.")
            direction(cache(s)) .*= T(linesearch(s).config.linesearch_nan_factor)
        else
            break
        end
    end
    α = solve(linesearch_problem(s, x, params), linesearch(s))
    compute_new_iterate!(x, α, direction(cache(s)))
    x
end

mean(x::AbstractVector) = sum(x) / length(x)

"""
    solve!(x, s)

# Extended help

!!! info
    The function `update!` calls [`increase_iteration_number!`](@ref).
"""
function solve!(x::AbstractArray, s::NonlinearSolver, state::NonlinearSolverState, params=NullParameters())
    initialize!(s, x)
    initialize!(state, x, value!(value(cache(s)), nonlinearproblem(s), x, params))

    while true
        increase_iteration_number!(state)
        solver_step!(x, s, state, params)
        update!(state, x, value!(value(cache(s)), nonlinearproblem(s), x, params))
        meets_stopping_criteria(state, config(s)) && break
    end

    status = NonlinearSolverStatus(state, config(s))
    nonlinear_solver_warnings(status, config(s))
    config(s).verbosity > 1 && print_status(status, config(s))

    x
end

solve!(x::AbstractArray, s::NonlinearSolver, params=NullParameters()) = solve!(x, s, NonlinearSolverState(x, value(cache(s))), params)
