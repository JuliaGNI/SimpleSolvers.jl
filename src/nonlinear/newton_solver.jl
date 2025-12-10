"""
    NewtonSolver

A `const` derived from [`NonlinearSolver`](@ref)

# Constructors

The `NewtonSolver` can be called with an [`NonlinearProblem`](@ref) or with a `Callable`. Note however that the latter will probably be deprecated in the future.
```jldoctest; setup=:(using SimpleSolvers)
linesearch = Quadratic()
F(y, x, params) = y .= tanh.(x)
x = [.5, .5]
y = zero(x)
F(y, x, nothing)

NewtonSolver(x, y; F = F, linesearch = linesearch)

# output

i=   0,
x= NaN,
f= NaN,
rxₐ= NaN,
rfₐ= NaN
```

What is shown here is the status of the `NewtonSolver`, i.e. an instance of [`NonlinearSolverStatus`](@ref).

# Keywords
- `nonlinearproblem::`[`NonlinearProblem`](@ref): the system that has to be solved. This can be accessed by calling [`nonlinearproblem`](@ref),
- `jacobian::`[`Jacobian`](@ref)
- `linear::`[`LinearSolver`](@ref): the linear solver is used to compute the [`direction`](@ref) of the solver step (see [`solver_step!`](@ref)). This can be accessed by calling [`linearsolver`](@ref),
- `linesearch::`[`LinesearchState`](@ref)
- `refactorize::Int`: determines after how many steps the Jacobian is updated and refactored (see [`factorize!`](@ref)). If we have `refactorize > 1`, then we speak of a [`QuasiNewtonSolver`](@ref),
- `cache::`[`NewtonSolverCache`](@ref)
- `config::`[`Options`](@ref)
- `status::`[`NonlinearSolverStatus`](@ref):
"""
const NewtonSolver{T} = NonlinearSolver{T, NewtonMethod}

function NewtonSolver(x::AT, nlp::NLST, ls::LST, linearsolver::LSoT, linesearch::LiSeT, cache::CT; jacobian::Jacobian=JacobianAutodiff(nlp.F, x), refactorize::Integer=1, options_kwargs...) where {T,AT<:AbstractVector{T},NLST,LST,LSoT,LiSeT,CT}
    cache = NewtonSolverCache(x, x)
    NonlinearSolver(x, nlp, ls, linearsolver, linesearch, cache; method = NewtonMethod(refactorize), jacobian=jacobian, options_kwargs...)
end

function QuasiNewtonSolver(x, nlp, ls, linearsolver, linesearch, cache; options_kwargs...)
    NewtonSolver(x, nlp, ls, linearsolver, linesearch, cache; refactorize = DEFAULT_ITERATIONS_QUASI_NEWTON_SOLVER, options_kwargs...)
end

"""
    NewtonSolver(x, F, y)

# Keywords
- `linear_solver_method`
- `DF!`
- `linesearch`
- `mode`
- `options_kwargs`: see [`Options`](@ref)
"""
function NewtonSolver(x::AT, F::Callable, y::AT; linear_solver_method=LU(), (DF!)=missing, linesearch=Backtracking(), jacobian=JacobianAutodiff(F, x, y), kwargs...) where {T,AT<:AbstractVector{T}}
    nlp = ismissing(DF!) ? NonlinearProblem(F, x, y) : NonlinearProblem(F, DF!, x, y)
    jacobian = ismissing(DF!) ? jacobian : JacobianFunction{T}(F, DF!)
    cache = NewtonSolverCache(x, y)
    linearproblem = LinearProblem(alloc_j(x, y))
    linearsolver = LinearSolver(linear_solver_method, y)
    ls = LinesearchState(linesearch; T=T)
    NewtonSolver(x, nlp, linearproblem, linearsolver, ls, cache; jacobian = jacobian, kwargs...)
end

function NewtonSolver(x::AT, y::AT; F=missing, kwargs...) where {T,AT<:AbstractVector{T}}
    !ismissing(F) || error("You have to provide an F.")
    NewtonSolver(x, F, y; kwargs...)
end

function solver_step!(s::NewtonSolver, x::AbstractVector{T}, params) where {T}
    update!(cache(s), x)
    value!!(nonlinearproblem(s), x, params)
    # first we update the rhs of the linearproblem
    update!(linearproblem(s), -value(nonlinearproblem(s)))
    rhs(cache(s)) .= rhs(linearproblem(s))
    # for a quasi-Newton method the Jacobian isn't updated in every iteration
    if (mod(iteration_number(s) - 1, method(s).refactorize) == 0 || iteration_number(s) == 1)
        jacobian!(s, x, params)
        update!(linearproblem(s), jacobian(s))
        factorize!(linearsolver(s), linearproblem(s))
    end
    ldiv!(direction(cache(s)), linearsolver(s), rhs(linearproblem(s)))
    α = linesearch(s)(linesearch_problem(s, params))
    x .= compute_new_iterate(x, α, direction(cache(s)))
    x
end

"""
    QuasiNewtonSolver

A convenience constructor for [`NewtonSolver`](@ref). Also see [`DEFAULT_ITERATIONS_QUASI_NEWTON_SOLVER`](@ref).

Calling `QuasiNewtonSolver` hence produces an instance of [`NewtonSolver`](@ref) with the difference that `refactorize ≠ 1`. The [`Jacobian`](@ref) is thus not evaluated and refactored in every step.

# Implementation
It does:

```julia
QuasiNewtonSolver(args...; kwargs...) = NewtonSolver(args...; refactorize=DEFAULT_ITERATIONS_QUASI_NEWTON_SOLVER, kwargs...)
```
"""
QuasiNewtonSolver(args...; kwargs...) = NewtonSolver(args...; refactorize=DEFAULT_ITERATIONS_QUASI_NEWTON_SOLVER, kwargs...)

cache(solver::NewtonSolver)::NewtonSolverCache = solver.cache
config(solver::NewtonSolver)::Options = solver.config
status(solver::NewtonSolver)::NonlinearSolverStatus = solver.status

linearproblem(solver::NewtonSolver) = solver.linearproblem

"""
    nonlinearproblem(solver)

Return the [`NonlinearProblem`](@ref) contained in the [`NewtonSolver`](@ref). Compare this to [`linearsolver`](@ref).
"""
nonlinearproblem(solver::NewtonSolver)::NonlinearProblem = solver.nonlinearproblem

value(solver::NewtonSolver) = value(nonlinearproblem(solver))

iteration_number(solver::NewtonSolver)::Integer = iteration_number(status(solver))

"""
    jacobian(solver::NewtonSolver)

Return the evaluated Jacobian (a Matrix) stored in the [`NonlinearProblem`](@ref) of `solver`.

Also see [`jacobian(::NonlinearProblem)`](@ref) and [`Jacobian(::NonlinearProblem)`](@ref).
"""
jacobian(solver::NewtonSolver)::AbstractMatrix = jacobian(nonlinearproblem(solver))

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
linearsolver(solver::NewtonSolver) = solver.linearsolver
linesearch(solver::NewtonSolver) = solver.linesearch

check_jacobian(s::NewtonSolver) = check_jacobian(jacobian(s))
print_jacobian(s::NewtonSolver) = print_jacobian(jacobian(s))

initialize!(s::NewtonSolver, x₀::AbstractArray) = initialize!(status(s), x₀)

"""
    update!(solver, x, params)

Update the `solver::`[`NewtonSolver`](@ref) based on `x`.
This updates the cache (instance of type [`NewtonSolverCache`](@ref)) and the status (instance of type [`NonlinearSolverStatus`](@ref)). In course of updating the latter, we also update the `nonlinear` stored in `solver` (and `status(solver)`).

!!! info
    At the moment this is neither used in `solver_step!` nor `solve!`.
"""
function update!(s::NewtonSolver, x₀::AbstractArray, params)
    update!(status(s), x₀, nonlinearproblem(s), params)
    update!(nonlinearproblem(s), Jacobian(s), x₀, params)
    update!(cache(s), x₀)

    s
end

"""
    solve!(s, x)

# Extended help

!!! info
    The function `update!` calls [`increase_iteration_number!`](@ref).
"""
solve!(x::AbstractArray, s::NewtonSolver) = solve!(x, s, NullParameters())

function solve!(x::AbstractArray, s::NewtonSolver, params)
    initialize!(s, x)
    update!(status(s), x, nonlinearproblem(s), params)

    while !meets_stopping_criteria(status(s), config(s))
        increase_iteration_number!(status(s))
        solver_step!(s, x, params)
        update!(status(s), x, nonlinearproblem(s), params)
        residual!(status(s))
    end

    print_status(status(s), config(s))
    warn_iteration_number(status(s), config(s))

    x
end

Base.show(io::IO, solver::NewtonSolver) = show(io, status(solver))
