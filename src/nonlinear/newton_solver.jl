"""
The default number of iterations before the [`Jacobian`](@ref) is refactored in the [`QuasiNewtonSolver`](@ref)
"""
const DEFAULT_ITERATIONS_QUASI_NEWTON_SOLVER = 5

"""
    NewtonSolver

A `struct` that comprises all Newton solvers. Those typically differ in the way the [`Jacobian`](@ref) is computed.

# Constructors

The `NewtonSolver` can be called with an [`NonlinearSystem`](@ref) or with a `Callable`. Note however that the latter will probably be deprecated in the future.
```jldoctest; setup=:(using SimpleSolvers)
linesearch = Quadratic()
F(x) = tanh.(x)
x = [.5, .5]
NewtonSolver(x, F(x); F = F, linesearch = linesearch)

# output

i=   0,
x= NaN,
f= NaN,
rxₐ= NaN,
rfₐ= NaN
```

What is shown here is the status of the `NewtonSolver`, i.e. an instance of [`NonlinearSolverStatus`](@ref).

# Keywords
- `nonlinearsystem::`[`NonlinearSystem`](@ref): the system that has to be solved. This can be accessed by calling [`nonlinearsystem`](@ref),
- `jacobian::`[`Jacobian`](@ref)
- `linear::`[`LinearSolver`](@ref): the linear solver is used to compute the [`direction`](@ref) of the solver step (see [`solver_step!`](@ref)). This can be accessed by calling [`linearsolver`](@ref),
- `linesearch::`[`LinesearchState`](@ref)
- `refactorize::Int`: determines after how many steps the Jacobian is updated and refactored (see [`factorize!`](@ref)). If we have `refactorize > 1`, then we speak of a [`QuasiNewtonSolver`](@ref),
- `cache::`[`NewtonSolverCache`](@ref)
- `config::`[`Options`](@ref)
- `status::`[`NonlinearSolverStatus`](@ref):
"""
struct NewtonSolver{T,AT,NLST<:NonlinearSystem{T},LST<:LinearSystem{T},LSoT<:LinearSolver{T},LiSeT<:LinesearchState{T},CT<:NewtonSolverCache{T},NSST<:NonlinearSolverStatus{T}} <: NonlinearSolver
    nonlinearsystem::NLST
    linearsystem::LST
    linearsolver::LSoT
    linesearch::LiSeT

    refactorize::Int

    cache::CT
    config::Options{T}
    status::NSST

    function NewtonSolver(x::AT, nls::NLST, ls::LST, linearsolver::LSoT, linesearch::LiSeT, cache::CT; refactorize::Integer=1, options_kwargs...) where {T,AT<:AbstractVector{T},NLST,LST,LSoT,LiSeT,CT}
        status = NonlinearSolverStatus(x)
        config = Options(T; options_kwargs...)
        new{T,AT,NLST,LST,LSoT,LiSeT,CT,typeof(status)}(nls, ls, linearsolver, linesearch, refactorize, cache, config, status)
    end
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
function NewtonSolver(x::AT, F::Callable, y::AT=F(x); linear_solver_method=LU(), (DF!)=missing, linesearch=Backtracking(), mode=:autodiff, kwargs...) where {T,AT<:AbstractVector{T}}
    nls = ismissing(DF!) ? NonlinearSystem(F, x; mode=mode) : NonlinearSystem(F, DF!, x)
    cache = NewtonSolverCache(x, y)
    linearsystem = LinearSystem(alloc_j(x, y))
    linearsolver = LinearSolver(linear_solver_method, y)
    ls = LinesearchState(linesearch; T=T)
    NewtonSolver(x, nls, linearsystem, linearsolver, ls, cache; kwargs...)
end

function NewtonSolver(x::AT, y::AT; F=missing, kwargs...) where {T,AT<:AbstractVector{T}}
    !ismissing(F) || error("You have to provide an F.")
    NewtonSolver(x, F, y; kwargs...)
end

"""
    solver_step!(s, x)

Compute one Newton step for `f` based on the [`Jacobian`](@ref) `jacobian!`.
"""
function solver_step!(s::NewtonSolver, x::AbstractVector{T}) where {T}
    update!(cache(s), x)
    value!(nonlinearsystem(s), x)
    # first we update the rhs of the linearsystem
    update!(linearsystem(s), -value(nonlinearsystem(s)))
    rhs(cache(s)) .= rhs(linearsystem(s))
    # for a quasi-Newton method the Jacobian isn't updated in every iteration
    if (mod(iteration_number(s) - 1, s.refactorize) == 0 || iteration_number(s) == 1)
        jacobian!(nonlinearsystem(s), x)
        update!(linearsystem(s), jacobian(s))
        factorize!(linearsolver(s), linearsystem(s))
    end
    ldiv!(direction(cache(s)), linearsolver(s), rhs(linearsystem(s)))
    α = linesearch(s)(linesearch_objective(s))
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

linearsystem(solver::NewtonSolver) = solver.linearsystem

"""
    nonlinearsystem(solver)

Return the [`NonlinearSystem`](@ref) contained in the [`NewtonSolver`](@ref). Compare this to [`linearsolver`](@ref).
"""
nonlinearsystem(solver::NewtonSolver)::NonlinearSystem = solver.nonlinearsystem

value(solver::NewtonSolver) = value(nonlinearsystem(solver))

iteration_number(solver::NewtonSolver)::Integer = iteration_number(status(solver))

"""
    Jacobian(solver::NewtonSolver)

Return the [`Jacobian`](@ref) stored in the [`NonlinearSystem`](@ref) of `solver`.
"""
Jacobian(solver::NewtonSolver)::Jacobian = Jacobian(nonlinearsystem(solver))

"""
    jacobian(solver::NewtonSolver)

Return the evaluated Jacobian (a Matrix) stored in the [`NonlinearSystem`](@ref) of `solver`.

Also see [`jacobian(::NonlinearSystem)`](@ref) and [`Jacobian(::NonlinearSystem)`](@ref).
"""
jacobian(solver::NewtonSolver)::AbstractMatrix = jacobian(nonlinearsystem(solver))

"""
    linearsolver(solver)

Return the linear part (i.e. a [`LinearSolver`](@ref)) of an [`NewtonSolver`](@ref).

# Examples

```jldoctest; setup = :(using SimpleSolvers; using SimpleSolvers: linearsolver)
x = rand(3)
y = rand(3)
F(x) = tanh.(x)
s = NewtonSolver(x, y; F = F)
linearsolver(s)

# output

LinearSolver{Float64, LU{Missing}, SimpleSolvers.LUSolverCache{Float64, StaticArraysCore.MMatrix{3, 3, Float64, 9}}}(LU{Missing}(missing, true), SimpleSolvers.LUSolverCache{Float64, StaticArraysCore.MMatrix{3, 3, Float64, 9}}([0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0], [0, 0, 0], [0, 0, 0], 0))
```
"""
linearsolver(solver::NewtonSolver) = solver.linearsolver
linesearch(solver::NewtonSolver) = solver.linesearch

function compute_jacobian!(s::NewtonSolver, x; kwargs...)
    compute_jacobian!(jacobian(s), x, Jacobian(s); kwargs...)
end

function compute_jacobian!(s::NewtonSolver, x, jacobian!::Union{Jacobian,Callable}; kwargs...)
    @warn "This function should not be called! Instead call `compute_jacobian!(s, x)`."
    compute_jacobian!(jacobian(nonlinearsystem(s)), x, jacobian!; kwargs...)
end

check_jacobian(s::NewtonSolver) = check_jacobian(jacobian(s))
print_jacobian(s::NewtonSolver) = print_jacobian(jacobian(s))

initialize!(s::NewtonSolver, x₀::AbstractArray) = initialize!(status(s), x₀)

"""
    update!(solver, x)

Update the `solver::`[`NewtonSolver`](@ref) based on `x`.
This updates the cache (instance of type [`NewtonSolverCache`](@ref)) and the status (instance of type [`NonlinearSolverStatus`](@ref)). In course of updating the latter, we also update the `nonlinear` stored in `solver` (and `status(solver)`).
"""
function update!(s::NewtonSolver, x₀::AbstractArray)
    update!(status(s), x₀, nonlinearsystem(s))
    update!(nonlinearsystem(s), x₀)
    update!(cache(s), x₀)

    s
end

"""
    solve!(s, x)

# Extended help

!!! info
    The function `update!` calls `next_iteration!`.
"""
function solve!(s::NewtonSolver, x::AbstractArray)
    initialize!(s, x)
    update!(status(s), x, nonlinearsystem(s))

    while !meets_stopping_criteria(status(s), config(s))
        solver_step!(s, x)
        update!(status(s), x, nonlinearsystem(s))
        residual!(status(s))
    end

    print_status(status(s), config(s))
    warn_iteration_number(status(s), config(s))

    x
end

Base.show(io::IO, solver::NewtonSolver) = show(io, status(solver))
