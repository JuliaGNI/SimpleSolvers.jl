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
x₁= NaN,
f= NaN,
rxₐ= NaN,
rxᵣ= NaN,
rfₐ= NaN,
rfᵣ= NaN
````

What is shown here is the status of the `NewtonSolver`, i.e. an instance of [`NonlinearSolverStatus`](@ref).

# Keywords
- `nonlinear::`[`NonlinearSystem`](@ref): the system that has to be solved. This can be accessed by calling [`nonlinearsystem`](@ref),
- `jacobian::`[`Jacobian`](@ref)
- `linear::`[`LinearSolver`](@ref): the linear solver is used to compute the [`direction`](@ref) of the solver step (see [`solver_step!`](@ref)). This can be accessed by calling [`linearsolver`](@ref),
- `linesearch::`[`LinesearchState`](@ref)
- `refactorize::Int`: determines after how many steps the Jacobian is updated and refactored (see [`factorize!`](@ref)). If we have `refactorize > 1`, then we speak of a [`QuasiNewtonSolver`](@ref),
- `cache::`[`NewtonSolverCache`](@ref)
- `config::`[`Options`](@ref)
- `status::`[`NonlinearSolverStatus`](@ref): 
"""
struct NewtonSolver{T, AT, NLST <: NonlinearSystem, LSyT <: LinearSystem, LSoT <: LinearSolver, LiSeT <: LinesearchState, TC <: NewtonSolverCache, NSST <: NonlinearSolverStatus{T}} <: NonlinearSolver
    nonlinearsystem::NLST
    linearsystem::LSyT

    linearsolver::LSoT
    linesearch::LiSeT

    refactorize::Int

    cache::TC
    config::Options{T}
    status::NSST

    function NewtonSolver(x::AT, nonlinearsystem::NLST, linearsystem::LST, linearsolver::TL, linesearch::TLS, cache::TC, config::Options; refactorize::Integer = 1) where {T, AT <: AbstractVector{T}, NLST, LST, TL, TLS, TC}
        status = NonlinearSolverStatus(x, nonlinearsystem)
        new{T, AT, NLST, LST, TL, TLS, TC, typeof(status)}(nonlinearsystem, linearsystem, linearsolver, linesearch, refactorize, cache, config, status)
    end
end

function NewtonSolver(x::AT, y::AT; F = missing, kwargs...) where {T, AT <: AbstractVector{T}}
    !ismissing(F) || error("You have to provide an F.")
    nonlinear = MultivariateObjective(F, x)
    NewtonSolver(nonlinear, x, y; kwargs...)
end

function NewtonSolver(nonlinearsystem::NonlinearSystem, x::AT, y::AT; DF! = missing, linesearch = Backtracking(), config = Options(), mode = :autodiff, kwargs...) where {T, AT <: AbstractVector{T}}
    n = length(y)
    jacobian = ismissing(DF!) ? Jacobian{T}(Function(nonlinearsystem), n; mode = mode) : Jacobian{T}(DF!, n; mode = :function)
    cache = NewtonSolverCache(x, y)
    linearsolver = LinearSolver(y; linearsolver = :julia)
    ls = LinesearchState(linesearch; T = T)
    options = Options(T, config)
    NewtonSolver(x, nonlinearsystem, linearsystem, linearsolver, linesearch, cache, options; kwargs...)
end

"""
    solver_step!(s)

Compute one Newton step for `f` based on the [`Jacobian`](@ref) `jacobian!`.
"""
function solver_step!(x::Union{AbstractVector{T}, T}, obj::NonlinearSystem, jacobian!::Jacobian, s::NewtonSolver{T}) where {T}
    # update Newton solver cache
    update!(s, x)

    update_rhs_and_direction!(s, jacobian!)

    # apply line search
    α = linesearch(s)(linesearch_nonlinear(obj, jacobian(s), cache(s)))
    x .+= compute_new_iterate(x, α, direction(cache(s)))
end

"""
    update_rhs_and_direction(solver, jacobian!, x)

Update the rhs and the [`direction`](@ref) of `solver::`[`NewtonSolver`](@ref).

This is used in addition to [`update!(::NewtonSolver, ::AbstractArray)`](@ref) when calling [`solver_step!`](@ref).
"""
function update_rhs_and_direction!(solver::NewtonSolver, jacobian!::Jacobian, x::AbstractVector)
    @info string(iteration_number(solver))
    if (mod(iteration_number(solver)-1, solver.refactorize) == 0 || iteration_number(solver) == 0 || iteration_number(solver) == 1)
        compute_jacobian!(solver, x, jacobian!)
        # factorize the jacobian stored in `s` and save the factorized matrix in the corresponding linear solver.
        factorize!(linearsolver(solver), jacobian(cache(solver)))
    end

    # compute RHS (f is an in-place function)
    cache(solver).rhs .= value(nonlinear(solver))
    rmul!(cache(solver).rhs, -1)

    # solve J δx = -f(x)
    ldiv!(direction(cache(solver)), linearsolver(solver), cache(solver).rhs)

    solver
end

function solver_linear_system!(solver::LinearSolver, ls::LinearSystem, x::AbstractVector)

end

update_rhs_and_direction!(solver::NewtonSolver, jacobian!::Jacobian) = update_rhs_and_direction!(solver, jacobian!::Jacobian, solution(cache(solver)))

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

"""
    nonlinearsystem(solver)

Return the [`NonlinearSystem`](@ref) contained in the [`NewtonSolver`](@ref). Compare this to [`linearsolver`](@ref).
"""
nonlinearsystem(solver::NewtonSolver)::NonlinearSystem = solver.nonlinear
iteration_number(solver::NewtonSolver)::Integer = iteration_number(status(solver))

"""
    jacobian(solver::NewtonSolver)

Calling `jacobian` on an instance of [`NewtonSolver`](@ref) produces a slight ambiguity since the `cache` (of type [`NewtonSolverCache`](@ref)) also stores a Jacobian, but in the latter case it is a matrix not an instance of type [`Jacobian`](@ref).
Hence we return the object of type [`Jacobian`](@ref) when calling `jacobian`. This is also used in [`solver_step!`](@ref).
"""
jacobian(solver::NewtonSolver)::Jacobian = solver.jacobian

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

LUSolver{Float64}(3, [0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0], [1, 2, 3], [1, 2, 3], 1)
```
"""
linearsolver(solver::NewtonSolver) = solver.linear
linesearch(solver::NewtonSolver) = solver.linesearch

compute_jacobian!(s::NewtonSolver, x, jacobian!::Union{Jacobian, Callable}; kwargs...) = compute_jacobian!(jacobian(cache(s)), x, jacobian!; kwargs...)

check_jacobian(s::NewtonSolver) = check_jacobian(jacobian(s))
print_jacobian(s::NewtonSolver) = print_jacobian(jacobian(s))

initialize!(s::NewtonSolver, x₀::AbstractArray, f) = initialize!(status(s), x₀, f)

"""
    update!(solver, x)

Update the `solver::`[`NewtonSolver`](@ref) based on `x`.
This updates the cache (instance of type [`NewtonSolverCache`](@ref)) and the status (instance of type [`NonlinearSolverStatus`](@ref)). In course of updating the latter, we also update the `nonlinear` stored in `solver` (and `status(solver)`).
"""
function update!(s::NewtonSolver, x₀::AbstractArray)
    update!(status(s), x₀, nonlinear(s))
    update!(cache(s), x₀)

    s
end

function solve!(x, f::Callable, s::NewtonSolver)
    solve!(x, f, jacobian(s), s)
end

Base.show(io::IO, solver::NewtonSolver) = show(io, status(solver))