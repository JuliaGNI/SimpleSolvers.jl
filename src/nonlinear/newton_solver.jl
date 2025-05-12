"""
The default number of iterations before the [`Jacobian`](@ref) is refactored in the [`QuasiNewtonSolver`](@ref)
"""
const DEFAULT_ITERATIONS_QUASI_NEWTON_SOLVER = 5

"""
    NewtonSolver

A struct that comprises all Newton solvers. Those typically differ in the way the Jacobian is computed.

# Keywords
- `obj::`[`AbstractObjective`](@ref)
- `jacobian::`[`Jacobian`](@ref)
- `linear`
- `linesearch::`[`LinesearchState`](@ref)
- `refactorize::Int`
- `cache::`[`NewtonSolverCache`](@ref)
- `config::`[`Options`](@ref)
- `status::`[`NonlinearSolverStatus`](@ref)
"""
struct NewtonSolver{T, AT, OT <: AbstractObjective, JT, TJ <: Jacobian, TL, TLS <: LinesearchState, TST <: NonlinearSolverStatus{T}} <: NonlinearSolver
    obj::OT
    jacobian::TJ

    linear::TL
    linesearch::TLS

    refactorize::Int

    cache::NewtonSolverCache{T,AT,JT}
    config::Options{T}
    status::TST

    function NewtonSolver{T,AT,OT, JT, TJ, TL, TS}(x, jacobian, objective, linear_solver, linesearch, cache, config; refactorize = 1) where {T, AT, OT, JT, TJ, TL, TS}
        status = NonlinearSolverStatus{T}(length(x))
        new{T, AT, OT, JT, TJ, TL, TS, typeof(status)}(jacobian, objective, linear_solver, linesearch, refactorize, cache, config, status)
    end
end

function NewtonSolver(x::AT, y::AT; F = missing, DF! = missing, linesearch = Backtracking(), config = Options(), mode = :autodiff, kwargs...) where {T, AT <: AbstractVector{T}}
    n = length(y)
    !ismissing(F) || error("You have to provide an F.")
    objective = MultivariateObjective(F, x)
    jacobian = ismissing(DF!) ? Jacobian{T}(F, n; mode = mode) : Jacobian{T}(DF!, n; mode = :function)
    cache = NewtonSolverCache(x, y)
    linear_solver = LinearSolver(y; linear_solver = :julia)
    ls = LinesearchState(linesearch; T = T)
    options = Options(T, config)
    NewtonSolver{T, AT, typeof(objective), typeof(cache.J), typeof(jacobian), typeof(linear_solver), typeof(ls)}(x, objective, jacobian, linear_solver, ls, cache, options; kwargs...)
end

"""
    solver_step!(s)

Compute one Newton step for `f` based on the [`Jacobian`](@ref) `jacobian!`.
"""
function solver_step!(x::Union{AbstractVector{T}, T}, obj::AbstractObjective, jacobian!::Jacobian, s::NewtonSolver{T}) where {T}
    # update Newton solver cache
    update!(s, x)

    if mod(status(s).i-1, s.refactorize) == 0
        _compute_jacobian!(s, x, jacobian!)
        # factorize the jacobian stored in `s` and save the factorized matrix in the corresponding linear solver.
        factorize!(linearsolver(s), jacobian(cache(s)))
    end

    # compute RHS (f is an in-place function)
    value!(obj, x)
    cache(s).rhs .= value(obj)
    rmul!(cache(s).rhs, -1)

    # solve J δx = -f(x)
    ldiv!(direction(cache(s)), linearsolver(s), cache(s).rhs)

    # apply line search
    α = linesearch(s)(linesearch_objective(obj, jacobian(s), cache(s)))
    x .+= α .* direction(cache(s))
end

_compute_jacobian!(s::NewtonSolver, x, jacobian!::Callable) = compute_jacobian!(s, x, jacobian!; mode = :function)
_compute_jacobian!(s::NewtonSolver, x, jacobian!::Jacobian) = compute_jacobian!(s, x, jacobian!)

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

cache(solver::NewtonSolver) = solver.cache
config(solver::NewtonSolver) = solver.config
status(solver::NewtonSolver) = solver.status

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
update!(s::NewtonSolver, x₀::AbstractArray) = update!(cache(s), x₀)

function solve!(x, f::Callable, s::NewtonSolver)
    solve!(x, f, jacobian(s), s)
end