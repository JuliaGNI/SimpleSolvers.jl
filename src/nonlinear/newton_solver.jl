"""
    NewtonSolver

A struct that comprises all Newton solvers. Those typically differ in the way the Jacobian is computed.
"""
struct NewtonSolver{T, AT, OT <: AbstractObjective, JT, TJ <: Jacobian, TL, TLS <: LinesearchState, TST <: NonlinearSolverStatus{T}} <: AbstractNewtonSolver{T,AT}
    obj::OT
    jacobian::TJ

    linear::TL
    linesearch::TLS

    cache::NewtonSolverCache{T,AT,JT}
    config::Options{T}
    status::TST

    function NewtonSolver{T,AT,OT, JT, TJ, TL, TS}(x, jacobian, objective, linear_solver, linesearch, cache, config) where {T, AT, OT, JT, TJ, TL, TS}
        status = NonlinearSolverStatus{T}(length(x))
        new{T, AT, OT, JT, TJ, TL, TS, typeof(status)}(jacobian, objective, linear_solver, linesearch, cache, config, status)
    end
end

function NewtonSolver(x::AT, y::AT; F = missing, DF! = missing, linesearch = Backtracking(), config = Options(), mode = :autodiff) where {T, AT <: AbstractVector{T}}
    n = length(y)
    !ismissing(F) || error("You have to provide an F.")
    objective = MultivariateObjective(F, x)
    jacobian = ismissing(DF!) ? Jacobian{T}(F, n; mode = mode) : Jacobian{T}(DF!, n; mode = :function)
    cache = NewtonSolverCache(x, y)
    linear_solver = LinearSolver(y; linear_solver = :julia)
    ls = LinesearchState(linesearch; T = T)
    options = Options(T, config)
    NewtonSolver{T, AT, typeof(objective), typeof(cache.J), typeof(jacobian), typeof(linear_solver), typeof(ls)}(x, objective, jacobian, linear_solver, ls, cache, options)
end

"""
    solver_step!(s)

Compute one Newton step for `f` based on the Jacobian `jacobian!`.
"""
function solver_step!(x::Union{AbstractVector{T}, T}, obj::AbstractObjective, jacobian!, s::NewtonSolver{T}) where {T}
    # update Newton solver cache
    update!(s, x)

    _compute_jacobian!(s, x, jacobian!)

    # factorize the jacobian stored in `s` and save the factorized matrix in the corresponding linear solver.
    factorize!(linearsolver(s), jacobian(cache(s)))

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

_compute_jacobian!(s::AbstractNewtonSolver, x, jacobian!::Callable) = compute_jacobian!(s, x, jacobian!; mode = :function)
_compute_jacobian!(s::AbstractNewtonSolver, x, jacobian!::Jacobian) = compute_jacobian!(s, x, jacobian!)