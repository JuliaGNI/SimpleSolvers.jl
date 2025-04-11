"""
    NewtonSolver

A struct that comprises all Newton solvers. Those typically differ in the way the Jacobian is computed.
"""
struct NewtonSolver{T, AT, JT, TJ, TL, TLS <: LinesearchState, TST <: NonlinearSolverStatus{T}} <: AbstractNewtonSolver{T,AT}
    @newton_solver_variables

    function NewtonSolver{T,AT,JT,TJ,TL,TS}(x, jacobian, linear_solver, linesearch, cache, config) where {T,AT,JT,TJ,TL,TS}
        status = NonlinearSolverStatus{T}(length(x))
        new{T,AT,JT,TJ,TL,TS, typeof(status)}(jacobian, linear_solver, linesearch, cache, config, status)
    end
end

function NewtonSolver(x::AT, y::AT; F = missing, DF! = missing, linesearch = Backtracking(), config = Options(), mode = :autodiff) where {T, AT <: AbstractVector{T}}
    n = length(y)
    !ismissing(F) || !ismissing(DF!) || error("You have to either provide F or DF!.")
    jacobian = (ismissing(DF!) && !ismissing(F)) ? Jacobian{T}(F, n; mode = mode) : Jacobian{T}(DF!, n; mode = :function)
    cache = NewtonSolverCache(x, y)
    linear_solver = LinearSolver(y)
    ls = LinesearchState(linesearch; T = T)
    options = Options(T, config)
    NewtonSolver{T, AT, typeof(cache.J), typeof(jacobian), typeof(linear_solver), typeof(ls)}(x, jacobian, linear_solver, ls, cache, options)
end

"""
    solver_step!(x, f, jacobian!, s)

Compute one Newton step for `f` based on the Jacobian `jacobian!`.
"""
function solver_step!(x::Union{AbstractVector{T}, T}, f, jacobian!, s::NewtonSolver{T}) where {T}
    # shortcuts
    rhs = cache(s).rhs
    δ = cache(s).δx

    # update Newton solver cache
    update!(s, x)

    # compute Jacobian
    compute_jacobian!(s, x, jacobian!)

    # factorize linear solver
    factorize!(linearsolver(s), cache(s).J)

    # compute RHS (f is an in-place function)
    f(rhs, x)
    rmul!(rhs, -1)

    # solve J δx = -f(x)
    ldiv!(δ, linearsolver(s), rhs)

    # apply line search
    α = s.linesearch(linesearch_objective(f, jacobian(s), cache(s)))
    x .+= α .* δ
end
