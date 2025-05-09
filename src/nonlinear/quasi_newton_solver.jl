
struct QuasiNewtonSolver{T, AT, OT, JT, TJ <: Jacobian, TL, TLS <: LinesearchState, TST <: NonlinearSolverStatus{T}} <: AbstractNewtonSolver{T,AT}
    objective::OT
    
    jacobian::TJ

    linear::TL
    linesearch::TLS

    cache::NewtonSolverCache{T,AT,JT}
    config::Options{T}
    status::TST

    refactorize::Int

    function QuasiNewtonSolver{T, OT, AT, JT, TJ, TL, TS}(x, objective, jacobian, linear_solver, linesearch, cache, config, refactorize) where {T, AT, OT, JT, TJ, TL, TS}
        status = NonlinearSolverStatus{T}(length(x))
        new{T, AT, OT, JT,TJ,TL,TS, typeof(status)}(objective, jacobian, linear_solver, linesearch, cache, config, status, refactorize)
    end
end

function QuasiNewtonSolver(x::AT, y::AT; F = missing, DF! = missing, linesearch = Backtracking(), config = Options(), refactorize=5, mode = :autodiff) where {T, AT <: AbstractVector{T}}
    n = length(y)
    !ismissing(F) || error("You have to provide an F.")
    objective = MultivariateObjective(F, x)
    jacobian = ismissing(DF!) ? Jacobian{T}(F, n; mode = mode) : Jacobian{T}(DF!, n; mode = :function)
    cache = NewtonSolverCache(x, y)
    linear_solver = LinearSolver(y)
    ls = LinesearchState(linesearch; T = T)
    options = Options(T, config)
    QuasiNewtonSolver{T, typeof(objective), AT, typeof(cache.J), typeof(jacobian), typeof(linear_solver), typeof(ls)}(x, objective, jacobian, linear_solver, ls, cache, options, refactorize)
end

function solver_step!(x, obj::AbstractObjective, jacobian!, s::QuasiNewtonSolver)
    # update Newton solver cache
    update!(s, x)

    # compute Jacobian and factorize
    if mod(status(s).i-1, s.refactorize) == 0
        _compute_jacobian!(s, x, jacobian!)
        factorize!(linearsolver(s), cache(s).J)
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
