
struct QuasiNewtonSolver{T, AT, TJ, TL, TS <: LinesearchState} <: AbstractNewtonSolver{T,AT}
    @newton_solver_variables

    refactorize::Int

    function QuasiNewtonSolver{T,AT,TJ,TL,TS}(x, jacobian, linear_solver, linesearch, cache, config, refactorize) where {T,AT,TJ,TL,TS}
        status = NonlinearSolverStatus{T}(length(x))
        new(jacobian, linear_solver, linesearch, cache, config, status, refactorize)
    end
end

function QuasiNewtonSolver(x::AT, y::AT; J! = missing, linesearch = Backtracking(), config = Options(), refactorize=5) where {T, AT <: AbstractVector{T}}
    n = length(y)
    jacobian = Jacobian{T}(J!, n)
    cache = NewtonSolverCache(x, y)
    linear_solver = LinearSolver(y)
    ls = LinesearchState(linesearch)
    options = Options(T, config)
    QuasiNewtonSolver{T, AT, typeof(jacobian), typeof(linear_solver), typeof(ls)}(x, jacobian, linear_solver, ls, cache, options, refactorize)
end

function solver_step!(x, f, forj, s::QuasiNewtonSolver{T}) where {T}
    # shortcuts
    rhs = s.cache.rhs
    δ = s.cache.δx

    # update Newton solver cache
    update!(s, x)

    # compute Jacobian and factorize
    if mod(s.status.i-1, s.refactorize) == 0
        compute_jacobian!(s, x, forj)
        factorize!(s.linear, s.cache.J)
    end

    # compute RHS
    f(rhs, x)
    rmul!(rhs, -1)

    # solve J δx = -f(x)
    ldiv!(δ, s.linear, rhs)

    # apply line search
    α = s.linesearch(linesearch_objective(f, jacobian(s), cache(s)))
    x .+= α .* δ
end
