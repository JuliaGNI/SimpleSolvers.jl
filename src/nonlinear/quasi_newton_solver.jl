
struct QuasiNewtonSolver{T, AT, JT, TJ, TL, TLS <: LinesearchState, TST <: NonlinearSolverStatus{T}} <: AbstractNewtonSolver{T,AT}
    @newton_solver_variables

    refactorize::Int

    function QuasiNewtonSolver{T,AT,JT,TJ,TL,TS}(x, jacobian, linear_solver, linesearch, cache, config, refactorize) where {T,AT,JT,TJ,TL,TS}
        status = NonlinearSolverStatus{T}(length(x))
        new{T,AT,JT,TJ,TL,TS, typeof(status)}(jacobian, linear_solver, linesearch, cache, config, status, refactorize)
    end
end

function QuasiNewtonSolver(x::AT, y::AT; J! = missing, linesearch = Backtracking(), config = Options(), refactorize=5) where {T, AT <: AbstractVector{T}}
    n = length(y)
    jacobian = Jacobian{T}(J!, n)
    cache = NewtonSolverCache(x, y)
    linear_solver = LinearSolver(y)
    ls = LinesearchState(linesearch)
    options = Options(T, config)
    QuasiNewtonSolver{T, AT, typeof(cache.J), typeof(jacobian), typeof(linear_solver), typeof(ls)}(x, jacobian, linear_solver, ls, cache, options, refactorize)
end

function solver_step!(x, f, forj, s::QuasiNewtonSolver)
    # shortcuts
    rhs = cache(s).rhs
    δ = cache(s).δx

    # update Newton solver cache
    update!(s, x)

    # compute Jacobian and factorize
    if mod(status(s).i-1, s.refactorize) == 0
        compute_jacobian!(s, x, forj)
        factorize!(linearsolver(s), cache(s).J)
    end

    # compute RHS
    f(rhs, x)
    rmul!(rhs, -1)

    # solve J δx = -f(x)
    ldiv!(δ, linearsolver(s), rhs)

    # apply line search
    α = s.linesearch(linesearch_objective(f, jacobian(s), cache(s)))
    x .+= α .* δ
end
