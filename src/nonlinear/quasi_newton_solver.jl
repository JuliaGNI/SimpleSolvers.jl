
struct QuasiNewtonSolver{T, AT, FT, TJ, TL, TS <: LinesearchState} <: AbstractNewtonSolver{T,AT}
    @newton_solver_variables

    refactorize::Int

    function QuasiNewtonSolver{T,AT,FT,TJ,TL,TS}(x, y, F!, Jparams, linear_solver, linesearch, cache, config, refactorize) where {T,AT,FT,TJ,TL,TS}
        J  = zero(linear_solver.A)

        status = NonlinearSolverStatus{T}(length(x))

        new(x, y, J, F!, Jparams, linear_solver, linesearch, cache, config, status, refactorize)
    end
end

function QuasiNewtonSolver(x::AbstractVector{T}, y::AbstractVector{T}, F!; J! = nothing, linesearch = Static(), config = Options(), refactorize=5) where {T}
    # ArmijoQuadraticState(F!, x, y)
    n = length(y)
    Jparams = JacobianParameters{T}(J!, F!, n)
    cache = NewtonSolverCache(x, y)
    linear_solver = LinearSolver(y)
    ls = LinesearchState(linesearch, linesearch_objective(F!, cache))

    QuasiNewtonSolver{T, typeof(x), typeof(F!), typeof(Jparams), typeof(linear_solver), typeof(ls)}(x, y, F!, Jparams, linear_solver, ls, cache, config, refactorize)
end

function solver_step!(s::QuasiNewtonSolver{T}) where {T}
    # compute Jacobian and factorize
    if mod(s.status.i-1, s.refactorize) == 0
        compute_jacobian!(s)
        s.linear.A .= s.J
        factorize!(s.linear)
    end

    # copy previous solution
    s.cache.x₀ .= s.x

    # b = - y₀
    s.linear.b .= -s.y

    # solve J δx = -f(x)
    solve!(s.linear)

    # δx = b
    s.cache.δx .= s.linear.b

    # apply line search
    s.linesearch(s.x, s.cache.δx)

    # compute residual
    s.F!(s.y, s.x)
    residual!(status(s), s.x, s.y)
end
