
struct QuasiNewtonSolver{T, AT, FT, TJ, TL, TS <: LinesearchState} <: AbstractNewtonSolver{T,AT}
    @newton_solver_variables

    refactorize::Int

    function QuasiNewtonSolver{T,AT,FT,TJ,TL,TS}(x, y, F!, Jparams, linear_solver, linesearch, cache, config, refactorize) where {T,AT,FT,TJ,TL,TS}
        status = NonlinearSolverStatus{T}(length(x))

        new(x, y, F!, Jparams, linear_solver, linesearch, cache, config, status, refactorize)
    end
end

function QuasiNewtonSolver(x::AbstractVector{T}, y::AbstractVector{T}, F!; J! = nothing, linesearch = Static(), config = Options(), refactorize=5) where {T}
    n = length(y)
    Jparams = JacobianParameters{T}(J!, F!, n)
    cache = NewtonSolverCache(x, y)
    linear_solver = LinearSolver(y)
    ls = LinesearchState(linesearch, linesearch_objective(F!, Jparams, cache))

    QuasiNewtonSolver{T, typeof(x), typeof(F!), typeof(Jparams), typeof(linear_solver), typeof(ls)}(x, y, F!, Jparams, linear_solver, ls, cache, config, refactorize)
end

function solver_step!(s::QuasiNewtonSolver{T}) where {T}
    # shortcuts
    x = s.x
    y = s.y
    δ = s.cache.δx

    # compute Jacobian and factorize
    if mod(s.status.i-1, s.refactorize) == 0
        compute_jacobian!(s)
        factorize!(s.linear, s.cache.J)
    end

    # copy previous solution
    s.cache.x₀ .= x

    # solve J δx = -f(x)
    rmul!(y, -1)
    ldiv!(δ, s.linear, y)

    # apply line search
    α = s.linesearch()
    x .+= α .* δ

    # compute residual
    s.F!(y, x)
    residual!(status(s), x, y)
end
