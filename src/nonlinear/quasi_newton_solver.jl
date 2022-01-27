
struct QuasiNewtonSolver{T, AT, FT, TJ, TL, TS <: LineSearch} <: AbstractNewtonSolver{T,AT}
    @newton_solver_variables

    refactorize::Int

    function QuasiNewtonSolver{T,AT,FT,TJ,TL,TS}(x, y, F!, Jparams, linear_solver, linesearch, cache, config, refactorize) where {T,AT,FT,TJ,TL,TS}
        J  = zero(linear_solver.A)

        status = NonlinearSolverStatus{T}(length(x))

        new(x, y, J, F!, Jparams, linear_solver, linesearch, cache, config, status, refactorize)
    end
end

function QuasiNewtonSolver(x::AbstractVector{T}, y::AbstractVector{T}, F!; J! = nothing, linesearch = ArmijoQuadratic(F!, x, y), config = Options(), refactorize=5) where {T}
    n = length(y)
    Jparams = JacobianParameters{T}(J!, F!, n)
    linear_solver = LinearSolver(y)

    cache = NewtonSolverCache(x, y)

    QuasiNewtonSolver{T, typeof(x), typeof(F!), typeof(Jparams), typeof(linear_solver), typeof(linesearch)}(x, y, F!, Jparams, linear_solver, linesearch, cache, config, refactorize)
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

    # x₁ = x₀ + δx
    s.cache.x₁ .= s.cache.x₀ .+ s.cache.δx
    
    # apply line search
    solve!(s.x, s.y, s.J, s.cache.x₀, s.cache.x₁, s.linesearch)

    # x₁ = x₀ + δx
    s.x .= s.cache.x₀ .+ s.cache.δx
    
    # compute residual
    s.F!(s.y, s.x)
    residual!(status(s), s.x, s.y)
end
