
struct QuasiNewtonSolver{T, FT, TJ, TL, TS <: LineSearch} <: AbstractNewtonSolver{T}
    @newton_solver_variables

    refactorize::Int

    function QuasiNewtonSolver{T,FT,TJ,TL,TS}(x, y, F!, Jparams, linear_solver, linesearch, config, refactorize) where {T,FT,TJ,TL,TS}
        J  = zero(linear_solver.A)
        x₀ = zero(x)
        x₁ = zero(x)
        y₀ = zero(y)
        y₁ = zero(y)
        δx = zero(x)
        δy = zero(y)

        status = NonlinearSolverStatus{T}(length(x))

        new(x, y, J, x₀, x₁, y₀, y₁, δx, δy, F!, Jparams, linear_solver, linesearch, config, status, refactorize)
    end
end

function QuasiNewtonSolver(x::AbstractVector{T}, y::AbstractVector{T}, F!; J! = nothing, linesearch = ArmijoQuadratic(F!, x, y), config = Options(), refactorize=5) where {T}
    n = length(y)
    Jparams = JacobianParameters{T}(J!, F!, n)
    linear_solver = LinearSolver(y)
    QuasiNewtonSolver{T, typeof(F!), typeof(Jparams), typeof(linear_solver), typeof(linesearch)}(x, y, F!, Jparams, linear_solver, linesearch, config, refactorize)
end

function solver_step!(s::QuasiNewtonSolver{T}) where {T}
    # compute Jacobian and factorize
    if mod(s.status.i-1, s.refactorize) == 0
        compute_jacobian!(s)
        s.linear.A .= s.J
        factorize!(s.linear)
    end

    # copy previous solution
    s.x₀ .= s.x

    # b = - y₀
    s.linear.b .= -s.y

    # solve J δx = -f(x)
    solve!(s.linear)

    # δx = b
    s.δx .= s.linear.b

    # x₁ = x₀ + δx
    s.x₁ .= s.x₀ .+ s.δx
    
    # apply line search
    solve!(s.x, s.y, s.J, s.x₀, s.x₁, s.ls)

    # compute residual
    s.F!(s.y, s.x)
    residual!(status(s), s.x, s.y)
end
