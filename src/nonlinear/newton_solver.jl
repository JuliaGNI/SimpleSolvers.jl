
struct NewtonSolver{T, AT, FT, TJ, TL, TS <: LinesearchState} <: AbstractNewtonSolver{T,AT}
    @newton_solver_variables

    function NewtonSolver{T,AT,FT,TJ,TL,TS}(x, y, F!, Jparams, linear_solver, linesearch, cache, config) where {T,AT,FT,TJ,TL,TS}
        J  = zero(linear_solver.A)

        status = NonlinearSolverStatus{T}(length(x))

        new(x, y, J, F!, Jparams, linear_solver, linesearch, cache, config, status)
    end
end

function NewtonSolver(x::AbstractVector{T}, y::AbstractVector{T}, F!; J! = nothing, linesearch = Static(), config = Options()) where {T}
    n = length(y)
    Jparams = JacobianParameters{T}(J!, F!, n)
    cache = NewtonSolverCache(x, y)
    linear_solver = LinearSolver(y)
    ls = LinesearchState(linesearch, linesearch_objective(F!, cache))
    NewtonSolver{T, typeof(x), typeof(F!), typeof(Jparams), typeof(linear_solver), typeof(ls)}(x, y, F!, Jparams, linear_solver, ls, cache, config)
end

function solver_step!(s::NewtonSolver{T}) where {T}
    # compute Jacobian
    compute_jacobian!(s)

    # copy Jacobian into linear solver
    s.linear.A .= s.J

    # factorize linear solver
    factorize!(s.linear)

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
    α = s.linesearch(1.0)
    s.x .= s.cache.x₀ .+ α .* s.cache.δx
    # solve!(s.x, s.y, s.J, s.cache.x₀, s.cache.x₁, s.linesearch)

    # compute residual
    s.F!(s.y, s.x)
    residual!(status(s), s.x, s.y)
end
