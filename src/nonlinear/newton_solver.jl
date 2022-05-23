
struct NewtonSolver{T, AT, FT, TJ, TL, TS <: LinesearchState} <: AbstractNewtonSolver{T,AT}
    @newton_solver_variables

    function NewtonSolver{T,AT,FT,TJ,TL,TS}(x, y, F!, Jparams, linear_solver, linesearch, cache, config) where {T,AT,FT,TJ,TL,TS}
        status = NonlinearSolverStatus{T}(length(x))

        new(x, y, F!, Jparams, linear_solver, linesearch, cache, config, status)
    end
end

function NewtonSolver(x::AbstractVector{T}, y::AbstractVector{T}, F!; J! = nothing, linesearch = Backtracking(), config = Options()) where {T}
    n = length(y)
    Jparams = Jacobian{T}(J!, F!, n)
    cache = NewtonSolverCache(x, y)
    linear_solver = LinearSolver(y)
    ls = LinesearchState(linesearch, linesearch_objective(F!, Jparams, cache))
    NewtonSolver{T, typeof(x), typeof(F!), typeof(Jparams), typeof(linear_solver), typeof(ls)}(x, y, F!, Jparams, linear_solver, ls, cache, config)
end

function solver_step!(s::NewtonSolver{T}) where {T}
    # shortcuts
    x = s.x
    y = s.y
    δ = s.cache.δx

    # compute Jacobian
    compute_jacobian!(s)

    # factorize linear solver
    factorize!(s.linear, s.cache.J)

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
