
struct NewtonSolver{T, AT, FT, TJ, TL, TS <: LinesearchState} <: AbstractNewtonSolver{T,AT}
    @newton_solver_variables

    function NewtonSolver{T,AT,FT,TJ,TL,TS}(x, F!, Jparams, linear_solver, linesearch, cache, config) where {T,AT,FT,TJ,TL,TS}
        status = NonlinearSolverStatus{T}(F!, length(x))

        new(F!, Jparams, linear_solver, linesearch, cache, config, status)
    end
end

function NewtonSolver(x::AbstractVector{T}, y::AbstractVector{T}, F!; J! = nothing, linesearch = Backtracking(), config = Options()) where {T}
    n = length(y)
    Jparams = Jacobian{T}(J!, F!, n)
    cache = NewtonSolverCache(x, y)
    linear_solver = LinearSolver(y)
    ls = LinesearchState(linesearch, linesearch_objective(F!, Jparams, cache))
    NewtonSolver{T, typeof(x), typeof(F!), typeof(Jparams), typeof(linear_solver), typeof(ls)}(x, F!, Jparams, linear_solver, ls, cache, config)
end

function solver_step!(x, s::NewtonSolver{T}) where {T}
    # shortcuts
    rhs = s.cache.rhs
    δ = s.cache.δx

    # update Newton solver cache
    update!(s, x)

    # compute Jacobian
    compute_jacobian!(s, x)

    # factorize linear solver
    factorize!(s.linear, s.cache.J)

    # compute RHS
    s.F!(rhs, x)
    rmul!(rhs, -1)

    # solve J δx = -f(x)
    ldiv!(δ, s.linear, rhs)

    # apply line search
    α = s.linesearch()
    x .+= α .* δ
end
