
struct NewtonSolver{T, AT, TJ, TL, TS <: LinesearchState} <: AbstractNewtonSolver{T,AT}
    @newton_solver_variables

    function NewtonSolver{T,AT,TJ,TL,TS}(x, jacobian, linear_solver, linesearch, cache, config) where {T,AT,TJ,TL,TS}
        status = NonlinearSolverStatus{T}(length(x))
        new(jacobian, linear_solver, linesearch, cache, config, status)
    end
end

function NewtonSolver(x::AbstractVector{T}, y::AbstractVector{T}, F!; J! = missing, linesearch = Backtracking(), config = Options()) where {T}
    n = length(y)
    jacobian = Jacobian{T}(J!, F!, n)
    cache = NewtonSolverCache(x, y)
    linear_solver = LinearSolver(y)
    ls = LinesearchState(linesearch)
    NewtonSolver{T, typeof(x), typeof(jacobian), typeof(linear_solver), typeof(ls)}(x, jacobian, linear_solver, ls, cache, config)
end

function solver_step!(x, f, forj, s::NewtonSolver{T}) where {T}
    # shortcuts
    rhs = s.cache.rhs
    δ = s.cache.δx

    # update Newton solver cache
    update!(s, x)

    # compute Jacobian
    compute_jacobian!(s, x, forj)

    # factorize linear solver
    factorize!(s.linear, s.cache.J)

    # compute RHS
    f(rhs, x)
    rmul!(rhs, -1)

    # solve J δx = -f(x)
    ldiv!(δ, s.linear, rhs)

    # apply line search
    α = s.linesearch(linesearch_objective(f, jacobian(s), cache(s)))
    x .+= α .* δ
end
