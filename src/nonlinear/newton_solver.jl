
struct NewtonSolver{T, AT, JT, TJ, TL, TLS <: LinesearchState, TST <: NonlinearSolverStatus{T}} <: AbstractNewtonSolver{T,AT}
    @newton_solver_variables

    function NewtonSolver{T,AT,JT,TJ,TL,TS}(x, jacobian, linear_solver, linesearch, cache, config) where {T,AT,JT,TJ,TL,TS}
        status = NonlinearSolverStatus{T}(length(x))
        new{T,AT,JT,TJ,TL,TS, typeof(status)}(jacobian, linear_solver, linesearch, cache, config, status)
    end
end

function NewtonSolver(x::AT, y::AT; J! = missing, linesearch = Backtracking(), config = Options()) where {T, AT <: AbstractVector{T}}
    n = length(y)
    jacobian = Jacobian{T}(J!, n)
    cache = NewtonSolverCache(x, y)
    linear_solver = LinearSolver(y)
    ls = LinesearchState(linesearch)
    options = Options(T, config)
    NewtonSolver{T, AT, typeof(cache.J), typeof(jacobian), typeof(linear_solver), typeof(ls)}(x, jacobian, linear_solver, ls, cache, options)
end

function solver_step!(x, f, forj, s::NewtonSolver)
    # shortcuts
    rhs = cache(s).rhs
    δ = cache(s).δx

    # update Newton solver cache
    update!(s, x)

    # compute Jacobian
    compute_jacobian!(s, x, forj)

    # factorize linear solver
    factorize!(linearsolver(s), cache(s).J)

    # compute RHS
    f(rhs, x)
    rmul!(rhs, -1)

    # solve J δx = -f(x)
    ldiv!(δ, linearsolver(s), rhs)

    # apply line search
    α = s.linesearch(linesearch_objective(f, jacobian(s), cache(s)))
    x .+= α .* δ
end
