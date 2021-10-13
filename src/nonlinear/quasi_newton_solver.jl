
struct QuasiNewtonSolver{T, FT, TJ, TL, TS <: LineSearch} <: AbstractNewtonSolver{T}
    @newton_solver_variables
    tx::Vector{T}
    ty::Vector{T}

    refactorize::Int

    function QuasiNewtonSolver{T,FT,TJ,TL,TS}(x, Fparams, Jparams, linear_solver, line_search) where {T,FT,TJ,TL,TS}
        J  = zero(linear_solver.A)
        x₀ = zero(x)
        x₁ = zero(x)
        y₁ = zero(x)
        y₀ = zero(x)
        δx = zero(x)
        δy = zero(x)
        tx = zero(x)
        ty = zero(x)

        nls_params = NonlinearSolverParameters(T)
        nls_status = NonlinearSolverStatus{T}(length(x))

        new(x, J, x₀, x₁, y₀, y₁, δx, δy, Fparams, Jparams, linear_solver, line_search,
            nls_params, nls_status, tx, ty, get_config(:quasi_newton_refactorize))
    end
end


function QuasiNewtonSolver(x::AbstractVector{T}, F!::Function; J!::Union{Function,Nothing}=nothing) where {T}
    n = length(x)
    Jparams = getJacobianParameters(J!, F!, T, n)
    linear_solver = getLinearSolver(x)
    line_search = Armijo(F!, x)
    QuasiNewtonSolver{T, typeof(F!), typeof(Jparams), typeof(linear_solver), typeof(line_search)}(x, F!, Jparams, linear_solver, line_search)
end


function solve!(s::QuasiNewtonSolver{T}; n::Int=0) where {T}
    local nmax::Int = n > 0 ? nmax = n : s.params.nmax
    local refactorize::Int = s.refactorize

    s.F!(s.x, s.y₀)
    residual_initial!(s.status, s.x, s.y₀)
    s.status.i = 0

    if s.status.rₐ ≥ s.params.atol² || n > 0 || s.params.nmin > 0
        computeJacobian(s)
        s.linear.A .= s.J
        factorize!(s.linear)

        for s.status.i in 1:nmax
            # copy previous solution
            s.x₀ .= s.x

            # b = - y₀
            s.linear.b .= -s.y₀

            # solve J δx = -f(x)
            solve!(s.linear)

            # δx = b
            s.δx .= s.linear.b

            # apply line search
            solve!(s.x, s.δx, s.x₀, s.y₀, s.J, s.ls)

            # compute residual
            s.F!(s.x, s.y₀)
            residual!(s.status, s.δx, s.x, s.y₀)

            if check_solver_converged(s.status, s.params) && s.status.i ≥ s.params.nmin && !(n > 0)
                if s.params.nwarn > 0 && s.status.i ≥ s.params.nwarn
                    println("WARNING: Quasi-Newton Solver took ", s.status.i, " iterations.")
                end
                break
            end

            if mod(s.status.i, refactorize) == 0
                computeJacobian(s)
                s.linear.A .= s.J
                factorize!(s.linear)
            end
        end
    end
end
