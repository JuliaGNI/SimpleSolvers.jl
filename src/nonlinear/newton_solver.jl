
struct NewtonSolver{T, FT, TJ, TL, TS <: LineSearch} <: AbstractNewtonSolver{T}
    @newton_solver_variables

    function NewtonSolver{T,FT,TJ,TL,TS}(x, F!, Jparams, linear_solver, line_search) where {T,FT,TJ,TL,TS}
        J  = zero(linear_solver.A)
        x₀ = zero(x)
        x₁ = zero(x)
        y₁ = zero(x)
        y₀ = zero(x)
        δx = zero(x)
        δy = zero(x)

        nls_params = NonlinearSolverParameters(T)
        nls_status = NonlinearSolverStatus{T}(length(x))

        new(x, J, x₀, x₁, y₀, y₁, δx, δy, F!, Jparams, linear_solver, line_search,
            nls_params, nls_status)
    end
end


function NewtonSolver(x::AbstractVector{T}, F!::Function; J!::Union{Function,Nothing}=nothing, linesearch=NoLineSearch()) where {T}
    n = length(x)
    Jparams = getJacobianParameters(J!, F!, T, n)
    linear_solver = getLinearSolver(x)
    NewtonSolver{T, typeof(F!), typeof(Jparams), typeof(linear_solver), typeof(linesearch)}(x, F!, Jparams, linear_solver, linesearch)
end


function solve!(s::NewtonSolver{T}; n::Int=0) where {T}
    local nmax::Int = n > 0 ? nmax = n : s.params.nmax

    s.F!(s.x, s.y₀)
    residual_initial!(s.status, s.x, s.y₀)
    s.status.i  = 0

    if s.status.rₐ ≥ s.params.atol²
        for s.status.i = 1:nmax
            computeJacobian(s)

            # copy Jacobian into linear solver
            s.linear.A .= s.J

            # b = - y₀
            s.linear.b .= -s.y₀

            # factorize linear solver
            factorize!(s.linear)

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
                break
            end
        end
    end
end
