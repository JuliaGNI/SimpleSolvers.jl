module SimpleSolvers

    using LinearAlgebra

    include("config.jl")
    include("utils.jl")

    export LinearSolver, LUSolver, LUSolverLAPACK,
           factorize!, solve!

    include("linear/linear_solvers.jl")
    include("linear/lu_solver.jl")
    include("linear/lu_solver_lapack.jl")

    export JacobianParameters, JacobianParametersAD, JacobianParametersFD,
           JacobianParametersUser, getJacobianParameters,
           computeJacobian, computeJacobianAD, computeJacobianFD

    export computeJacobian, check_jacobian, print_jacobian

    export NonlinearSolver, NonlinearSolverException,
           AbstractNewtonSolver, NLsolveNewton, NewtonSolver, QuasiNewtonSolver,
           params, status,
           residual_initial!, residual_absolute!, residual_relative!,
           print_solver_status, check_solver_converged, check_solver_status,
           get_solver_status, get_solver_status!,
           solve!

    include("nonlinear/nonlinear_solvers.jl")
    include("nonlinear/jacobian.jl")
    include("nonlinear/abstract_newton_solver.jl")
    include("nonlinear/newton_solver.jl")
    include("nonlinear/quasi_newton_solver.jl")
    include("nonlinear/nlsolve_newton.jl")


    function __init__()
        default_params = (
            (:ls_solver, :julia),
            (:nls_atol,  2eps()),
            (:nls_rtol,  2eps()),
            (:nls_stol,  2eps()),
            (:nls_atol_break,  1E3),
            (:nls_rtol_break,  1E3),
            (:nls_stol_break,  1E3),
            (:nls_nmax,  10000),
            (:nls_nmin,  0),
            (:nls_nwarn, 100),
            (:nls_solver, NewtonSolver),
            (:jacobian_autodiff, true),
            (:jacobian_fd_ϵ, 8sqrt(eps())),
            (:quasi_newton_refactorize, 5),
            (:linesearch_nmax, 50),
            (:linesearch_armijo_λ₀, 1.0),
            (:linesearch_armijo_σ₀, 0.1),
            (:linesearch_armijo_σ₁, 0.5),
            (:linesearch_armijo_ϵ,  0.5),
        )

        for param in default_params
            add_config(param...)
        end
    end

end
