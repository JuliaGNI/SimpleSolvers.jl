module SimpleSolvers

    using ForwardDiff
    using LinearAlgebra

    import Base: Callable

    include("config.jl")
    include("utils.jl")

    export solve!


    export GradientParameters,
           GradientParametersAD,
           GradientParametersFD,
           GradientParametersUser
           
    export compute_gradient,
           compute_gradient!,
           compute_gradient_ad!,
           compute_gradient_fd!

    export check_gradient,
           print_gradient

    include("base/gradient.jl")

    export HessianParameters,
           HessianParametersAD,
           HessianParametersUser

    export compute_hessian,
           compute_hessian!,
           compute_hessian_ad!

    export check_hessian,
           print_hessian

    include("base/hessian.jl")

    export JacobianParameters,
           JacobianParametersAD,
           JacobianParametersFD,
           JacobianParametersUser

    export compute_jacobian!,
           compute_jacobian_ad!,
           compute_jacobian_fd!

    export check_jacobian,
           print_jacobian

    include("base/jacobian.jl")

    export UnivariateObjective,
           MultivariateObjective

    export value, value!, value!!,
           derivative, derivative!, derivative!!,
           gradient, gradient!, gradient!!,
           hessian, hessian!, hessian!!,
           d_calls, f_calls, g_calls, h_calls

    include("base/objectives.jl")

    export LinearSolver, LUSolver, LUSolverLAPACK,
           factorize!

    include("linear/linear_solvers.jl")
    include("linear/lu_solver.jl")
    include("linear/lu_solver_lapack.jl")

    export bracket_minimum

    include("bracketing/bracketing.jl")
    include("bracketing/bracket_minimum.jl")

    export LineSearch, NoLineSearch
    export Armijo, armijo,
           ArmijoQuadratic, armijo_quadratic,
           Bisection, bisection

    include("linesearch/linesearch.jl")
    include("linesearch/nolinesearch.jl")
    include("linesearch/armijo.jl")
    include("linesearch/armijo_quadratic.jl")
    include("linesearch/bisection.jl")

    export NonlinearSolver, NonlinearSolverException,
           AbstractNewtonSolver, NLsolveNewton, NewtonSolver, QuasiNewtonSolver,
           params, status,
           residual_initial!, residual_absolute!, residual_relative!,
           print_solver_status, check_solver_converged, check_solver_status,
           get_solver_status, get_solver_status!,
           solve!

    include("nonlinear/nonlinear_solver.jl")
    include("nonlinear/abstract_newton_solver.jl")
    include("nonlinear/newton_solver.jl")
    include("nonlinear/quasi_newton_solver.jl")
    include("nonlinear/nlsolve_newton.jl")

    export Optimizer,
           QuasiNewtonOptimizer,
           BFGSOptimizer,
           DFPOptimizer,
           HessianBFGS,
           HessianDFP,
           initialize!

    include("optimization/optimizer.jl")
    include("optimization/hessian_bfgs.jl")
    include("optimization/hessian_dfp.jl")
    include("optimization/quasi_newton_optimizer.jl")


    function __init__()
        default_params = (
            (:verbosity, 1),
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
