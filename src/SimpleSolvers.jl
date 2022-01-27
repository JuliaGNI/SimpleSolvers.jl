module SimpleSolvers

    using ForwardDiff
    using LinearAlgebra
    using Printf

    include("utils.jl")

    export solve!
    export config, status
    

    export Options

    include("base/options.jl")

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
           residual_initial!, residual_absolute!, residual_relative!,
           assess_convergence, assess_convergence!,
           print_status, check_solver_status,
           get_solver_status, get_solver_status!,
           solve!

    include("nonlinear/nonlinear_solver_status.jl")
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
           HessianDFP

    include("optimization/optimizer_status.jl")
    include("optimization/optimizer.jl")
    include("optimization/hessian_bfgs.jl")
    include("optimization/hessian_dfp.jl")
    include("optimization/abstract_newton_optimizer.jl")
    include("optimization/quasi_newton_optimizer.jl")

end
