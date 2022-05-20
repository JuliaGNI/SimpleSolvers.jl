module SimpleSolvers

    using Distances
    using ForwardDiff
    using LinearAlgebra
    using Printf

    import Base.minimum
    import Base.Callable

    include("utils.jl")

    export solve!
    export config, result, state, status
    export algorithm, objective
    export solution, minimizer, minimum
    
    export Newton, DFP, BFGS

    include("base/methods.jl")

    export UnivariateObjective,
           MultivariateObjective

    export value, value!, value!!,
           derivative, derivative!, derivative!!,
           gradient, gradient!, gradient!!,
           hessian, hessian!, hessian!!,
           d_calls, f_calls, g_calls, h_calls

    include("base/objectives.jl")

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

    export Hessian,
           HessianAD,
           HessianUser

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

    export LinearSolver, LUSolver, LUSolverLAPACK,
           factorize!

    include("linear/linear_solvers.jl")
    include("linear/lu_solver.jl")
    include("linear/lu_solver_lapack.jl")

    export bracket_minimum

    include("bracketing/bracketing.jl")
    include("bracketing/bracket_minimum.jl")

    export Linesearch, Static
    export Backtracking, backtracking,
           Bisection, bisection,
           Quadratic, quadratic

    include("linesearch/linesearch.jl")
    include("linesearch/static.jl")
    include("linesearch/backtracking.jl")
    include("linesearch/bisection.jl")
    include("linesearch/quadratic.jl")

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
           OptimizationAlgorithm,
           NewtonOptimizer,
           BFGSOptimizer,
           DFPOptimizer,
           HessianBFGS,
           HessianDFP

    include("optimization/optimizer_status.jl")
    include("optimization/optimizer_result.jl")
    include("optimization/optimizer.jl")
    include("optimization/hessian_bfgs.jl")
    include("optimization/hessian_dfp.jl")
    include("optimization/newton_optimizer.jl")

end
