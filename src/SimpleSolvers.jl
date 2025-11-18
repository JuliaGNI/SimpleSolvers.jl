module SimpleSolvers

    using Distances
    using ForwardDiff
    using StaticArrays
    using LinearAlgebra
    using Printf

    import Base.minimum
    import Base.Callable
    import GeometricBase: AbstractSolver, SolverMethod, AbstractProblem, update!, NullParameters, OptionalParameters
    import GeometricBase: value

    include("utils.jl")
    include("base/realcomplex.jl")
    include("base/initialize.jl")

    export update!
    export solve!, solve
    export config, result, state, status
    export algorithm, problem
    export solution, minimizer, minimum
    
    export SolverMethod
    export BracketingMethod
    export LinearMethod, DirectMethod, IterativeMethod
    export NonlinearMethod, PicardMethod, LinesearchMethod

    export NewtonMethod, Newton, DFP, BFGS

    include("base/methods.jl")

    export Gradient,
           GradientAutodiff,
           GradientFiniteDifferences,
           GradientFunction
           
    export compute_gradient,
           compute_gradient!,
           compute_gradient_ad!,
           compute_gradient_fd!

    export check_gradient
    
    include("base/gradient.jl")

    export LinesearchProblem,
           OptimizerProblem

    export value, value!, value!!,
           derivative, derivative!, derivative!!,
           gradient, gradient!, gradient!!,
           hessian, hessian!, hessian!!
           
    include("base/optimizer_problems.jl")

    export Options

    include("base/options.jl")

    export Hessian,
           HessianAutodiff,
           HessianFunction

    export compute_hessian,
           compute_hessian!,
           compute_hessian_ad!

    export check_hessian,
           print_hessian

    include("base/hessian.jl")

    export Jacobian,
           JacobianAutodiff,
           JacobianFiniteDifferences,
           JacobianFunction

    export compute_jacobian!,
           compute_jacobian_ad!,
           compute_jacobian_fd!

    export check_jacobian,
           print_jacobian

    include("base/jacobian.jl")

    include("base/solver_problems.jl")

    export LinearProblem, NonlinearProblem

    export LinearSolver, LU, LUSolverLAPACK,
           factorize!, linearproblem

    include("linear/linear_solver_method.jl")
    include("linear/linear_solver_cache.jl")
    include("linear/linear_solvers.jl")
    include("linear/lu_solver.jl")
    include("linear/lu_solver_lapack.jl")

    export bracket_minimum

    include("bracketing/bracket_minimum.jl")
    include("bracketing/triple_point_finder.jl")

    export Linesearch, Static
    export Backtracking,
           Bisection,
           Quadratic,
           Quadratic2,
           BierlaireQuadratic

    include("linesearch/methods.jl")
    include("linesearch/linesearch.jl")
    include("linesearch/static.jl")
    include("linesearch/backtracking/backtracking.jl")
    include("linesearch/backtracking/condition.jl")
    include("linesearch/backtracking/sufficient_decrease_condition.jl")
    include("linesearch/backtracking/curvature_condition.jl")
    include("linesearch/bisection.jl")
    include("linesearch/quadratic.jl")
    include("linesearch/custom_quadratic.jl")
    include("linesearch/bierlaire_quadratic.jl")
    include("linesearch/dummy_linesearch.jl")

    export NonlinearSolver, NonlinearSolverException,
           AbstractNewtonSolver, NewtonSolver, QuasiNewtonSolver,
           residual_initial!, residual_absolute!, residual_relative!,
           assess_convergence, assess_convergence!,
           print_status, check_solver_status,
           get_solver_status, get_solver_status!,
           solve!

    export FixedPointIterator

    include("nonlinear/nonlinear_solver_status.jl")
    include("nonlinear/nonlinear_solver.jl")
    include("nonlinear/newton_solver_cache.jl")
    include("nonlinear/newton_solver_linesearch_problem.jl")
    include("nonlinear/fixed_point_iterator.jl")
    include("nonlinear/newton_solver.jl")

    export Optimizer,
           OptimizationAlgorithm, isaOptimizationAlgorithm,
           NewtonOptimizer,
           BFGSOptimizer,
           DFPOptimizer,
           HessianAutodiff,
           HessianBFGS,
           HessianDFP

    include("optimization/optimizer_status.jl")
    include("optimization/optimizer_result.jl")
    include("optimization/optimizer.jl")
    include("optimization/hessian_bfgs.jl")
    include("optimization/hessian_dfp.jl")
    include("optimization/newton_optimizer_cache.jl")
    include("optimization/newton_optimizer_linesearch_problem.jl")
    include("optimization/newton_optimizer_state.jl")

end
