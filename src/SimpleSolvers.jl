module SimpleSolvers

using Distances
using ForwardDiff
using StaticArrays
using LinearAlgebra
using Printf

import Base.minimum
import Base.Callable
import GeometricBase: AbstractSolver, SolverMethod, AbstractProblem, update!, NullParameters, OptionalParameters, AbstractSolverState
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
export DirectMethod, IterativeMethod
export NonlinearMethod, PicardMethod, LinesearchMethod

export NewtonMethod, Newton, DFP, BFGS

include("base/methods.jl")
include("base/optimizer_methods.jl")

export Gradient,
    GradientAutodiff,
    GradientFiniteDifferences,
    GradientFunction

export check_gradient

include("base/gradient.jl")

export LinesearchProblem,
    OptimizerProblem

export value,
    gradient,
    derivative,
    hessian

include("optimization/optimizer_problems.jl")

export Options

include("base/options.jl")

export Hessian,
    HessianAutodiff,
    HessianFunction

export check_hessian,
    print_hessian

include("base/hessian.jl")

export Jacobian,
    JacobianAutodiff,
    JacobianFiniteDifferences,
    JacobianFunction

export check_jacobian,
    print_jacobian

include("base/jacobian.jl")


export LinearProblem, LinearSolver, LU, LUSolverLAPACK,
    factorize!, linearproblem

include("linear/linear_problem.jl")
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
include("linesearch/bierlaire_quadratic.jl")

export NonlinearProblem, NonlinearSolver, NonlinearSolverException, NonlinearSolverState,
    NewtonSolver, QuasiNewtonSolver, assess_convergence, solve!

export FixedPointIterator

include("nonlinear/nonlinear_problem.jl")
include("nonlinear/nonlinear_solver_state.jl")
include("nonlinear/nonlinear_solver_cache.jl")
include("nonlinear/nonlinear_solver_status.jl")
include("nonlinear/nonlinear_solver.jl")
include("nonlinear/fixed_point_iterator.jl")
include("nonlinear/newton_solver.jl")
include("nonlinear/linesearch_problem.jl")

export Optimizer,
    OptimizerState, isaOptimizerState,
    NewtonOptimizerState,
    NewtonOptimizer,
    BFGSOptimizer,
    DFPOptimizer,
    HessianAutodiff,
    HessianBFGS,
    HessianDFP

include("optimization/optimizer_state.jl")
include("optimization/optimizer_cache.jl")
include("optimization/optimizer_status.jl")
include("optimization/optimizer_result.jl")
include("optimization/iterative_hessians/iterative_hessians.jl")
include("optimization/iterative_hessians/bfgs/hessian_bfgs.jl")
include("optimization/iterative_hessians/dfp/hessian_dfp.jl")
include("optimization/newton_optimizer/newton_optimizer_cache.jl")
include("optimization/optimizer_linesearch_problem.jl")
include("optimization/newton_optimizer/newton_optimizer_state.jl")

include("optimization/iterative_hessians/bfgs/bfgs_state.jl")
include("optimization/iterative_hessians/dfp/dfp_state.jl")

include("optimization/iterative_hessians/bfgs/bfgs_cache.jl")
include("optimization/iterative_hessians/dfp/dfp_cache.jl")

include("optimization/optimizer.jl")

end
