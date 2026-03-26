module SimpleSolvers

using Distances
using ForwardDiff
using StaticArrays
using LinearAlgebra
using Printf

import Base.minimum
import Base.Callable
import GeometricBase: AbstractProblem, AbstractSolver, AbstractSolverState
import GeometricBase: NullParameters, OptionalParameters, SolverMethod, SolverState
import GeometricBase: update!, value
import GeometricBase.Utils: L2norm, l2norm

include("utils.jl")
include("base/realcomplex.jl")
include("base/initialize.jl")

export update!
export solve!, solve
export config, result, state, status
export algorithm, problem
export solution, minimizer, minimum

export SolverMethod, SolverState
export BracketingMethod
export DirectMethod, IterativeMethod
export NonlinearMethod, PicardMethod, LinesearchMethod

export NewtonMethod, Newton

include("base/methods.jl")

export Gradient,
    GradientAutodiff,
    GradientFiniteDifferences,
    GradientFunction

export check_gradient

include("base/gradient.jl")

export LinesearchProblem

export value,
    gradient,
    derivative,
    hessian

include("linesearch/linesearch_problem.jl")

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

export Linesearch
export Backtracking,
    Bisection,
    Quadratic,
    BierlaireQuadratic,
    Static

include("linesearch/linesearch.jl")
include("linesearch/backtracking/backtracking_condition.jl")
include("linesearch/backtracking/curvature_condition.jl")
include("linesearch/backtracking/sufficient_decrease_condition.jl")
include("linesearch/backtracking.jl")
include("linesearch/bisection.jl")
include("linesearch/quadratic.jl")
include("linesearch/quadratic_bierlaire.jl")
include("linesearch/static.jl")

export NonlinearProblem, NonlinearSolver, NonlinearSolverException, NonlinearSolverState,
    NewtonSolver, QuasiNewtonSolver, assess_convergence, solve!, NewtonMethod, QuasiNewtonMethod

export PicardSolver

export DogLegSolver, DogLeg

include("nonlinear/nonlinear_problem.jl")
include("nonlinear/nonlinear_solver_state.jl")
include("nonlinear/nonlinear_solver_cache.jl")
include("nonlinear/nonlinear_solver_status.jl")
include("nonlinear/nonlinear_solver.jl")
include("nonlinear/newton_solver.jl")
include("nonlinear/picard_solver.jl")
include("nonlinear/dogleg_cache.jl")
include("nonlinear/dogleg_solver.jl")
include("nonlinear/linesearch_problem.jl")

SolverState(s::NonlinearSolver) = NonlinearSolverState(solution(cache(s)), value(cache(s)))

end
