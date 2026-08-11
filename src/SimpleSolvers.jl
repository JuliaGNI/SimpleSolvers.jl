module SimpleSolvers

using Distances
using ForwardDiff
using StaticArrays
using LinearAlgebra
using Printf

import LinearAlgebra: checksquare

import Base.Callable
import GeometricBase: AbstractProblem, AbstractSolver, AbstractSolverState
import GeometricBase: NullParameters, OptionalParameters, SolverMethod, SolverState
import GeometricBase: update!, value
import GeometricBase.Utils: L2norm, l2norm

include("utils.jl")

export update!
export solve!, solve
export config
export problem
export solution

export SolverMethod, SolverState
export DirectMethod
export NonlinearSolverMethod, Picard, LinesearchMethod

export Newton, QuasiNewton

export Gradient,
    GradientAutodiff,
    GradientFiniteDifferences,
    GradientFunction

export check_gradient

include("base/gradient.jl")

export LinesearchProblem

export value,
    derivative

include("linesearch/linesearch_problem.jl")

export Options

include("base/options.jl")

export Hessian,
    HessianAutodiff,
    HessianFunction

export check_hessian

include("base/hessian.jl")

export Jacobian,
    JacobianAutodiff,
    JacobianFiniteDifferences,
    JacobianFunction

export check_jacobian,
    print_jacobian

include("base/jacobian.jl")


export LinearProblem, LinearSolver, LU,
    factorize!, linearproblem

include("linear/linear_problem.jl")
include("linear/linear_solver_method.jl")
include("linear/linear_solver_cache.jl")
include("linear/linear_solvers.jl")
include("linear/lu_solver.jl")

export bracket_minimum

include("bracketing/bracket_minimum.jl")
include("bracketing/triple_point_finder.jl")

export Linesearch
export Backtracking,
    Bisection,
    Quadratic,
    BierlaireQuadratic,
    Static,
    StrongWolfe

# Types and enum values are exported; the predicates and accessors that go with them
# (`steplength`, `outcome`, `trials`, `issufficient`, `isfloor`) are not, because they are
# generic names that a package doing `using SimpleSolvers` may well want for itself. Reach them
# as `SimpleSolvers.steplength` and friends.
export LinesearchStatus, LinesearchOutcome, solve_with_status
export LINESEARCH_DECREASED,
    LINESEARCH_FLOOR,
    LINESEARCH_EXHAUSTED,
    LINESEARCH_NO_DESCENT,
    LINESEARCH_STATIONARY,
    LINESEARCH_UNKNOWN

include("linesearch/linesearch.jl")
include("linesearch/linesearch_status.jl")
include("linesearch/backtracking_condition.jl")
include("linesearch/curvature_condition.jl")
include("linesearch/sufficient_decrease_condition.jl")
include("linesearch/backtracking.jl")
include("linesearch/bisection.jl")
include("linesearch/quadratic.jl")
include("linesearch/quadratic_bierlaire.jl")
include("linesearch/static.jl")
include("linesearch/wolfe.jl")

export NonlinearProblem, NonlinearSolver, NonlinearSolverException, NonlinearSolverState,
    NewtonSolver, assess_convergence

# The mutating counterpart of the line-search `solve_with_status`: it overwrites `x`, hence the
# `!`. `solve!` and `solve` are exported above.
export solve_with_status!

# As above: the type is exported, `isconverged`/`isstalled`/`status` are not. `status` in
# particular would be hostile to export — a downstream package that does `using SimpleSolvers`
# and defines its own `status` gets a method-definition error, not a shadowing warning.
export NonlinearSolverStatus

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
