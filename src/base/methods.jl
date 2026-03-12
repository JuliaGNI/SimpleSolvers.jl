"A supertype collecting all nonlinear methods, including `NewtonMethod`s."
abstract type NonlinearMethod <: SolverMethod end

abstract type NonlinearSolverMethod <: SolverMethod end

"""
    NewtonMethod(refactorize)

Make an instance of a *quasi Newton solver* based on an integer *refactorize* that determines how often the rhs is refactored.
"""
struct NewtonMethod{QT} <: NonlinearSolverMethod
    refactorize::Int

    NewtonMethod() = new{true}(1)

    NewtonMethod{false}(refactorize=DEFAULT_ITERATIONS_QUASI_NEWTON_SOLVER) = new{false}(refactorize)
end

"""
The default number of iterations before the [`Jacobian`](@ref) is refactored in the [`QuasiNewtonSolver`](@ref)
"""
const DEFAULT_ITERATIONS_QUASI_NEWTON_SOLVER = 5

const QuasiNewtonMethod = NewtonMethod{false}

"""
    PicardMethod()

Make an instance of a *Picard solver* (fixed point iterator).
"""
struct PicardMethod <: NonlinearSolverMethod end

"""
    DogLeg()

*Powell's dogleg method* [powell1970new](@cite).
"""
struct DogLeg <: NonlinearSolverMethod end
