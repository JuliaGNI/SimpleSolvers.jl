abstract type LinearMethod <: SolverMethod end
"A supertype collecting all nonlinear methods, including `NewtonMethod`s."
abstract type NonlinearMethod <: SolverMethod end

abstract type NonlinearSolverMethod <: SolverMethod end

# abstract type DirectMethod <: LinearMethod end
# abstract type IterativeMethod <: LinearMethod end


"""
    NewtonMethod(refactorize)

Make an instance of a *quasi Newton solver* based on an integer *refactorize* that determines how often the rhs is refactored.
"""
struct NewtonMethod <: NonlinearSolverMethod
    refactorize::Int
end

NewtonMethod() = NewtonMethod(1)

"""
The default number of iterations before the [`Jacobian`](@ref) is refactored in the [`QuasiNewtonSolver`](@ref)
"""
const DEFAULT_ITERATIONS_QUASI_NEWTON_SOLVER = 5

QuasiNewtonMethod() = NewtonMethod(DEFAULT_ITERATIONS_QUASI_NEWTON_SOLVER)

"""
    PicardMethod()

Make an instance of a *Picard solver* (fixed point iterator).
"""
struct PicardMethod <: NonlinearSolverMethod end

"""
    OptimizerMethod <: SolverMethod

The `OptimizerMethod` is used in [`Optimizer`](@ref) and determines the algorithm that is used.
"""
abstract type OptimizerMethod <: SolverMethod end

"""
    QuasiNewtonOptimizerMethod <: OptimizerMethod

Includes [`BFGS`](@ref) and [`DFP`](@ref).
"""
abstract type QuasiNewtonOptimizerMethod <: OptimizerMethod end

struct Newton <: OptimizerMethod end

"""
Algorithm taken from [nocedal2006numerical](@cite).
"""
struct DFP <: QuasiNewtonOptimizerMethod end

"""
Algorithm taken from [nocedal2006numerical](@cite).
"""
struct BFGS <: QuasiNewtonOptimizerMethod end

Base.show(io::IO, alg::Newton) = print(io, "Newton")
Base.show(io::IO, alg::DFP) = print(io, "DFP")
Base.show(io::IO, alg::BFGS) = print(io, "BFGS")