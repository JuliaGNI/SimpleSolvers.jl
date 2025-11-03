abstract type LinearMethod <: SolverMethod end
"A supertype collecting all nonlinear methods, including `NewtonMethod`s."
abstract type NonlinearMethod <: SolverMethod end

abstract type DirectMethod <: LinearMethod end
abstract type IterativeMethod <: LinearMethod end

abstract type NewtonMethod <: NonlinearMethod end
abstract type PicardMethod <: NonlinearMethod end

"""
    OptimizerMethod <: SolverMethod

The `OptimizerMethod` is used in [`Optimizer`](@ref) and determines the algorithm that is used.
"""
abstract type OptimizerMethod <: SolverMethod end

"""
    QuasiNewtonMethod <: OptimizerMethod

Includes [`BFGS`](@ref) and [`DFP`](@ref).
"""
abstract type QuasiNewtonMethod <: OptimizerMethod end

struct Newton <: OptimizerMethod end

"""
Algorithm taken from [nocedal2006numerical](@cite).
"""
struct DFP <: QuasiNewtonMethod end

"""
Algorithm taken from [nocedal2006numerical](@cite).
"""
struct BFGS <: QuasiNewtonMethod end

Base.show(io::IO, alg::Newton) = print(io, "Newton")
Base.show(io::IO, alg::DFP) = print(io, "DFP")
Base.show(io::IO, alg::BFGS) = print(io, "BFGS")