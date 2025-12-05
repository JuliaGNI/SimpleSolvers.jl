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