
abstract type BracketingMethod <: SolverMethod end
abstract type LinearMethod <: SolverMethod end
abstract type NonlinearMethod <: SolverMethod end

abstract type DirectMethod <: LinearMethod end
abstract type IterativeMethod <: LinearMethod end

abstract type NewtonMethod <: NonlinearMethod end
abstract type PicardMethod <: NonlinearMethod end

Base.show(io::IO, alg::Newton) = print(io, "Newton")
Base.show(io::IO, alg::DFP) = print(io, "DFP")
Base.show(io::IO, alg::BFGS) = print(io, "BFGS")