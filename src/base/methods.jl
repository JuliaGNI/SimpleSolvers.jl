
abstract type SolverMethod end

abstract type BracketingMethod <: SolverMethod end
abstract type LinearMethod <: SolverMethod end
abstract type NonlinearMethod <: SolverMethod end

abstract type DirectMethod <: LinearMethod end
abstract type IterativeMethod <: LinearMethod end

abstract type NewtonMethod <: NonlinearMethod end
abstract type PicardMethod <: NonlinearMethod end
abstract type LinesearchMethod <: NonlinearMethod end

struct Newton <: NewtonMethod end
struct DFP <: NewtonMethod end
struct BFGS <: NewtonMethod end

struct Backtracking <: LinesearchMethod end
struct Bisection <: LinesearchMethod end
struct Quadratic <: LinesearchMethod end

struct Static{T} <: LinesearchMethod
    alpha::T
    Static(alpha::T = 1.0) where {T} = new{T}(alpha)
end


Base.show(io::IO, alg::Newton) = print(io, "Newton")
Base.show(io::IO, alg::DFP) = print(io, "DFP")
Base.show(io::IO, alg::BFGS) = print(io, "BFGS")

Base.show(io::IO, alg::Static) = print(io, "Static")
Base.show(io::IO, alg::Backtracking) = print(io, "Backtracking")
Base.show(io::IO, alg::Bisection) = print(io, "Bisection")
Base.show(io::IO, alg::Quadratic) = print(io, "Quadratic Polynomial")
