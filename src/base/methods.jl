
abstract type Method end

abstract type BracketingMethod <: Method end
abstract type LinearMethod <: Method end
abstract type NonlinearMethod <: Method end

abstract type DirectMethod <: LinearMethod end
abstract type IterativeMethod <: LinearMethod end

abstract type NewtonMethod <: NonlinearMethod end
abstract type LinesearchMethod <: NonlinearMethod end

struct Newton <: NewtonMethod end
struct DFP <: NewtonMethod end
struct BFGS <: NewtonMethod end

struct Bisection <: LinesearchMethod end
struct Armijo <: LinesearchMethod end
struct ArmijoQuadratic <: LinesearchMethod end

struct Static{T} <: LinesearchMethod
    alpha::T
    Static(alpha::T = 1.0) where {T} = new{T}(alpha)
end


Base.show(io::IO, alg::Newton) = print(io, "Newton")
Base.show(io::IO, alg::DFP) = print(io, "DFP")
Base.show(io::IO, alg::BFGS) = print(io, "BFGS")

Base.show(io::IO, alg::Static) = print(io, "Static")
Base.show(io::IO, alg::Bisection) = print(io, "Bisection")
Base.show(io::IO, alg::Armijo) = print(io, "Armijo")
Base.show(io::IO, alg::ArmijoQuadratic) = print(io, "Armijo (quadratic)")
