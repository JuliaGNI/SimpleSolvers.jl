"""
    LinesearchMethod

Examples include [`StaticState`](@ref), [`BacktrackingState`](@ref), [`BisectionState`](@ref) and [`QuadraticState`](@ref).
See these examples for specific information on linesearch algorithms.
"""
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

Base.show(io::IO, alg::Static) = print(io, "Static")
Base.show(io::IO, alg::Backtracking) = print(io, "Backtracking")
Base.show(io::IO, alg::Bisection) = print(io, "Bisection")
Base.show(io::IO, alg::Quadratic) = print(io, "Quadratic Polynomial")