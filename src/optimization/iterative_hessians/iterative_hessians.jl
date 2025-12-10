"""
    IterativeHessian <: Hessian

An abstract type derived from [`Hessian`](@ref).
Its main purpose is defining a supertype that encompasses [`HessianBFGS`](@ref) and [`HessianDFP`](@ref) for dispatch.
"""
abstract type IterativeHessian{T} <: Hessian{T} end