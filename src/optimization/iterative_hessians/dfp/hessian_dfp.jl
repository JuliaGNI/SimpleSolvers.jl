"""
    HessianDFP <: Hessian

The [`Hessian`](@ref) corresponding to the [`DFP`](@ref) method.
"""
struct HessianDFP{T,FT <: Callable} <: IterativeHessian{T}
    F::FT

    function HessianDFP(F::FT, ::AbstractVector{T}) where {T, FT <: Callable}
        new{T, FT}(F)
    end
end

HessianDFP{T}(F::Callable, n::Integer) where {T} = HessianDFP(F, zeros(T, n))

HessianDFP(obj::OptimizerProblem, x::AbstractVector) = HessianDFP(obj.F, x)

Hessian(::DFP, F::Callable, x::AbstractVector) = HessianDFP(F, x)

Hessian(::DFP, Obj::OptimizerProblem, x::AbstractVector) = HessianDFP(Obj.F, x)

(hes::HessianDFP)(::AbstractMatrix, ::AbstractVector) = error("This has to be called together with a cache.")