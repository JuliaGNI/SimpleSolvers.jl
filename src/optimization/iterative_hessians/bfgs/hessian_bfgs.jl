"""
    HessianBFGS <: Hessian

A `struct` derived from [`Hessian`](@ref) to be used for an [`Optimizer`](@ref).

# Fields
- `problem::`[`OptimizerProblem`](@ref): 
- `x̄`: previous solution,
- `x`: current solution,
- `δ`: *descent direction*,
- `ḡ`: previous gradient,
- `g`: current gradient,
- `γ`: difference between current and previous gradient,
- `Q`: 
- `T1`:
- `T2`:
- `T3`:
- `δγ`: the outer product of `δ` and `γ`.
- `δδ`: 

Also compare those fields with the ones of [`NewtonOptimizerCache`](@ref).
"""
struct HessianBFGS{T, FT <: Callable} <: IterativeHessian{T}
    F::FT

    function HessianBFGS(F::FT, ::AbstractVector{T}) where {T, FT <: Callable}
        new{T, FT}(F)
    end
end

HessianBFGS{T}(F::Callable, n::Integer) where {T} = HessianBFGS(F, zeros(T, n))

HessianBFGS(obj::OptimizerProblem, x::AbstractVector) = HessianBFGS(obj.F, x)

Hessian(::BFGS, ForOBJ::Callable, x::AbstractVector) = HessianBFGS(ForOBJ, x)

Hessian(::BFGS, ForOBJ::OptimizerProblem, x::AbstractVector) = HessianBFGS(ForOBJ.F, x)

(hes::Hessian)(::AbstractMatrix, ::AbstractVector) = error("This has to be called together with a cache.")