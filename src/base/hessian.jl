
abstract type HessianParameters{T} end

compute_hessian!(h, x, hessian::HessianParameters) = hessian(h,x)

function compute_hessian(x, hessian::HessianParameters)
    h = alloc_h(x)
    hessian(h,x)
    return h
end

function check_hessian(H::AbstractMatrix)
    println("Condition Number of Hessian: ", cond(H))
    println("Determinant of Hessian:      ", det(H))
    println("minimum(|Hessian|):          ", minimum(abs.(H)))
    println("maximum(|Hessian|):          ", maximum(abs.(H)))
    println()
end

function print_hessian(H::AbstractMatrix)
    display(H)
    println()
end


struct HessianParametersUser{T, HT <: Callable} <: HessianParameters{T}
    H!::HT
end

HessianParametersUser(H!::HT, ::AbstractVector{T}) where {T,HT} = HessianParametersUser{T,HT}(H!)

function (hes::HessianParametersUser{T})(H::AbstractMatrix{T}, x::AbstractVector{T}) where {T}
    hes.H!(H, x)
end


struct HessianParametersAD{T, FT <: Callable, HT <: ForwardDiff.HessianConfig} <: HessianParameters{T}
    F::FT
    Hconfig::HT

    function HessianParametersAD{T}(F::FT, Hconfig::HT) where {T, FT, HT}
        new{T, FT, HT}(F, Hconfig)
    end
end

function HessianParametersAD(F::FT, x::AbstractVector{T}) where {T, FT <: Callable}
    Hconfig = ForwardDiff.HessianConfig(F, x)
    HessianParametersAD{T}(F, Hconfig)
end

HessianParametersAD{T}(F, nx::Int) where {T} = HessianParametersAD{T}(F, zeros(T, nx))

function (hes::HessianParametersAD{T})(H::AbstractMatrix{T}, x::AbstractVector{T}) where {T}
    ForwardDiff.hessian!(H, hes.F, x, hes.Hconfig)
end

function compute_hessian_ad!(H::AbstractMatrix{T}, x::AbstractVector{T}, F::FT) where {T, FT <: Callable}
    hes = HessianParametersAD(F, x)
    hes(H,x)
end



function HessianParameters(ForH::Callable, x::AbstractVector{T}; mode = :autodiff, kwargs...) where {T}
    if mode == :autodiff
        Hparams = HessianParametersAD(ForH, x)
    else
        Hparams = HessianParametersUser(ForH, x)
    end
    return Hparams
end

HessianParameters(H!::Callable, F, x::AbstractVector; kwargs...) where {T} = HessianParameters(H!, nx; mode = :user, kwargs...)

HessianParameters(H!::Nothing, F, x::AbstractVector; kwargs...) where {T} = HessianParameters(F, nx;  mode = :autodiff, kwargs...)

HessianParameters{T}(ForH::Callable, nx::Int; kwargs...) where {T} = HessianParameters(ForH, zeros(T,nx); kwargs...)

HessianParameters{T}(H!::Callable, F, nx::Int; kwargs...) where {T} = HessianParameters(H!, nx; kwargs...)

HessianParameters{T}(H!::Nothing, F, nx::Int; kwargs...) where {T} = HessianParameters(F, nx; kwargs...)

function compute_hessian!(h::AbstractMatrix, x::AbstractVector, ForH::Callable; kwargs...)
    hessian = HessianParameters(ForH, x; kwargs...)
    hessian(h,x)
end
