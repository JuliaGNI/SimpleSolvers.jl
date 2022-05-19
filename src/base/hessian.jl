
abstract type Hessian{T} end

initialize!(::Hessian) = nothing

compute_hessian!(h::AbstractMatrix, x::AbstractVector, hessian::Hessian) = hessian(h,x)

function compute_hessian(x, hessian::Hessian)
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


struct HessianUser{T, HT} <: Hessian{T}
    H!::HT
end

HessianUser(H!::HT, ::AbstractVector{T}) where {T,HT} = HessianUser{T,HT}(H!)

function (hes::HessianUser{T})(H::AbstractMatrix{T}, x::AbstractVector{T}) where {T}
    hes.H!(H, x)
end


struct HessianAD{T, FT, HT <: AbstractMatrix, CT <: ForwardDiff.HessianConfig} <: Hessian{T}
    F::FT
    H::HT
    Hconfig::CT

    function HessianAD{T}(F::FT, H::HT, Hconfig::CT) where {T, FT, HT, CT}
        new{T, FT, HT, CT}(F, H, Hconfig)
    end
end

function HessianAD(F::Callable, x::AbstractVector{T}) where {T}
    Hconfig = ForwardDiff.HessianConfig(F, x)
    HessianAD{T}(F, alloc_h(x), Hconfig)
end

HessianAD(F::MultivariateObjective, x) = HessianAD(F.F, x)

HessianAD{T}(F, nx::Int) where {T} = HessianAD{T}(F, zeros(T, nx))

function (hes::HessianAD{T})(H::AbstractMatrix{T}, x::AbstractVector{T}) where {T}
    ForwardDiff.hessian!(H, hes.F, x, hes.Hconfig)
end

function (hes::HessianAD{T})(x::AbstractVector{T}) where {T}
    ForwardDiff.hessian!(hes.H, hes.F, x, hes.Hconfig)
end

function compute_hessian_ad!(H::AbstractMatrix{T}, x::AbstractVector{T}, F::FT) where {T, FT}
    hes = HessianAD(F, x)
    hes(H,x)
end

initialize!(H::HessianAD, x) = H(x)
update!(H::HessianAD, x::AbstractVector) = H(x)

Base.inv(H::HessianAD) = inv(H.H)

Base.:\(H::HessianAD, b) = H.H \ b

LinearAlgebra.ldiv!(x, H::HessianAD, b) = x .= H \ b
# LinearAlgebra.ldiv!(x, H::HessianAD, b) = LinearAlgebra.ldiv!(x, H.H, b)
# TODO: Make this work!


function Hessian(ForH, x::AbstractVector{T}; mode = :autodiff, kwargs...) where {T}
    if mode == :autodiff
        Hparams = HessianAD(ForH, x)
    else
        Hparams = HessianUser(ForH, x)
    end
    return Hparams
end

Hessian(H!, F, x::AbstractVector; kwargs...) where {T} = Hessian(H!, nx; mode = :user, kwargs...)

Hessian(H!::Nothing, F, x::AbstractVector; kwargs...) where {T} = Hessian(F, nx;  mode = :autodiff, kwargs...)

Hessian{T}(ForH, nx::Int; kwargs...) where {T} = Hessian(ForH, zeros(T,nx); kwargs...)

Hessian{T}(H!, F, nx::Int; kwargs...) where {T} = Hessian(H!, nx; kwargs...)

Hessian{T}(H!::Nothing, F, nx::Int; kwargs...) where {T} = Hessian(F, nx; kwargs...)

function compute_hessian!(h::AbstractMatrix, x::AbstractVector, ForH; kwargs...)
    hessian = Hessian(ForH, x; kwargs...)
    hessian(h,x)
end
