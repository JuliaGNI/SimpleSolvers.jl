
const DEFAULT_JACOBIAN_ϵ = 8sqrt(eps())


abstract type JacobianParameters{T} end

compute_jacobian!(j::AbstractMatrix, x::AbstractVector, jacobian::JacobianParameters) = jacobian(j,x)

function check_jacobian(J::AbstractMatrix)
    println("Condition Number of Jacobian: ", cond(J))
    println("Determinant of Jacobian:      ", det(J))
    println("minimum(|Jacobian|):          ", minimum(abs.(J)))
    println("maximum(|Jacobian|):          ", maximum(abs.(J)))
    println()
end

function print_jacobian(J::AbstractMatrix)
    display(J)
    println()
end


struct JacobianParametersUser{T, JT} <: JacobianParameters{T}
    J!::JT
end

JacobianParametersUser(J!::JT, ::AbstractArray{T}) where {T,JT} = JacobianParametersUser{T,JT}(J!)

function (jac::JacobianParametersUser{T})(J::AbstractMatrix{T}, x::AbstractVector{T}) where {T}
    jac.J!(J, x)
end


struct JacobianParametersAD{T, FT, JT <: ForwardDiff.JacobianConfig, YT <: AbstractVector{T}} <: JacobianParameters{T}
    F!::FT
    Jconfig::JT
    ty::YT

    function JacobianParametersAD{T}(F!::FT, Jconfig::JT, y::YT) where {T, FT, JT, YT}
        new{T, FT, JT, YT}(F!, Jconfig, y)
    end

end

function JacobianParametersAD(F!::FT, x::AbstractVector{T}, y::AbstractVector{T}) where {T, FT}
    Jconfig = ForwardDiff.JacobianConfig(F!, y, x)
    JacobianParametersAD{T}(F!, Jconfig, zero(y))
end

function JacobianParametersAD{T}(F!::FT, nx::Int, ny::Int) where {T, FT}
    tx = zeros(T, nx)
    ty = zeros(T, ny)
    JacobianParametersAD(F!, tx, ty)
end

JacobianParametersAD{T}(F!, n) where {T} = JacobianParametersAD{T}(F!, n, n)

function (jac::JacobianParametersAD{T})(J::AbstractMatrix{T}, x::AbstractVector{T}) where {T}
    ForwardDiff.jacobian!(J, jac.F!, jac.ty, x, jac.Jconfig)
end

function compute_jacobian_ad!(J::AbstractMatrix{T}, x::AbstractVector{T}, F!::FT) where {T, FT}
    jac = JacobianParametersAD{T}(F!, length(x))
    jac(J,x)
end


struct JacobianParametersFD{T, FT} <: JacobianParameters{T}
    ϵ::T
    F!::FT
    f1::Vector{T}
    f2::Vector{T}
    e::Vector{T}
    tx::Vector{T}
end

function JacobianParametersFD{T}(F!::FT, nx::Int, ny::Int; ϵ=DEFAULT_JACOBIAN_ϵ) where {T, FT}
    f1 = zeros(T, ny)
    f2 = zeros(T, ny)
    e  = zeros(T, nx)
    tx = zeros(T, nx)
    JacobianParametersFD{T,FT}(ϵ, F!, f1, f2, e, tx)
end

JacobianParametersFD{T}(F!, n; kwargs...) where {T} = JacobianParametersFD{T}(F!, n, n; kwargs...)

function (jac::JacobianParametersFD{T})(J::AbstractMatrix{T}, x::AbstractVector{T}) where {T}
    local ϵⱼ::T

    for j in eachindex(x)
        ϵⱼ = jac.ϵ * x[j] + jac.ϵ
        fill!(jac.e, 0)
        jac.e[j] = 1
        jac.tx .= x .- ϵⱼ .* jac.e
        jac.F!(jac.f1, jac.tx)
        jac.tx .= x .+ ϵⱼ .* jac.e
        jac.F!(jac.f2, jac.tx)
        for i in eachindex(x)
            J[i,j] = (jac.f2[i] - jac.f1[i]) / (2ϵⱼ)
        end
    end
end

function compute_jacobian_fd!(J::AbstractMatrix{T}, x::AbstractVector{T}, F!::FT; kwargs...) where {T, FT}
    jac = JacobianParametersFD{T}(F!, length(x); kwargs...)
    jac(J,x)
end


function JacobianParameters{T}(ForJ, nx::Int, ny::Int; mode = :autodiff, diff_type = :forward, kwargs...) where {T}
    if mode == :autodiff
        if diff_type == :forward
            Jparams = JacobianParametersAD{T}(ForJ, nx, ny)
        else
            Jparams = JacobianParametersFD{T}(ForJ, nx, ny; kwargs...)
        end
    else
        Jparams = JacobianParametersUser{T, typeof(ForJ)}(ForJ)
    end
    return Jparams
end

JacobianParameters{T}(ForJ, n::Int; kwargs...) where {T} = JacobianParameters{T}(ForJ, n, n; kwargs...)

JacobianParameters{T}(J!, F!, nx, ny; kwargs...) where {T} = JacobianParameters{T}(J!, nx, ny; mode = :user, kwargs...)

JacobianParameters{T}(J!::Nothing, F!, nx, ny; kwargs...) where {T} = JacobianParameters{T}(F!, nx, ny;  mode = :autodiff, kwargs...)

JacobianParameters{T}(J!, F!, n; kwargs...) where {T} = JacobianParameters{T}(J!, F!, n, n; kwargs...)

function compute_jacobian!(j::AbstractMatrix{T}, x::AbstractVector{T}, ForJ; kwargs...) where {T}
    jacobian = JacobianParameters{T}(ForJ, size(j,1), size(j,2); kwargs...)
    jacobian(j,x)
end
