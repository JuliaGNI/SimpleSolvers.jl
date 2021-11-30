
abstract type JacobianParameters{T} end


struct JacobianParametersUser{T, JT <: Callable} <: JacobianParameters{T}
    J::JT
end

struct JacobianParametersAD{T, FT <: Callable, JT <: ForwardDiff.JacobianConfig, VT <: AbstractVector{T}} <: JacobianParameters{T}
    F!::FT
    Jconfig::JT
    tx::VT
    ty::VT

    function JacobianParametersAD(F!::FT, Jconfig::JT, tx::VT, ty::VT) where {T, FT, JT, VT <: AbstractVector{T}}
        new{T, FT, JT, VT}(F!, Jconfig, tx, ty)
    end
end

function JacobianParametersAD(F!::FT, T, nx::Int, ny::Int) where {FT <: Callable}
    tx = zeros(T, nx)
    ty = zeros(T, ny)
    Jconfig = ForwardDiff.JacobianConfig(F!, ty, tx)
    JacobianParametersAD(F!, Jconfig, tx, ty)
end

JacobianParametersAD(F!, T, n) = JacobianParametersAD(F!, T, n, n)

struct JacobianParametersFD{T, FT} <: JacobianParameters{T}
    ϵ::T
    F!::FT
    f1::Vector{T}
    f2::Vector{T}
    e::Vector{T}
    tx::Vector{T}
end

function JacobianParametersFD(F!::FT, ϵ, T, nx::Int, ny::Int) where {FT <: Callable}
    f1 = zeros(T, ny)
    f2 = zeros(T, ny)
    e  = zeros(T, nx)
    tx = zeros(T, nx)
    JacobianParametersFD{T, typeof(F!)}(ϵ, F!, f1, f2, e, tx)
end

JacobianParametersFD(F!, ϵ, T, n) = JacobianParametersFD(F!, ϵ, T, n, n)

function computeJacobian(x::AbstractVector{T}, J::AbstractMatrix{T}, params::JacobianParametersUser{T}) where {T}
    params.J(J, x)
end

function computeJacobian(x::AbstractVector{T}, J::AbstractMatrix{T}, params::JacobianParametersFD{T}) where {T}
    local ϵⱼ::T

    for j in eachindex(x)
        ϵⱼ = params.ϵ * x[j] + params.ϵ
        fill!(params.e, 0)
        params.e[j] = 1
        params.tx .= x .- ϵⱼ .* params.e
        params.F!(params.f1, params.tx)
        params.tx .= x .+ ϵⱼ .* params.e
        params.F!(params.f2, params.tx)
        for i in eachindex(x)
            J[i,j] = (params.f2[i]-params.f1[i])/(2ϵⱼ)
        end
    end
end

function computeJacobian(x::AbstractVector{T}, J::AbstractMatrix{T}, Jparams::JacobianParametersAD{T}) where {T}
    ForwardDiff.jacobian!(J, Jparams.F!, Jparams.ty, x, Jparams.Jconfig)
end

function computeJacobianFD(x::AbstractVector{T}, J::AbstractMatrix{T}, F!::FT, ϵ::T) where{T, FT <: Callable}
    params = JacobianParametersFD(F!, ϵ, T, length(x))
    computeJacobian(x, J, params)
end

function computeJacobianAD(x::AbstractVector{T}, J::AbstractMatrix{T}, F!::FT) where{T, FT <: Callable}
    params = JacobianParametersAD(F!, T, length(x))
    computeJacobian(x, J, params)
end


function getJacobianParameters(J, F!, T, nx, ny)
    if J === nothing
        if get_config(:jacobian_autodiff)
            Jparams = JacobianParametersAD(F!, T, nx, ny)
        else
            Jparams = JacobianParametersFD(F!, get_config(:jacobian_fd_ϵ), T, nx, ny)
        end
    else
        Jparams = JacobianParametersUser{T, typeof(J)}(J)
    end
    return Jparams
end

getJacobianParameters(J, F!, T, n) = getJacobianParameters(J, F!, T, n, n)


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
