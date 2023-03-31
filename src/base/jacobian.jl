
const DEFAULT_JACOBIAN_ϵ = 8sqrt(eps())


abstract type Jacobian{T} end

compute_jacobian!(j::AbstractMatrix, x::AbstractVector, jacobian::Jacobian) = jacobian(j,x)

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


struct JacobianFunction{T} <: Jacobian{T}
end

JacobianFunction(::AbstractArray{T}) where {T} = JacobianFunction{T}()

function (::JacobianFunction{T})(j::AbstractMatrix{T}, x::AbstractVector{T}, jac::Callable) where {T}
    jac(j, x)
end


struct JacobianAutodiff{T, JT <: ForwardDiff.JacobianConfig, YT <: AbstractVector{T}} <: Jacobian{T}
    Jconfig::JT
    ty::YT

    function JacobianAutodiff{T}(Jconfig::JT, y::YT) where {T, JT, YT}
        new{T, JT, YT}(Jconfig, y)
    end

end

function JacobianAutodiff(x::AbstractVector{T}, y::AbstractVector{T}) where {T}
    Jconfig = ForwardDiff.JacobianConfig(nothing, y, x)
    JacobianAutodiff{T}(Jconfig, zero(y))
end

function JacobianAutodiff{T}(nx::Int, ny::Int) where {T}
    tx = zeros(T, nx)
    ty = zeros(T, ny)
    JacobianAutodiff(tx, ty)
end

JacobianAutodiff{T}(F!, n) where {T} = JacobianAutodiff{T}(F!, n, n)

function (jac::JacobianAutodiff{T})(J::AbstractMatrix{T}, x::AbstractVector{T}, f::Callable) where {T}
    ForwardDiff.jacobian!(J, f, jac.ty, x, jac.Jconfig)
end


struct JacobianFiniteDifferences{T} <: Jacobian{T}
    ϵ::T
    f1::Vector{T}
    f2::Vector{T}
    e::Vector{T}
    tx::Vector{T}
end

function JacobianFiniteDifferences{T}(nx::Int, ny::Int; ϵ=DEFAULT_JACOBIAN_ϵ) where {T}
    f1 = zeros(T, ny)
    f2 = zeros(T, ny)
    e  = zeros(T, nx)
    tx = zeros(T, nx)
    JacobianFiniteDifferences{T}(ϵ, f1, f2, e, tx)
end

JacobianFiniteDifferences{T}(n; kwargs...) where {T} = JacobianFiniteDifferences{T}(n, n; kwargs...)

function (jac::JacobianFiniteDifferences{T})(J::AbstractMatrix{T}, x::AbstractVector{T}, f::Callable) where {T}
    local ϵⱼ::T

    for j in eachindex(x)
        ϵⱼ = jac.ϵ * x[j] + jac.ϵ
        fill!(jac.e, 0)
        jac.e[j] = 1
        jac.tx .= x .- ϵⱼ .* jac.e
        f(jac.f1, jac.tx)
        jac.tx .= x .+ ϵⱼ .* jac.e
        f(jac.f2, jac.tx)
        for i in eachindex(x)
            J[i,j] = (jac.f2[i] - jac.f1[i]) / (2ϵⱼ)
        end
    end
end


function Jacobian{T}(nx::Int, ny::Int; mode = :autodiff, diff_type = :forward, kwargs...) where {T}
    if mode == :autodiff
        if diff_type == :forward
            Jparams = JacobianAutodiff{T}(nx, ny)
        else
            Jparams = JacobianFiniteDifferences{T}(nx, ny; kwargs...)
        end
    else
        Jparams = JacobianFunction{T}()
    end
    return Jparams
end

Jacobian{T}(n::Int; kwargs...) where {T} = Jacobian{T}(n, n; kwargs...)

Jacobian{T}(J!::Callable, F!, nx, ny; kwargs...) where {T} = Jacobian{T}(nx, ny; mode = :user, kwargs...)

Jacobian{T}(J!::Nothing, F!, nx, ny; kwargs...) where {T} = Jacobian{T}(nx, ny; mode = :autodiff, kwargs...)

Jacobian{T}(J!, F!, n; kwargs...) where {T} = Jacobian{T}(J!, F!, n, n; kwargs...)

function compute_jacobian!(j::AbstractMatrix{T}, x::AbstractVector{T}, ForJ::Callable; kwargs...) where {T}
    jacobian = Jacobian{T}(size(j,1), size(j,2); kwargs...)
    jacobian(j,x,ForJ)
end

function compute_jacobian!(j::AbstractMatrix, x::AbstractVector, ForJ::Callable, jacobian::Jacobian)
    jacobian(j,x,ForJ)
end
