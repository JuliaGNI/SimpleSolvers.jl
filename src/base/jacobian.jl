
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


struct JacobianFunction{T, JT} <: Jacobian{T}
    J!::JT
end

JacobianFunction(J!::JT, ::AbstractArray{T}) where {T,JT} = JacobianFunction{T,JT}(J!)

function (jac::JacobianFunction{T})(J::AbstractMatrix{T}, x::AbstractVector{T}) where {T}
    jac.J!(J, x)
end


struct JacobianAutodiff{T, FT, JT <: ForwardDiff.JacobianConfig, YT <: AbstractVector{T}} <: Jacobian{T}
    F!::FT
    Jconfig::JT
    ty::YT

    function JacobianAutodiff{T}(F!::FT, Jconfig::JT, y::YT) where {T, FT, JT, YT}
        new{T, FT, JT, YT}(F!, Jconfig, y)
    end

end

function JacobianAutodiff(F!::FT, x::AbstractVector{T}, y::AbstractVector{T}) where {T, FT}
    Jconfig = ForwardDiff.JacobianConfig(F!, y, x)
    JacobianAutodiff{T}(F!, Jconfig, zero(y))
end

function JacobianAutodiff{T}(F!::FT, nx::Int, ny::Int) where {T, FT}
    tx = zeros(T, nx)
    ty = zeros(T, ny)
    JacobianAutodiff(F!, tx, ty)
end

JacobianAutodiff{T}(F!, n) where {T} = JacobianAutodiff{T}(F!, n, n)

function (jac::JacobianAutodiff{T})(J::AbstractMatrix{T}, x::AbstractVector{T}) where {T}
    ForwardDiff.jacobian!(J, jac.F!, jac.ty, x, jac.Jconfig)
end

function compute_jacobian_ad!(J::AbstractMatrix{T}, x::AbstractVector{T}, F!::FT) where {T, FT}
    jac = JacobianAutodiff{T}(F!, length(x))
    jac(J,x)
end


struct JacobianFiniteDifferences{T, FT} <: Jacobian{T}
    ϵ::T
    F!::FT
    f1::Vector{T}
    f2::Vector{T}
    e::Vector{T}
    tx::Vector{T}
end

function JacobianFiniteDifferences{T}(F!::FT, nx::Int, ny::Int; ϵ=DEFAULT_JACOBIAN_ϵ) where {T, FT}
    f1 = zeros(T, ny)
    f2 = zeros(T, ny)
    e  = zeros(T, nx)
    tx = zeros(T, nx)
    JacobianFiniteDifferences{T,FT}(ϵ, F!, f1, f2, e, tx)
end

JacobianFiniteDifferences{T}(F!, n; kwargs...) where {T} = JacobianFiniteDifferences{T}(F!, n, n; kwargs...)

function (jac::JacobianFiniteDifferences{T})(J::AbstractMatrix{T}, x::AbstractVector{T}) where {T}
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
    jac = JacobianFiniteDifferences{T}(F!, length(x); kwargs...)
    jac(J,x)
end


function Jacobian{T}(ForJ, nx::Int, ny::Int; mode = :autodiff, diff_type = :forward, kwargs...) where {T}
    if mode == :autodiff
        if diff_type == :forward
            Jparams = JacobianAutodiff{T}(ForJ, nx, ny)
        else
            Jparams = JacobianFiniteDifferences{T}(ForJ, nx, ny; kwargs...)
        end
    else
        Jparams = JacobianFunction{T, typeof(ForJ)}(ForJ)
    end
    return Jparams
end

Jacobian{T}(ForJ, n::Int; kwargs...) where {T} = Jacobian{T}(ForJ, n, n; kwargs...)

Jacobian{T}(J!, F!, nx, ny; kwargs...) where {T} = Jacobian{T}(J!, nx, ny; mode = :user, kwargs...)

Jacobian{T}(J!::Nothing, F!, nx, ny; kwargs...) where {T} = Jacobian{T}(F!, nx, ny;  mode = :autodiff, kwargs...)

Jacobian{T}(J!, F!, n; kwargs...) where {T} = Jacobian{T}(J!, F!, n, n; kwargs...)

function compute_jacobian!(j::AbstractMatrix{T}, x::AbstractVector{T}, ForJ::Callable; kwargs...) where {T}
    jacobian = Jacobian{T}(ForJ, size(j,1), size(j,2); kwargs...)
    jacobian(j,x)
end
