using SimpleSolvers
using Test


@testset "Basic Jacobian functionality and consistency" begin

    n = 1
    T = Float64
    x = [T(π),]
    j = reshape(2x, 1, 1)


    function F!(f::AbstractVector, x::AbstractVector, params)
        f .= x .^ 2
    end

    function J!(g::AbstractMatrix, x::AbstractVector, params)
        g .= 0
        for i in eachindex(x)
            g[i, i] = 2x[i]
        end
        g
    end


    JPAD = JacobianAutodiff{T}(F!, n, n)
    JPFD = JacobianFiniteDifferences{T}(F!, n, n)
    JPUS = JacobianFunction{T}(F!, J!)

    @test typeof(JPAD) <: JacobianAutodiff
    @test typeof(JPFD) <: JacobianFiniteDifferences
    @test typeof(JPUS) <: JacobianFunction

    @test JPAD == JacobianAutodiff{T}(F!, n)
    @test JPAD == JacobianAutodiff(F!, x)
    @test JPFD == JacobianFiniteDifferences{T}(F!, n)
    @test JPFD == JacobianFiniteDifferences(F!, x)


    jad = zero(j)
    jfd = zero(j)
    jus = zero(j)

    JPAD(jad, x, nothing)
    JPFD(jfd, x, nothing)
    JPUS(jus, x, nothing)

    @test jad ≈ j atol = eps()
    @test jfd ≈ j atol = 1E-7
    @test jus == j


    jad1 = zero(j)
    jfd1 = zero(j)
    jus1 = zero(j)

    JPAD(jad1, x, nothing)
    JPFD(jfd1, x, nothing)
    JPUS(jus1, x, nothing)

    @test jad1 == jad
    @test jfd1 == jfd
    @test jus1 == jus


    jad2 = zero(j)
    jfd2 = zero(j)
    jus2 = zero(j)

    JPAD(jad2, x, nothing)
    JPFD(jfd2, x, nothing)
    JPUS(jus2, x, nothing)

    @test jad2 == jad
    @test jfd2 == jfd
    @test jus2 == jus

end


@testset "Jacobians with parameter-dependent functions" begin

    function F(f::AbstractVector{T}, x::AbstractVector{T}, params) where {T}
        f .= (params.A * x + params.b) .^ 2
    end

    function DF!(jacobian_matrix::AbstractMatrix{T}, x::AbstractVector{T}, params) where {T}
        for i in axes(jacobian_matrix, 1)
            for j in axes(jacobian_matrix, 2)
                jacobian_matrix[j, i] = 2 * params.A[j, i] * (params.A[j, :]' * x + params.b[j])
            end
        end
        jacobian_matrix
    end

    A₁ = [3.0 6.0 7.0; 9.0 18.0 19.0; 11.0 22.0 23.0]
    b₁ = [1.0, 1.0, 2.0]

    A₂ = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
    b₂ = [1.0, 1.0, 1.0]

    jac₁ = JacobianAutodiff{eltype(A₁)}(F, size(A₁)[2], size(A₁)[1])
    jac₂ = JacobianFiniteDifferences{eltype(A₁)}(F, size(A₁)[2], size(A₁)[1])
    jac₃ = JacobianFunction{eltype(A₁)}(F, DF!)

    function test_various_jacobians(A::AbstractMatrix{T}, b::AbstractVector{T}) where {T}
        params = (A=A, b=b)
        x = rand(T, length(A[1, :]))
        j₁ = zero(A)
        j₂ = zero(A)
        j₃ = zero(A)

        @test jac₁(j₁, x, params) ≈ jac₂(j₂, x, params) ≈ jac₃(j₃, x, params)
    end

    for (A, b) in ((A₁, b₁), (A₂, b₂))
        test_various_jacobians(A, b)
    end

end
