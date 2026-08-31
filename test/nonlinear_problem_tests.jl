using SimpleSolvers
using SimpleSolvers: value!, jacobian!
using Test

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

const A₁ = [3.0 6.0 7.0; 9.0 18.0 19.0; 11.0 22.0 23.0]
const b₁ = [1.0, 1.0, 2.0]

const A₂ = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
const b₂ = [1.0, 1.0, 1.0]

const sys₁ = NonlinearProblem(F, A₁[:, 1], A₁[1, :])
const sys₂ = NonlinearProblem(F, DF!, A₁[:, 1], A₁[1, :]) # the analytic Jacobian is stored in the problem

function test_various_nonlinearproblems(A::AbstractMatrix{T}, b::AbstractVector{T}) where {T}
    params = (A = A, b = b)
    x = rand(T, length(A[1, :]))

    @test value!(zero(x), sys₁, x, params) ≈ value!(zero(x), sys₂, x, params) ≈
          F(zero(x), x, params)
    # sys₂ carries the analytic Jacobian DF!; its evaluated Jacobian must match a
    # direct call to DF!.
    @test jacobian!(zero(A₁), sys₂, x, params) ≈ DF!(zero(A₁), x, params)
    @test_throws "NonlinearProblem does not contain Jacobian." jacobian!(zero(A₁), sys₁, x, params)
end

for (A, b) in ((A₁, b₁), (A₂, b₂))
    test_various_nonlinearproblems(A, b)
end

# `NonlinearProblem` carries exactly two type parameters (the function
# and Jacobian types).
@testset "NonlinearProblem has no phantom eltype parameter" begin
    @test sys₁ isa NonlinearProblem{typeof(F), Missing}
    @test sys₂ isa NonlinearProblem{typeof(F), typeof(DF!)}
end

# Verify that the inner constructor allows for mixed container types (e.g. a
# `Vector` and a `SubArray` with the same eltype).
@testset "NonlinearProblem accepts independent x/f container types" begin
    M = [1.0 2.0; 3.0 4.0]
    xv = M[:, 1]            # Vector{Float64}
    fv = @view M[1, :]      # SubArray{Float64}
    @test typeof(xv) != typeof(fv)
    @test NonlinearProblem(F, xv, fv) isa NonlinearProblem
    @test NonlinearProblem(F, DF!, xv, fv) isa NonlinearProblem
end
