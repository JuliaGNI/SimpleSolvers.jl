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
    params = (A=A, b=b)
    x = rand(T, length(A[1, :]))

    @test value!(zero(x), sys₁, x, params) ≈ value!(zero(x), sys₂, x, params) ≈ F(zero(x), x, params)
    @test jacobian!(zero(A₁), sys₂, x, params) ≈ jacobian!(zero(A₁), sys₂, x, params) ≈ jacobian!(zero(A₁), sys₂, x, params)
    @test_throws "NonlinearProblem does not contain Jacobian." jacobian!(zero(A₁), sys₁, x, params)
end

for (A, b) in ((A₁, b₁), (A₂, b₂))
    test_various_nonlinearproblems(A, b)
end

# Phase 3.1 regression: the phantom eltype parameter `T` was removed from
# `NonlinearProblem`, so it now carries exactly two type parameters (the function
# and Jacobian types).  No field depended on `T`.
@testset "NonlinearProblem has no phantom eltype parameter (§4)" begin
    # The two-parameter form (function type, Jacobian type) is now the full type;
    # the old three-parameter `NonlinearProblem{Float64,...}` no longer exists.
    @test sys₁ isa NonlinearProblem{typeof(F),Missing}
    @test sys₂ isa NonlinearProblem{typeof(F),typeof(DF!)}
end

# Phase 3.3 regression: the inner constructor used to force `x` and `f` to the
# *same* concrete array type (`x::Tx, f::Tx`), so mixed container types (e.g. a
# `Vector` and a `SubArray` with the same eltype) failed deep in construction.
@testset "NonlinearProblem accepts independent x/f container types (§4)" begin
    M = [1.0 2.0; 3.0 4.0]
    xv = M[:, 1]            # Vector{Float64}
    fv = @view M[1, :]      # SubArray{Float64}
    @test typeof(xv) != typeof(fv)
    @test NonlinearProblem(F, xv, fv) isa NonlinearProblem
    @test NonlinearProblem(F, DF!, xv, fv) isa NonlinearProblem
end
