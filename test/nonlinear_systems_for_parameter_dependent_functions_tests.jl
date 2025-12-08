using SimpleSolvers
using SimpleSolvers: value!
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

const A₁ = [3. 6. 7.; 9. 18. 19.; 11. 22. 23.]
const b₁ = [1., 1., 2.]

const A₂ = [1. 2. 3.; 4. 5. 6.; 7. 8. 9.]
const b₂ = [1., 1., 1.]

const sys₁ = NonlinearProblem(F, A₁[:, 1], A₁[1, :])
const sys₂ = NonlinearProblem(F, DF!, A₁[:, 1], A₁[1, :]) # the analytic Jacobian is stored in the problem

const jac₁ = JacobianFunction(F, DF!, A₁[:, 1])
const jac₂ = JacobianAutodiff(F, A₁[:, 1])
const jac₃ = JacobianFiniteDifferences{Float64}(F, size(A₁, 1), size(A₁, 2))

function test_various_nonlinearproblems(A::AbstractMatrix{T}, b::AbstractVector{T}) where {T}
    params = (A = A, b = b)
    x = rand(T, length(A[1, :]))

    @test value!(sys₁, x, params) ≈ value!(sys₂, x, params) ≈ F(zero(x), x, params)
    @test jac₁(sys₁, x, params) ≈ jac₁(sys₂, x, params) ≈ jac₂(sys₁, x, params) ≈ jac₂(sys₂, x, params) ≈ jac₃(sys₁, x, params) ≈ jac₃(sys₂, x, params)
end

for (A, b) in ((A₁, b₁), (A₂, b₂))
    test_various_nonlinearproblems(A, b)
end