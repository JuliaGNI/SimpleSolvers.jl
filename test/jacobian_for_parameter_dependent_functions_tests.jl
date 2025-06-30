using SimpleSolvers
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

const jac₁ = Jacobian{eltype(A₁)}(F, size(A₁)[2], size(A₁)[1]; mode = :autodiff)
const jac₂ = Jacobian{eltype(A₁)}(F, size(A₁)[2], size(A₁)[1]; mode = :finite)
const jac₃ = Jacobian{eltype(A₁)}(DF!, size(A₁)[2], size(A₁)[1]; mode = :user)

function test_various_jacobians(A::AbstractMatrix{T}, b::AbstractVector{T}) where {T}
    params = (A = A, b = b)
    x = rand(T, length(A[1, :]))
    j₁ = zero(A)
    j₂ = zero(A)
    j₃ = zero(A)

    @test compute_jacobian!(j₁, x, jac₁, params) ≈ compute_jacobian!(j₂, x, jac₂, params) ≈ compute_jacobian!(j₃, x, jac₃, params)
end

for (A, b) in ((A₁, b₁), (A₂, b₂))
    test_various_jacobians(A, b)
end