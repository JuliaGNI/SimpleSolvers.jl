using SimpleSolvers
using SimpleSolvers: jacobian!
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

const sys₁ = NonlinearSystem(F, A₁[:, 1], A₁[1, :]; mode = :autodiff)
const sys₂ = NonlinearSystem(F, A₁[:, 1], A₁[1, :]; mode = :finite)
const jac₃ = Jacobian{eltype(A₁)}(DF!, size(A₁, 1); mode = :user)
const sys₃ = NonlinearSystem(F, jac₃, A₁[:, 1], A₁[1, :])

function test_various_nonlinearsystems(A::AbstractMatrix{T}, b::AbstractVector{T}) where {T}
    params = (A = A, b = b)
    x = rand(T, length(A[1, :]))

    @test value!(sys₁, x, params) ≈ value!(sys₂, x, params) ≈ value!(sys₃, x, params) ≈ F(zero(x), x, params)
    @test jacobian!(sys₁, x, params) ≈ jacobian!(sys₂, x, params) ≈ jacobian!(sys₃, x, params)
end

for (A, b) in ((A₁, b₁), (A₂, b₂))
    test_various_nonlinearsystems(A, b)
end