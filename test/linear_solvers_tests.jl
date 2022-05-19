
using LinearAlgebra: ldiv!
using SimpleSolvers
using Test


struct LinearSolverTest{T} <: LinearSolver{T} end

test_solver = LinearSolverTest{Float64}()

@test_throws ErrorException factorize!(test_solver)
@test_throws ErrorException ldiv!(test_solver)


A = [[+4.  +5.  -2.]
     [+7.  -1.  +2.]
     [+3.  +1.  +4.]]
x = [+4., -4., +5.]
b = [-14., +42., +28.]


function test_lu_solver(solver, A, b, x)
    for T in (Float64, ComplexF64, Float32, ComplexF32)
        AT = convert(Matrix{T}, A)
        bT = convert(Vector{T}, b)
        xT = convert(Vector{T}, x)

        lu1 = solver(AT)
        x1  = ldiv!(zero(xT), lu1, bT)
        @test x1 ≈ xT atol=8*eps(real(T))

        lu2 = solver(rand(T, size(A)...))
        x2  = zero(xT)
        factorize!(lu2, AT)
        ldiv!(x2, lu2, bT)
        @test x2 ≈ xT atol=8*eps(real(T))
    end
end

test_lu_solver(LUSolverLAPACK, A, b, x)
test_lu_solver(LUSolver, A, b, x)
