using LinearAlgebra: ldiv!
using SimpleSolvers
using SimpleSolvers: update!
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

function solve_with_constructor(solver, xT::AbstractVector{T}, AT::AbstractMatrix{T}, bT::AbstractVector{T}) where {T}
    x1 = similar(xT)
    ls1 = LinearSystem(x1, copy(AT), bT)
    lu1 = solver(ls1)
    x1
end

function solve_with_factorize_and_ldiv(solver, xT::AbstractVector{T}, AT::AbstractMatrix{T}, bT::AbstractVector{T}) where {T}
    lu2 = solver(rand(T, size(AT)...))
    x2  = similar(xT)
    factorize!(lu2, AT)
    ldiv!(x2, lu2, bT)
    x2
end

function solve_with_update_and_solve(solver, xT::AbstractVector{T}, AT::AbstractMatrix{T}, bT::AbstractVector{T}) where {T}
    lu3 = solver(similar(AT))
    x3 = similar(xT)
    update!(linearsystem(lu3), x3, AT, bT)
    solve!(lu3)
    solution(lu3) # here we can't return x3 since this has been copied into linearsystem(lu3) with update!
end

function test_lu_solver(solver, A, b, x)
    for T in (Float64, ComplexF64, Float32, ComplexF32)
        AT = convert(Matrix{T}, A)
        bT = convert(Vector{T}, b)
        xT = convert(Vector{T}, x)

        x1 = solve_with_constructor(solver, xT, AT, bT)
        @test x1 ≈ xT atol=8*eps(real(T))

        x2 = solve_with_factorize_and_ldiv(solver, xT, AT, bT)
        @test x2 ≈ xT atol=8*eps(real(T))

        x3 = solve_with_update_and_solve(solver, xT, AT, bT)
        @test x3 ≈ xT atol=8*eps(real(T))
    end
end

test_lu_solver(LUSolverLAPACK, A, b, x)
test_lu_solver(LUSolver, A, b, x)
