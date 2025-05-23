using LinearAlgebra: ldiv!
using SimpleSolvers
using SimpleSolvers: update!, LinearSolverMethod, LinearSolverCache
using Test

struct TestMethod <: LinearSolverMethod end
struct TestCache{T} <: LinearSolverCache{T}
    TestCache(::AbstractVector{T}) where {T} = new{T}()
end

y = [.1, .1]
x = similar(y)
test_solver = LinearSolver(TestMethod(), TestCache(x))

@test_throws ErrorException factorize!(test_solver)

@test_throws ErrorException ldiv!(x, test_solver, y)

A = [[+4.  +5.  -2.]
     [+7.  -1.  +2.]
     [+3.  +1.  +4.]]
x = [+4., -4., +5.]
b = [-14., +42., +28.]

function solve_with_factorize_and_ldiv(solver_method::LinearSolverMethod, xT::AbstractVector{T}, AT::AbstractMatrix{T}, bT::AbstractVector{T}) where {T}
    ls1 = LinearSolver(solver_method, rand(T, size(AT)...))
    x1  = similar(xT)
    factorize!(ls1, AT)
    ldiv!(x1, ls1, bT)
    x1
end

function solve_with_solve(solver_method, ::AbstractVector{T}, AT::AbstractMatrix{T}, bT::AbstractVector{T}) where {T}
    solve(solver_method, AT, bT)
end

function solve_with_solve!(solver_method, xT::AbstractVector{T}, AT::AbstractMatrix{T}, bT::AbstractVector{T}) where {T}
    ls3 = LinearSolver(solver_method, rand(T, length(xT)))
    x3 = similar(xT)
    solve!(x3, ls3, AT, bT)
    x3
end

function test_lu_solver(solver, A, b, x)
    for T in (Float64, ComplexF64, Float32, ComplexF32)
        AT = convert(Matrix{T}, A)
        bT = convert(Vector{T}, b)
        xT = convert(Vector{T}, x)

        x1 = solve_with_factorize_and_ldiv(solver, xT, AT, bT)
        @test x1 ≈ xT atol=8*eps(real(T))

        x2 = solve_with_solve(solver, xT, AT, bT)
        @test x2 ≈ xT atol=8*eps(real(T))

        x3 = solve_with_solve!(solver, xT, AT, bT)
        @test x3 ≈ xT atol=8*eps(real(T))
    end
end

test_lu_solver(LU(; static=false), A, b, x)
test_lu_solver(LU(; static=true), A, b, x)
