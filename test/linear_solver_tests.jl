using LinearAlgebra: ldiv!, SingularException
using SimpleSolvers
using SimpleSolvers: LinearSolverMethod, LinearSolverCache, matrix, factorize!, cache
using Test

# §1.4 regression: `LinearProblem` must accept a non-square `A` (the RHS length
# matches the number of rows `size(A, 1)`, not the columns). Previously the inner
# constructor asserted `length(y) == size(A, 2)`, so `LinearProblem(A)` threw for
# every non-square `A`, contradicting `LinearProblem{T}(n, m)`.
@testset "LinearProblem non-square dimensions" begin
    Ans = Float64[1.0 2.0 3.0; 4.0 5.0 6.0]   # 2×3
    lp = LinearProblem(Ans)
    @test lp isa LinearProblem
    @test size(matrix(lp)) == (2, 3)

    lp2 = LinearProblem{Float64}(2, 3)
    @test lp2 isa LinearProblem
    @test size(matrix(lp2)) == (2, 3)
end

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


# §2.5 regression: a singular matrix used to leave `cache.info` set but unchecked,
# so `ldiv!` silently produced NaN/Inf.  It now throws a `SingularException`.
@testset "LU singular matrix throws (§2.5)" begin
    Asing = [1.0 2.0; 2.0 4.0]   # rank 1 → singular
    bsing = [1.0, 2.0]
    ls = LinearSolver(LU(), Asing)
    factorize!(ls, Asing)
    @test cache(ls).info != 0
    @test_throws SingularException ldiv!(similar(bsing), ls, bsing)

    # After factorizing a nonsingular matrix, `info` is reset and solving works
    # (no stale nonzero info persists).
    Aok = [4.0 3.0; 6.0 3.0]
    factorize!(ls, Aok)
    @test cache(ls).info == 0
    xok = similar(bsing)
    ldiv!(xok, ls, [1.0, 0.0])
    @test Aok * xok ≈ [1.0, 0.0] atol = 1e-12
end

# §2.5 regression: `ldiv!(x, lsolver, b)` used to corrupt the result when `x === b`
# because the permutation gather read entries it had already overwritten.
@testset "LU ldiv! with aliased x === b (§2.5)" begin
    Aa = [4.0 5.0 -2.0; 7.0 -1.0 2.0; 3.0 1.0 4.0]
    ba = [-14.0, 42.0, 28.0]
    xref = [4.0, -4.0, 5.0]
    ls = LinearSolver(LU(), Aa)
    factorize!(ls, Aa)
    v = copy(ba)
    ldiv!(v, ls, v)      # aliased in-place solve
    @test v ≈ xref atol = 1e-10
end

# §1.7 / 2.5 regression: an integer input matrix used to throw in `factorize!`
# (no eltype promotion in the cache).  The cache now promotes the element type
# to a fractional type (like `LinearAlgebra.lutype`), so factorization and
# solving work.
@testset "LU integer matrix promotes eltype (§1.7)" begin
    ls = LinearSolver(LU(), [1 2; 3 4])   # construction promotes Int → Float64
    @test eltype(cache(ls).A) == Float64
    factorize!(ls)                        # factorize the (promoted) stored matrix
    x = zeros(2)
    ldiv!(x, ls, [1.0, 1.0])
    @test [1.0 2.0; 3.0 4.0] * x ≈ [1.0, 1.0] atol = 1e-12
end
