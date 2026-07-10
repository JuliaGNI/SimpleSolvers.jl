using LinearAlgebra: ldiv!, SingularException
using SimpleSolvers
using SimpleSolvers: LinearSolverMethod, LinearSolverCache, matrix, factorize!, cache, pivot_index, solve!, alloc_x, alloc_g, alloc_h, alloc_j
using StaticArrays: SMatrix, MMatrix
using Test

# Regression: `LinearProblem` must accept a non-square `A` (the RHS length
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


# Regression: a singular matrix used to leave `cache.info` set but unchecked,
# so `ldiv!` silently produced NaN/Inf.  It now throws a `SingularException`.
@testset "LU singular matrix throws" begin
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

# Regression: `ldiv!(x, lsolver, b)` used to corrupt the result when `x === b`
# because the permutation gather read entries it had already overwritten.
@testset "LU ldiv! with aliased x === b" begin
    Aa = [4.0 5.0 -2.0; 7.0 -1.0 2.0; 3.0 1.0 4.0]
    ba = [-14.0, 42.0, 28.0]
    xref = [4.0, -4.0, 5.0]
    ls = LinearSolver(LU(), Aa)
    factorize!(ls, Aa)
    v = copy(ba)
    ldiv!(v, ls, v)      # aliased in-place solve
    @test v ≈ xref atol = 1e-10
end

# Regression: an integer input matrix used to throw in `factorize!`
# (no eltype promotion in the cache).  The cache now promotes the element type
# to a fractional type (like `LinearAlgebra.lutype`), so factorization and
# solving work.
@testset "LU integer matrix promotes eltype" begin
    ls = LinearSolver(LU(), [1 2; 3 4])   # construction promotes Int → Float64
    @test eltype(cache(ls).A) == Float64
    factorize!(ls)                        # factorize the (promoted) stored matrix
    x = zeros(2)
    ldiv!(x, ls, [1.0, 1.0])
    @test [1.0 2.0; 3.0 4.0] * x ≈ [1.0, 1.0] atol = 1e-12
end

# The default `LU()` cache matrix type is chosen by size via `_static(A)`: a matrix
# whose leading dimension does not exceed `N_STATIC_THRESHOLD` is stored as a mutable
# static (`MMatrix`) cache, a larger one as a plain `Matrix`.  An explicit
# `static=true`/`false` keyword overrides the size-based choice.
@testset "LU cache type is chosen by the static size threshold" begin
    @test isdefined(SimpleSolvers, :_static)
    @test isdefined(SimpleSolvers, :N_STATIC_THRESHOLD)

    Asmall = [4.0 5.0 -2.0; 7.0 -1.0 2.0; 3.0 1.0 4.0]   # 3×3, ≤ threshold
    @test SimpleSolvers._static(Asmall)
    @test cache(LinearSolver(LU(), Asmall)).A isa MMatrix

    Abig = zeros(SimpleSolvers.N_STATIC_THRESHOLD + 1, SimpleSolvers.N_STATIC_THRESHOLD + 1)
    @test !SimpleSolvers._static(Abig)
    @test cache(LinearSolver(LU(), Abig)).A isa Matrix
    @test !(cache(LinearSolver(LU(), Abig)).A isa MMatrix)

    # A `StaticArray` input is stored as a mutable static (`MMatrix`) cache.
    Astat = SMatrix{3,3}(Asmall)
    @test cache(LinearSolver(LU(), Astat)).A isa MMatrix

    # An explicit `static` keyword overrides the size-based choice.
    @test cache(LinearSolver(LU(; static=true), Asmall)).A isa MMatrix
    @test cache(LinearSolver(LU(; static=false), Asmall)).A isa Matrix
end

# `LUSolverCache` carries a `pivots` field, populated in `factorize!` alongside
# `perms`.
@testset "LUSolverCache has a pivots field" begin
    ls = LinearSolver(LU(), [1.0 2.0; 3.0 4.0])
    @test hasproperty(cache(ls), :pivots)
    @test fieldnames(typeof(cache(ls))) == (:A, :pivots, :perms, :info)
end

# `find_maximum_value` was renamed to `pivot_index` (internal,
# unexported) and returns the index of the largest-|·| entry from `k` onward.
@testset "pivot_index returns index of largest |entry|" begin
    @test !isdefined(SimpleSolvers, :find_maximum_value)
    v = [0.1, -3.0, 2.0, -0.5]
    @test pivot_index(v, 1) == 2      # |-3| is the largest overall
    @test pivot_index(v, 3) == 3      # from index 3 on, |2| is the largest
end

# `solve!(x, lsolver, A, b)` copies `A` straight into the existing
# cache instead of allocating a throwaway `LinearProblem`; the result must still
# match the direct solve.
@testset "solve!(x, lsolver, A, b) copies into cache" begin
    A = [1.0 2.0 3.0; 5.0 7.0 11.0; 13.0 17.0 19.0]
    b = [1.0, 2.0, 3.0]
    lsolver = LinearSolver(LU(), zeros(3))
    x = zeros(3)
    solve!(x, lsolver, A, b)
    @test A * x ≈ b atol = 1e-10
end

# The `alloc_*` helpers initialize with `NaN`, which only floating
# point (and complex-of-float) element types support.  An integer input now
# raises a clear error rather than a cryptic `InexactError` deep in setup.
@testset "alloc_* rejects non-NaN-capable eltypes" begin
    @test all(isnan, alloc_x([1.0, 2.0]))
    @test all(isnan, alloc_g([1.0, 2.0]))
    @test isnan(alloc_x(1.0))
    @test_throws ErrorException alloc_x([1, 2])
    @test_throws ErrorException alloc_g([1, 2])
    @test_throws ErrorException alloc_h([1, 2])
    @test_throws ErrorException alloc_j([1, 2], [1, 2])
    @test_throws ErrorException alloc_x(1)
end

# Interface-consistency fixes (verification 2026-07-10):
# (a) `LinearProblem(A, y)` now stores copies of its arguments (it used to
#     NaN-initialize both, so a freshly constructed problem was unusable without
#     an extra `update!`);
# (b) `solve(::LinearSolver, …)` exists as the non-mutating counterpart of
#     `solve!` (it used to be a `MethodError`, while `solve(::LU, …)` worked);
# (c) `solve!(x, lsolver, b)` — documented all along — now has an LU
#     implementation that solves against the stored factorization (it used to
#     always throw the generic "no method implemented" error).
@testset "Linear solver interface consistency" begin
    A = [4.0 1.0; 1.0 3.0]
    b = [1.0, 2.0]
    xref = A \ b

    # (a) constructor keeps values; copies, not aliases
    lp = LinearProblem(A, b)
    @test matrix(lp) == A
    @test SimpleSolvers.rhs(lp) == b
    @test matrix(lp) !== A && SimpleSolvers.rhs(lp) !== b
    A[1, 1] = 99.0
    @test matrix(lp)[1, 1] == 4.0        # a copy, unaffected by caller mutation
    A[1, 1] = 4.0

    # (a) a freshly constructed problem solves without update!
    @test solve(LU(), LinearProblem(A, b)) ≈ xref
    @test solve(LU(), A, b) ≈ xref

    # (b) non-mutating solve through a prebuilt LinearSolver
    lsolver = LinearSolver(LU(), A)
    @test solve(lsolver, LinearProblem(A, b)) ≈ xref
    @test solve(lsolver, A, b) ≈ xref

    # (c) solve! / solve with a bare RHS against the stored factorization
    factorize!(lsolver, A)
    x = zeros(2)
    @test solve!(x, lsolver, b) ≈ xref
    @test solve(lsolver, b) ≈ xref
end
