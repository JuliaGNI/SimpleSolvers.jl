using LinearAlgebra: LinearAlgebra, det, ldiv!, I, SingularException
using Random: Random
using SimpleSolvers
using SimpleSolvers: LinearSolverMethod, LinearSolverCache, matrix, factorize!, factorization, cache, pivot_index, singular_index, solve, solve!, alloc_x, alloc_g, alloc_h, alloc_j
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
# so `ldiv!` silently produced NaN/Inf.  It throws a `SingularException`.
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

# LinearSolvers only support floating-point problems — real (`AbstractFloat`) or complex
# (`Complex{<:AbstractFloat}`). A non-float input matrix (integer, rational, …) is rejected
# at construction with a clear error rather than being silently promoted.
@testset "LU restricts to floating-point element types" begin
    @test_throws ArgumentError LinearSolver(LU(), [1 2; 3 4])                 # Int
    @test_throws ArgumentError LinearSolver(LU(), [1//1 2//1; 3//1 4//1])     # Rational

    # real and complex float element types are accepted, and the cache keeps the type
    @test eltype(cache(LinearSolver(LU(), [1.0 2.0; 3.0 4.0])).A) == Float64
    @test eltype(cache(LinearSolver(LU(), Float32[1 2; 3 4])).A) == Float32
    @test eltype(cache(LinearSolver(LU(), ComplexF64[1 2; 3 4])).A) == ComplexF64
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
# point (and complex-of-float) element types support.  An integer input
# raises a clear error.
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

# Verify interface-consistency fixes:
# (a) `LinearProblem(A, y)` stores copies of its arguments (it used to
#     NaN-initialize both, so a freshly constructed problem was unusable without
#     an extra `update!`);
# (b) `solve(::LinearSolver, …)` exists as the non-mutating counterpart of
#     `solve!` (it used to be a `MethodError`, while `solve(::LU, …)` worked);
# (c) `solve!(x, lsolver, b)` — documented all along — has an LU
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

# Regression: the bare-RHS forms `solve!(x, lsolver, b)` / `solve(lsolver, b)` solve
# against the *stored* factorization and must not be usable before `factorize!`.  An
# unfactorized cache has `perms` all zero, so `ldiv!` would gather `b[perms[i]] = b[0]`
# and silently return garbage; it now throws instead.
@testset "bare-RHS solve on an unfactorized LinearSolver errors" begin
    A = [4.0 1.0; 1.0 3.0]
    b = [1.0, 2.0]
    lsolver = LinearSolver(LU(), A)          # constructed but not yet factorized
    x = zeros(2)
    @test_throws ArgumentError ldiv!(x, lsolver, b)
    @test_throws ArgumentError solve!(x, lsolver, b)
    @test_throws ArgumentError solve(lsolver, b)
    # after factorizing, the same calls work
    factorize!(lsolver, A)
    @test solve!(x, lsolver, b) ≈ A \ b
end


# --------------------------------------------------------------------------
# LapackLU
# --------------------------------------------------------------------------
#
# The point of this method is that it produces the same answers as `LU` while delegating the
# factorization to LAPACK, so the tests are mostly agreement tests against `LU`.

@testset "LapackLU" begin
    Al = [[+4.0 +5.0 -2.0]
          [+7.0 -1.0 +2.0]
          [+3.0 +1.0 +4.0]]
    xl = [+4.0, -4.0, +5.0]
    bl = [-14.0, +42.0, +28.0]

    ls = LinearSolver(LapackLU(), Al)
    factorize!(ls, Al)
    y = zero(xl)
    ldiv!(y, ls, bl)
    @test y ≈ xl

    # the same answer as the built-in LU, which is the whole contract
    lu_ref = LinearSolver(LU(), Al)
    factorize!(lu_ref, Al)
    y_ref = zero(xl)
    ldiv!(y_ref, lu_ref, bl)
    @test y ≈ y_ref

    # every call form `LU` offers, on the same system: the cache is seeded from `A`, so the
    # single-argument `factorize!` has something to factorize, exactly as for `LU`
    @test ldiv!(zero(xl), factorize!(LinearSolver(LapackLU(), Al)), bl) ≈ xl
    @test solve!(zero(xl), LinearSolver(LapackLU(), Al), LinearProblem(Al, bl)) ≈ xl
    @test solve!(zero(xl), LinearSolver(LapackLU(), Al), Al, bl) ≈ xl
    @test solve!(zero(xl), factorize!(LinearSolver(LapackLU(), Al), Al), bl) ≈ xl
    @test solve!(LinearSolver(LapackLU(), Al), LinearProblem(Al, bl)) ≈ xl
    @test solve!(LinearSolver(LapackLU(), Al), Al, bl) ≈ xl
    @test solve(LinearSolver(LapackLU(), Al), Al, bl) ≈ xl
    @test solve(LapackLU(), LinearProblem(Al, bl)) ≈ xl
    @test solve(LapackLU(), Al, bl) ≈ xl

    # `ldiv!` solves in place, so it has to tolerate `x === b`
    aliased = copy(bl)
    @test ldiv!(aliased, factorize!(LinearSolver(LapackLU(), Al), Al), aliased) ≈ xl

    # on larger random systems, against the reference solution
    Random.seed!(1234)
    for n in (5, 17, 64)
        M = randn(n, n) + n * I
        rhs = randn(n)
        s = LinearSolver(LapackLU(), M)
        factorize!(s, M)
        z = zeros(n)
        ldiv!(z, s, rhs)
        @test M * z ≈ rhs
        # and agreeing with `LU` on the same system
        z_ref = zeros(n)
        ldiv!(z_ref, factorize!(LinearSolver(LU(), M), M), rhs)
        @test z ≈ z_ref
    end

    # the other three element types LAPACK provides
    for T in (Float32, ComplexF32, ComplexF64)
        M = T.(Al) + T(3) * I
        rhs = T.(bl)
        z = zeros(T, 3)
        ldiv!(z, factorize!(LinearSolver(LapackLU(), M), M), rhs)
        @test M * z ≈ rhs
        @test eltype(z) == T
    end

    # using the factorization before it exists is an error, not a wrong answer
    fresh = LinearSolver(LapackLU(), Al)
    @test_throws ArgumentError ldiv!(zero(xl), fresh, bl)
    @test_throws ArgumentError singular_index(fresh)

    # a singular matrix is reported when the factorization is USED, so that a
    # quasi-Newton method may factorize speculatively without being interrupted
    Asing = [1.0 2.0; 2.0 4.0]
    ssing = LinearSolver(LapackLU(), Asing)
    factorize!(ssing, Asing)                       # does not throw
    @test singular_index(ssing) == 2
    @test_throws SingularException ldiv!(zeros(2), ssing, [1.0, 2.0])
    # ... and the reported index is the one `LU` reports too
    ssing_ref = LinearSolver(LU(), Asing)
    factorize!(ssing_ref, Asing)
    @test singular_index(ssing_ref) == singular_index(ssing)

    # a matrix of the wrong size is refused rather than silently copied in piecewise
    @test_throws DimensionMismatch factorize!(LinearSolver(LapackLU(), Al), randn(2, 2))
    @test_throws DimensionMismatch LinearSolver(LapackLU(), randn(3, 4))

    # LAPACK does not know about every number type, and says so by name
    @test_throws ArgumentError LinearSolverCache(LapackLU(), [big(1.0) big(2.0); big(3.0) big(4.0)])

    # the working matrix and the pivot vector are both allocated once and reused, so
    # refactorizing and solving are allocation-free, exactly as they are for `LU`
    Mbig = randn(50, 50) + 50 * I
    zbig = zeros(50)
    rbig = randn(50)
    sbig = LinearSolver(LapackLU(), Mbig)
    factorize!(sbig, Mbig)
    ldiv!(zbig, sbig, rbig)
    @test (@allocated factorize!(sbig, Mbig)) == 0
    @test (@allocated ldiv!(zbig, sbig, rbig)) == 0

    # `factorization` hands out a LinearAlgebra.LU view of those same arrays
    F = factorization(factorize!(LinearSolver(LapackLU(), Al), Al))
    @test F isa LinearAlgebra.LU
    @test det(F) ≈ det(Al)
    @test_throws ArgumentError factorization(LinearSolver(LapackLU(), Al))

    # `LU` solves with scalar loops and so takes any one-based vector; a non-contiguous
    # one cannot be handed to `getrs`, but it must not become an error `LU` does not have
    xstride = view(zeros(6), 1:2:6)
    bstride = view(zeros(6), 1:2:6)
    bstride .= bl
    @test ldiv!(xstride, factorize!(LinearSolver(LapackLU(), Al), Al), bstride) ≈ xl
end

@testset "LapackLU inside a nonlinear solve" begin
    # the substitution has to be invisible to the nonlinear solver, for every method that
    # takes a `linear_solver_method`
    F(y, x, params) = y .= x .^ 3 .- 2
    for method in (Newton(), QuasiNewton(), DogLeg())
        x1 = [1.5]
        x2 = [1.5]
        solve!(x1, NonlinearProblem(F, zeros(1)), method; verbosity=0)
        solve!(x2, NonlinearProblem(F, zeros(1)), method; verbosity=0,
            linear_solver_method=LapackLU())
        @test x1 ≈ x2
        @test x2[1] ≈ cbrt(2.0)
    end
end
