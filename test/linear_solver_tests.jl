using LinearAlgebra: LinearAlgebra, det, diag, ldiv!, I, SingularException, Diagonal, norm,
                     nullspace, opnorm, pinv, qr, rank, triu
using Random: Random
using RecursiveFactorization: RecursiveFactorization
using SparseArrays: SparseArrays, SparseMatrixCSC, sparse, spzeros, nnz, nonzeros,
                    dropzeros!
using Sparspak: Sparspak
using SimpleSolvers
using SimpleSolvers: zero_like, LinearSolverMethod, LinearSolverCache, matrix, factorize!,
                     factorization, cache, pivot_index, singular_index, solve, solve!,
                     alloc_x, alloc_g, alloc_h, alloc_j, alloc_rhs,
                     default_linear_solver_method, fill_nan!, copy_matrix!,
                     add_to_diagonal!, PivotedLUCache, method, linearsolver, linearproblem,
                     jacobianmatrix, RankRevealingMethod, rank_tolerance, singular_values
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

y = [0.1, 0.1]
x = similar(y)
test_solver = LinearSolver(TestMethod(), TestCache(x))

@test_throws ErrorException factorize!(test_solver)

@test_throws ErrorException ldiv!(x, test_solver, y)

A = [[+4.0 +5.0 -2.0]
     [+7.0 -1.0 +2.0]
     [+3.0 +1.0 +4.0]]
x = [+4.0, -4.0, +5.0]
b = [-14.0, +42.0, +28.0]

function solve_with_factorize_and_ldiv(
        solver_method::LinearSolverMethod, xT::AbstractVector{T},
        AT::AbstractMatrix{T}, bT::AbstractVector{T}) where {T}
    ls1 = LinearSolver(solver_method, rand(T, size(AT)...))
    x1 = similar(xT)
    factorize!(ls1, AT)
    ldiv!(x1, ls1, bT)
    x1
end

function solve_with_solve(solver_method, ::AbstractVector{T},
        AT::AbstractMatrix{T}, bT::AbstractVector{T}) where {T}
    solve(solver_method, AT, bT)
end

function solve_with_solve!(solver_method, xT::AbstractVector{T},
        AT::AbstractMatrix{T}, bT::AbstractVector{T}) where {T}
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

test_lu_solver(LU(; static = false), A, b, x)
test_lu_solver(LU(; static = true), A, b, x)

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
    Astat = SMatrix{3, 3}(Asmall)
    @test cache(LinearSolver(LU(), Astat)).A isa MMatrix

    # An explicit `static` keyword overrides the size-based choice.
    @test cache(LinearSolver(LU(; static = true), Asmall)).A isa MMatrix
    @test cache(LinearSolver(LU(; static = false), Asmall)).A isa Matrix

    # Size is not the only condition: an `MArray` cannot `setindex!` a non-isbitstype element,
    # so a small `BigFloat` matrix gets a `Matrix` cache. Without this the default `LU()` —
    # which is what `default_linear_solver_method` picks for a `BigFloat` — built an `MMatrix`
    # and then died inside `factorize!`.
    Aqbig = big.(Asmall)
    @test !SimpleSolvers._static(Aqbig)
    @test cache(LinearSolver(LU(), Aqbig)).A isa Matrix
    @test ldiv!(zeros(BigFloat, 3), factorize!(LinearSolver(LU(), Aqbig), Aqbig), big.([
        1.0, 2.0, 3.0])) ≈ Asmall \ [1.0, 2.0, 3.0]
    # and an explicit `static = true` says why rather than failing later in StaticArrays
    @test_throws ArgumentError LinearSolver(LU(; static = true), Aqbig)
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
    @test_throws ArgumentError LinearSolverCache(
        LapackLU(), [big(1.0) big(2.0);
                     big(3.0) big(4.0)])

    # the working matrix and the pivot vector are both allocated once and reused, so
    # refactorizing and solving are allocation-free, exactly as they are for `LU`
    Mbig = randn(50, 50) + 50 * I
    zbig = zeros(50)
    rbig = randn(50)
    sbig = LinearSolver(LapackLU(), Mbig)
    factorize!(sbig, Mbig)
    ldiv!(zbig, sbig, rbig)
    @test (@allocated ldiv!(zbig, sbig, rbig)) == 0
    @test (@allocated factorize!(sbig, Mbig)) == 0
    # the exact zero is `getrf!(A, ipiv)` filling the cached pivot vector rather than handing
    # back a fresh one, so nothing selects between that and a one-argument fallback
    @test !isdefined(SimpleSolvers, :HAS_PREALLOCATED_GETRF)

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
        solve!(x1, NonlinearProblem(F, zeros(1)), method; verbosity = 0)
        solve!(x2, NonlinearProblem(F, zeros(1)), method; verbosity = 0,
            linear_solver_method = LapackLU())
        @test x1 ≈ x2
        @test x2[1] ≈ cbrt(2.0)
    end
end

# --------------------------------------------------------------------------
# RecursiveLU
# --------------------------------------------------------------------------
#
# Like `LapackLU`, this delegates only the factorization, so these are agreement tests
# against `LU`. It shares `PivotedLUCache` and the LAPACK triangular solve with `LapackLU` —
# RecursiveFactorization writes LAPACK-layout factors with LAPACK's pivot convention — so a
# disagreement here would most likely mean that assumption has broken.

@testset "RecursiveLU" begin
    Ar = [[+4.0 +5.0 -2.0]
          [+7.0 -1.0 +2.0]
          [+3.0 +1.0 +4.0]]
    xr = [+4.0, -4.0, +5.0]
    br = [-14.0, +42.0, +28.0]

    ls = LinearSolver(RecursiveLU(), Ar)
    factorize!(ls, Ar)
    @test ldiv!(zero(xr), ls, br) ≈ xr

    # the cache is the one `LapackLU` uses, which is the point of the shared core
    @test cache(ls) isa PivotedLUCache

    lu_ref = LinearSolver(LU(), Ar)
    factorize!(lu_ref, Ar)
    @test ldiv!(zero(xr), ls, br) ≈ ldiv!(zero(xr), lu_ref, br)

    # every call form, as for `LapackLU`
    @test ldiv!(zero(xr), factorize!(LinearSolver(RecursiveLU(), Ar)), br) ≈ xr
    @test solve!(zero(xr), LinearSolver(RecursiveLU(), Ar), LinearProblem(Ar, br)) ≈ xr
    @test solve!(zero(xr), LinearSolver(RecursiveLU(), Ar), Ar, br) ≈ xr
    @test solve!(zero(xr), factorize!(LinearSolver(RecursiveLU(), Ar), Ar), br) ≈ xr
    @test solve!(LinearSolver(RecursiveLU(), Ar), LinearProblem(Ar, br)) ≈ xr
    @test solve!(LinearSolver(RecursiveLU(), Ar), Ar, br) ≈ xr
    @test solve(RecursiveLU(), LinearProblem(Ar, br)) ≈ xr
    @test solve(RecursiveLU(), Ar, br) ≈ xr

    # aliasing: `ldiv!` must tolerate x === b
    balias = copy(br)
    @test ldiv!(balias, factorize!(LinearSolver(RecursiveLU(), Ar), Ar), balias) ≈ xr

    # agreement with `LU` at a few sizes, including past the static threshold
    Random.seed!(4321)
    for n in (5, 17, 64)
        M = randn(n, n) + n * I
        v = randn(n)
        srec = factorize!(LinearSolver(RecursiveLU(), M), M)
        sref = factorize!(LinearSolver(LU(), M), M)
        @test ldiv!(zeros(n), srec, v) ≈ ldiv!(zeros(n), sref, v)
    end

    # Float32 is supported; the complex BLAS types are not, and RecursiveLU says so rather
    # than silently failing later. This is the one place it is *narrower* than `LapackLU`.
    A32 = Float32.(Ar)
    @test ldiv!(zeros(Float32, 3), factorize!(LinearSolver(RecursiveLU(), A32), A32), Float32.(br)) ≈
          Float32.(xr) rtol = 1e-5
    @test_throws ArgumentError LinearSolverCache(RecursiveLU(), ComplexF64.(Ar))
    @test_throws ArgumentError LinearSolverCache(
        RecursiveLU(), [big(1.0) big(2.0);
                        big(3.0) big(4.0)])

    # using the factorization before it exists is an error, not a wrong answer
    @test_throws ArgumentError ldiv!(zero(xr), LinearSolver(RecursiveLU(), Ar), br)

    # a singular matrix is reported when the factorization is USED, and with the same index
    # `LU` and `LapackLU` report
    Asing = [1.0 2.0; 2.0 4.0]
    ssing = factorize!(LinearSolver(RecursiveLU(), Asing), Asing)
    @test singular_index(ssing) ==
          singular_index(factorize!(LinearSolver(LapackLU(), Asing), Asing))
    @test_throws SingularException ldiv!(zeros(2), ssing, [1.0, 2.0])

    @test_throws DimensionMismatch factorize!(LinearSolver(RecursiveLU(), Ar), randn(2, 2))
    @test_throws DimensionMismatch LinearSolver(RecursiveLU(), randn(3, 4))

    # allocation-free like `LapackLU` above, and for the same reason: RecursiveFactorization
    # takes the pre-allocated pivot vector, exactly as `getrf!(A, ipiv)` does
    Mbig = randn(50, 50) + 50 * I
    zbig = zeros(50)
    rbig = randn(50)
    sbig = factorize!(LinearSolver(RecursiveLU(), Mbig), Mbig)
    ldiv!(zbig, sbig, rbig)
    @test (@allocated ldiv!(zbig, sbig, rbig)) == 0
    @test (@allocated factorize!(sbig, Mbig)) == 0
end

# --------------------------------------------------------------------------
# The sparse direct methods
# --------------------------------------------------------------------------

# A periodic banded matrix — the shape a Galerkin assembly produces, and the case these
# methods exist for. Diagonally dominant so it is non-singular without pivoting luck.
function banded_spd(n, p = 2, T = Float64)
    Is, Js, Vs = Int[], Int[], T[]
    for i in 1:n, k in (-p):p

        push!(Is, i)
        push!(Js, mod1(i + k, n))
        push!(Vs, k == 0 ? T(4 + 2p) : T(1) / T(abs(k) + 1))
    end
    sparse(Is, Js, Vs, n, n)
end

@testset "$(nameof(typeof(m)))" for m in (UmfpackLU(), SparspakLU())
    S = banded_spd(24)
    Sd = Matrix(S)
    bs = collect(range(-1.0, 1.0; length = 24))
    xs = Sd \ bs

    ls = LinearSolver(m, S)
    factorize!(ls, S)
    @test ldiv!(zeros(24), ls, bs) ≈ xs

    # the same answer as the dense methods, which is the contract
    @test ldiv!(zeros(24), factorize!(LinearSolver(LU(), Sd), Sd), bs) ≈
          ldiv!(zeros(24), ls, bs)

    # every call form except the single-argument `factorize!`, which this cache cannot offer
    @test solve!(zeros(24), LinearSolver(m, S), LinearProblem(S, bs)) ≈ xs
    @test solve!(zeros(24), LinearSolver(m, S), S, bs) ≈ xs
    @test solve!(zeros(24), factorize!(LinearSolver(m, S), S), bs) ≈ xs
    @test solve!(LinearSolver(m, S), LinearProblem(S, bs)) ≈ xs
    @test solve!(LinearSolver(m, S), S, bs) ≈ xs
    @test solve(m, LinearProblem(S, bs)) ≈ xs
    @test solve(m, S, bs) ≈ xs
    @test_throws ErrorException factorize!(LinearSolver(m, S))

    # aliasing
    balias = copy(bs)
    @test ldiv!(balias, factorize!(LinearSolver(m, S), S), balias) ≈ xs

    # a few sizes
    for n in (5, 17, 64)
        M = banded_spd(n)
        v = collect(range(-1.0, 1.0; length = n))
        @test ldiv!(zeros(n), factorize!(LinearSolver(m, M), M), v) ≈ Matrix(M) \ v
    end

    # a dense matrix is refused rather than converted: a SparseMatrixCSC with no structural
    # zeros factorizes slower than LapackLU does
    @test_throws ArgumentError LinearSolver(m, Sd)
    @test_throws ArgumentError LinearSolver(m, randn(4, 4))

    # using the factorization before it exists
    @test_throws ArgumentError ldiv!(zeros(24), LinearSolver(m, S), bs)

    # wrong size, and non-square
    @test_throws DimensionMismatch factorize!(LinearSolver(m, S), banded_spd(12))
    @test_throws DimensionMismatch LinearSolver(m, spzeros(3, 4))

    # a singular matrix must raise, not return NaNs. For UMFPACK the factorization knows;
    # for Sparspak it does not, and the wrapper's isfinite guard is what catches it.
    Ssing = copy(S)
    Ssing[:, 3] .= 0.0
    dropzeros!(Ssing)
    lsing = factorize!(LinearSolver(m, Ssing), Ssing)
    @test_throws SingularException ldiv!(zeros(24), lsing, bs)
end

@testset "UmfpackLU specifics" begin
    S = banded_spd(24)
    bs = collect(range(-1.0, 1.0; length = 24))
    ls = factorize!(LinearSolver(UmfpackLU(), S), S)
    x = zeros(24)
    ldiv!(x, ls, bs)
    # the solve is allocation-free even though the factorization is not; see
    # `SparseFactorizationCache`
    @test (@allocated ldiv!(x, ls, bs)) == 0

    # singularity is known at factorization time, unlike Sparspak
    Ssing = copy(S)
    Ssing[:, 3] .= 0.0
    dropzeros!(Ssing)
    @test singular_index(factorize!(LinearSolver(UmfpackLU(), Ssing), Ssing)) != 0
    @test singular_index(ls) == 0

    # complex is supported; a generic element type is not, and it names the alternative
    Sc = SparseMatrixCSC{ComplexF64, Int}(S)
    @test ldiv!(zeros(ComplexF64, 24), factorize!(LinearSolver(UmfpackLU(), Sc), Sc), ComplexF64.(bs)) ≈
          Matrix(S) \ bs
    @test_throws ArgumentError LinearSolver(UmfpackLU(), SparseMatrixCSC{BigFloat, Int}(S))

    # The 32-bit BLAS types are refused at construction, not later: SuiteSparse converts them
    # in `lu`/`lu!` — so the cache builds — but has no 32-bit solve, and `ldiv!` would be a
    # `MethodError` naming an `UmfpackLU{Float64}` the caller never asked for.
    @test_throws ArgumentError LinearSolver(UmfpackLU(), SparseMatrixCSC{Float32, Int}(S))
    @test_throws ArgumentError LinearSolver(UmfpackLU(), SparseMatrixCSC{ComplexF32, Int}(S))
end

# The two routes the error message names have to work, or the advice is worthless.
@testset "a sparse matrix outside Float64/ComplexF64 has no default, but two answers" begin
    S32 = SparseMatrixCSC{Float32, Int}(banded_spd(24))
    b32 = Float32.(collect(range(-1.0, 1.0; length = 24)))
    ref = Matrix{Float32}(S32) \ b32

    @test_throws ArgumentError default_linear_solver_method(S32)
    @test_throws ArgumentError default_linear_solver_method(SparseMatrixCSC{
        ComplexF32, Int}(banded_spd(24)))
    # an explicit dense method still densifies happily — that is the escape hatch the error
    # message points at, not something the default does behind the caller's back

    # `SparspakLU` keeps it sparse
    @test ldiv!(zeros(Float32, 24), factorize!(LinearSolver(SparspakLU(), S32), S32), b32) ≈
          ref rtol = 1e-4
    # `LapackLU` densifies, as it does for any sparse input
    lsd = LinearSolver(LapackLU(), S32)
    @test cache(lsd).A isa Matrix{Float32}
    @test ldiv!(zeros(Float32, 24), factorize!(lsd, S32), b32) ≈ ref rtol = 1e-4

    # and a nonlinear solve says the same thing rather than failing at the first ldiv!
    F32(y, x, params) = y .= x .^ 3 .- 2
    DF32(j, x, params) = (fill!(nonzeros(j), zero(Float32)); for i in axes(j, 1)
        j[i, i] = 3x[i]^2
    end)
    proto32 = SparseMatrixCSC{Float32, Int}(sparse(Float32(1) * I, 4, 4))
    @test_throws ArgumentError NewtonSolver(zeros(Float32, 4), zeros(Float32, 4);
        F = F32, DF! = DF32, jacobian_prototype = proto32)
    s32 = NewtonSolver(zeros(Float32, 4), zeros(Float32, 4); F = F32, DF! = DF32,
        jacobian_prototype = proto32, linear_solver_method = SparspakLU(), verbosity = 0)
    x32 = fill(1.5f0, 4)
    solve!(x32, s32)
    @test all(≈(cbrt(2.0f0); rtol = 1e-4), x32)

    # and the same for a non-BLAS element type, where the dense escape hatch is `LU` rather
    # than `LapackLU` — the branch the message picks between
    protoQ = SparseMatrixCSC{BigFloat, Int}(sparse(1.0I, 4, 4))
    @test_throws ArgumentError NewtonSolver(zeros(BigFloat, 4), zeros(BigFloat, 4);
        F = F32, DF! = DF32, jacobian_prototype = protoQ)
    for lsm in (SparspakLU(), LU())
        sQ = NewtonSolver(zeros(BigFloat, 4), zeros(BigFloat, 4); F = F32, DF! = DF32,
            jacobian_prototype = copy(protoQ), linear_solver_method = lsm, verbosity = 0)
        xQ = fill(big(1.5), 4)
        solve!(xQ, sQ)
        @test all(≈(cbrt(big(2.0)); rtol = 1e-20), xQ)
    end
end

@testset "SparspakLU specifics" begin
    S = banded_spd(24)
    bs = collect(range(-1.0, 1.0; length = 24))

    # the reason this method exists: element types UMFPACK refuses
    for T in (BigFloat, Rational{BigInt})
        ST = SparseMatrixCSC{T, Int}(S)
        bT = T.(bs)
        ls = factorize!(LinearSolver(SparspakLU(), ST), ST)
        x = ldiv!(zeros(T, 24), ls, bT)
        @test maximum(abs, Float64.(ST * x .- bT)) < 1e-25
    end
    # exact over the rationals, which nothing else here can do
    SQ = SparseMatrixCSC{Rational{BigInt}, Int}(S)
    bQ = Rational{BigInt}.(1, 1:24)
    xQ = ldiv!(zeros(Rational{BigInt}, 24), factorize!(LinearSolver(SparspakLU(), SQ), SQ), bQ)
    @test SQ * xQ == bQ

    # singular_index is a flag, and only after a failed solve — see the docstring
    Ssing = copy(S)
    Ssing[:, 3] .= 0.0
    dropzeros!(Ssing)
    lsing = factorize!(LinearSolver(SparspakLU(), Ssing), Ssing)
    @test singular_index(lsing) == 0
    @test_throws SingularException ldiv!(zeros(24), lsing, bs)
    @test singular_index(lsing) != 0
end

# --------------------------------------------------------------------------
# The sparse-aware plumbing helpers
# --------------------------------------------------------------------------

@testset "sparse-aware helpers" begin
    S = sparse([1, 2, 3, 1], [1, 2, 3, 3], [1.0, 2.0, 3.0, 4.0])

    # `fill_nan!` preserves the pattern, which the symbolic factorization depends on
    n0 = nnz(S)
    Sn = copy(S)
    fill_nan!(Sn)
    @test nnz(Sn) == n0
    @test all(isnan, nonzeros(Sn))
    @test all(isnan, fill_nan!(zeros(3, 3)))

    # `copy_matrix!` copies stored values and refuses a different pattern
    S2 = sparse([1, 2, 3, 1], [1, 2, 3, 3], [5.0, 6.0, 7.0, 8.0])
    @test nonzeros(copy_matrix!(copy(S), S2)) == nonzeros(S2)
    @test_throws ArgumentError copy_matrix!(copy(S), sparse([1, 2, 3], [1, 2, 3], [
        1.0, 2.0, 3.0]))
    @test_throws ArgumentError copy_matrix!(copy(S), Matrix(S))
    @test copy_matrix!(zeros(3, 3), S) == Matrix(S)

    # `add_to_diagonal!` is a no-op at α = 0 (the default) and needs a stored diagonal
    @test diag(add_to_diagonal!(copy(S), 10.0)) == [11.0, 12.0, 13.0]
    @test add_to_diagonal!(copy(S), 0.0) == S
    @test diag(add_to_diagonal!(zeros(3, 3), 2.0)) == [2.0, 2.0, 2.0]
    @test_throws ArgumentError add_to_diagonal!(sparse([1, 2], [2, 1], [1.0, 1.0], 3, 3), 1.0)

    # `zero_like` keeps the pattern where `zero` drops it. This is what the line search's
    # private Jacobian buffer needs: `zero(::SparseMatrixCSC)` has no stored entries at all,
    # so a `DF!` assembling into the pattern would find nowhere to write.
    @test nnz(zero_like(S)) == nnz(S)
    @test all(iszero, nonzeros(zero_like(S)))
    @test nnz(zero(S)) == 0        # the behaviour being worked around
    @test zero_like(ones(2, 2)) == zeros(2, 2)

    # the right-hand side stays dense even for a sparse matrix
    @test alloc_rhs(S) isa Vector{Float64}
    @test length(alloc_rhs(S)) == 3
    @test matrix(LinearProblem(S)) isa SparseMatrixCSC
    @test SimpleSolvers.rhs(LinearProblem(S)) isa Vector{Float64}
end

@testset "default_linear_solver_method" begin
    @test default_linear_solver_method(zeros(4, 4)) isa LapackLU
    @test default_linear_solver_method(zeros(Float32, 4, 4)) isa LapackLU
    @test default_linear_solver_method(zeros(ComplexF64, 4, 4)) isa LapackLU
    @test default_linear_solver_method(fill(big(0.0), 4, 4)) isa LU
    @test default_linear_solver_method(banded_spd(8)) isa UmfpackLU
    @test default_linear_solver_method(SparseMatrixCSC{ComplexF64, Int}(banded_spd(8))) isa
          UmfpackLU
    # a sparse 32-bit float has two good explicit answers and no defensible default
    @test_throws ArgumentError default_linear_solver_method(SparseMatrixCSC{Float32, Int}(banded_spd(8)))
    # No sparse element type outside Float64/ComplexF64 gets a default: densifying would
    # discard structure the caller built on purpose, and `SparspakLU` is an extension, so a
    # default reaching for it would depend on what was imported. Both are legitimate choices,
    # so the caller makes them.
    @test_throws ArgumentError default_linear_solver_method(SparseMatrixCSC{BigFloat, Int}(banded_spd(8)))
    @test_throws ArgumentError default_linear_solver_method(SparseMatrixCSC{
        Rational{BigInt}, Int}(banded_spd(8)))
    # ... and the message names the densifying way out that actually works for that element
    # type: `LapackLU` for a BLAS one, `LU` for another float, and neither for a `Rational`,
    # where `LU`'s `lucache_eltype` refuses and `SparspakLU` is the only method that works.
    sparse_default_message(S) =
        try
            default_linear_solver_method(S)
            ""
        catch e
            e.msg
        end
    m32 = sparse_default_message(SparseMatrixCSC{Float32, Int}(banded_spd(8)))
    @test occursin("SparspakLU()", m32) && occursin("or LapackLU()", m32)
    mbig = sparse_default_message(SparseMatrixCSC{BigFloat, Int}(banded_spd(8)))
    @test occursin("SparspakLU()", mbig) && occursin("or LU()", mbig)
    mrat = sparse_default_message(SparseMatrixCSC{Rational{BigInt}, Int}(banded_spd(8)))
    @test occursin("SparspakLU()", mrat)
    @test !occursin("or LU()", mrat) && !occursin("or LapackLU()", mrat)

    # the resolved default reaches the solver
    F(y, x, params) = y .= x .^ 3 .- 2
    @test method(linearsolver(NewtonSolver(zeros(4), zeros(4); F = F))) isa LapackLU
    @test method(linearsolver(NewtonSolver(zeros(BigFloat, 4), zeros(BigFloat, 4); F = F))) isa
          LU
    # and an explicit method still wins
    @test method(linearsolver(NewtonSolver(zeros(4), zeros(4); F = F, linear_solver_method = LU()))) isa
          LU
end

# `LU` densifies a sparse input rather than failing inside the scalar factorization loops.
@testset "LU densifies a sparse matrix" begin
    S = banded_spd(12)
    bs = collect(range(-1.0, 1.0; length = 12))
    ls = LinearSolver(LU(; static = false), S)
    @test cache(ls).A isa Matrix
    @test ldiv!(zeros(12), factorize!(ls, Matrix(S)), bs) ≈ Matrix(S) \ bs
end

@testset "the new methods inside a nonlinear solve" begin
    F(y, x, params) = y .= x .^ 3 .- 2
    for lsm in (RecursiveLU(), LapackLU(), LU())
        for nlm in (Newton(), QuasiNewton(), DogLeg())
            x = [1.5]
            solve!(x, NonlinearProblem(F, zeros(1)), nlm; verbosity = 0,
                linear_solver_method = lsm)
            @test x[1] ≈ cbrt(2.0)
        end
    end
end

# The end-to-end sparse path: a banded nonlinear problem solved with a sparse Jacobian must
# reach the same answer as the dense default, and must not densify anywhere along the way.
@testset "a sparse Jacobian through a nonlinear solve" begin
    n = 50
    Random.seed!(99)
    bvec = randn(n) ./ 10
    function Fb!(f, x, params)
        for i in 1:n
            f[i] = x[i] + 0.1 * x[i]^2 - 0.2 * (x[mod1(i - 1, n)] + x[mod1(i + 1, n)]) -
                   bvec[i]
        end
        nothing
    end
    proto = sparse([1:n; 1:n; 1:n],
        [1:n; [mod1(i - 1, n) for i in 1:n]; [mod1(i + 1, n) for i in 1:n]],
        [ones(n); fill(-0.2, n); fill(-0.2, n)])
    function DFb!(j, x, params)
        fill!(nonzeros(j), 0.0)
        for i in 1:n
            j[i, i] = 1 + 0.2 * x[i]
            j[i, mod1(i - 1, n)] -= 0.2
            j[i, mod1(i + 1, n)] -= 0.2
        end
        nothing
    end

    xdense = zeros(n)
    solve!(xdense, NewtonSolver(zeros(n), zeros(n); F = Fb!))

    # the prototype is a prototype: it is copied, so the caller's matrix survives intact and
    # two solvers built from one do not share a Jacobian
    protovals = copy(nonzeros(proto))
    ssp = NewtonSolver(zeros(n), zeros(n); F = Fb!, DF! = DFb!, jacobian_prototype = proto)
    @test nonzeros(proto) == protovals
    @test jacobianmatrix(cache(ssp)) !== proto
    @test matrix(linearproblem(ssp)) !== proto
    ssp2 = NewtonSolver(zeros(n), zeros(n); F = Fb!, DF! = DFb!, jacobian_prototype = proto)
    @test jacobianmatrix(cache(ssp2)) !== jacobianmatrix(cache(ssp))

    xsparse = zeros(n)
    solve!(xsparse, ssp)

    @test xsparse ≈ xdense
    f = zeros(n)
    Fb!(f, xsparse, nothing)
    @test maximum(abs, f) < 1e-10

    # the storage survives: the Jacobian, the linear problem and the solver cache are all
    # still sparse with the prototype's pattern
    @test method(linearsolver(ssp)) isa UmfpackLU
    @test jacobianmatrix(cache(ssp)) isa SparseMatrixCSC
    @test nnz(jacobianmatrix(cache(ssp))) == nnz(proto)
    @test matrix(linearproblem(ssp)) isa SparseMatrixCSC
    @test nnz(matrix(linearproblem(ssp))) == nnz(proto)

    # a sparse prototype with an autodiff Jacobian would write to structurally-zero
    # positions, so it is refused at construction
    @test_throws ArgumentError NewtonSolver(zeros(n), zeros(n); F = Fb!,
        jacobian_prototype = proto)
    @test_throws ArgumentError DogLegSolver(zeros(n), NonlinearProblem(Fb!, zeros(n));
        jacobian_prototype = proto)
end

# --------------------------------------------------------------------------
# PivotedQR and SVDSolver
# --------------------------------------------------------------------------
#
# The contract of a `RankRevealingMethod` is one sentence — on a rank-deficient system it
# returns the *minimum-norm* solution rather than throwing — and `pinv` is the independent
# statement of it, so that is what these agree against rather than against each other.

"A square `n × n` matrix of element type `T` with exactly `r` non-zero singular values."
function rank_deficient(T, n, r)
    # the shape goes into the seed, so each `(n, r)` in a sweep draws its own stream and the
    # `randn` a caller takes afterwards is not the same one every time
    Random.seed!(4242 + 1000n + r)
    U = Matrix(qr(randn(T, n, n)).Q)
    V = Matrix(qr(randn(T, n, n)).Q)
    s = zeros(real(T), n)
    # `range` refuses a single point between two different endpoints, so rank one is spelled
    # out rather than swept up by the decay
    s[1:r] .= r == 1 ? one(real(T)) : exp.(range(0, -3, length = r))
    U * Diagonal(T.(s)) * V'
end

@testset "$(nameof(typeof(m)))" for m in (PivotedQR(), LapackPivotedQR(), SVDSolver(),
    LapackSVDSolver())
    # a rank-deficient CONSISTENT system: solved exactly, and with no null-space component
    for T in (Float64, Float32, ComplexF64, ComplexF32)
        tol = sqrt(eps(real(T)))
        for (n, r) in ((13, 5), (16, 8), (13, 13))
            A = rank_deficient(T, n, r)
            b = A * randn(T, n)
            xref = pinv(A; rtol = tol) * b

            x = solve(m, A, b)
            @test norm(A * x - b) < 100 * tol * norm(b)        # consistent: solved exactly
            @test norm(x - xref) < 100 * tol * norm(xref)      # and it is the min-norm one
        end
    end

    A = rank_deficient(Float64, 13, 5)
    b = A * randn(13)

    # the minimum-norm property, stated without reference to `pinv`: the solution has no
    # component along the null space, and every other solution of the same system is longer
    N = nullspace(A)
    x = solve(m, A, b)
    @test norm(N' * x) < 1e-10
    for _ in 1:5
        other = x + N * randn(size(N, 2))
        @test norm(A * other - b) < 1e-8              # also a solution, ...
        @test norm(other) > norm(x)                   # ... and longer
    end

    # a singular matrix does not raise: that is the whole point of the addition
    ls = LinearSolver(m, A)
    factorize!(ls, A)
    @test rank(ls) == 5
    @test singular_index(ls) == 0                     # no singular case to report
    @test ldiv!(zeros(13), ls, b) ≈ x

    # an INCONSISTENT system gives the least-squares solution of the truncated problem, which
    # is what `pinv` computes and is documented behaviour rather than an error
    binc = randn(13)
    xls = solve(m, A, binc)
    @test xls ≈ pinv(A; rtol = sqrt(eps())) * binc
    @test norm(A * xls - binc) > 0.1                  # it does not pretend to have solved it

    # an exactly zero matrix has rank 0, and the only solution to minimize over is zero
    Z = zeros(4, 4)
    lz = LinearSolver(m, Z)
    factorize!(lz, Z)
    @test rank(lz) == 0
    @test ldiv!(ones(4), lz, ones(4)) == zeros(4)

    # every call form, on a full-rank system, agreeing with the LU methods
    Af = [[+4.0 +5.0 -2.0]
          [+7.0 -1.0 +2.0]
          [+3.0 +1.0 +4.0]]
    xf = [+4.0, -4.0, +5.0]
    bf = [-14.0, +42.0, +28.0]
    @test ldiv!(zero(xf), factorize!(LinearSolver(m, Af)), bf) ≈ xf
    @test ldiv!(zero(xf), factorize!(LinearSolver(m, Af), Af), bf) ≈ xf
    @test solve!(zero(xf), LinearSolver(m, Af), LinearProblem(Af, bf)) ≈ xf
    @test solve!(zero(xf), LinearSolver(m, Af), Af, bf) ≈ xf
    @test solve!(LinearSolver(m, Af), LinearProblem(Af, bf)) ≈ xf
    @test solve!(LinearSolver(m, Af), Af, bf) ≈ xf
    @test solve(LinearSolver(m, Af), Af, bf) ≈ xf
    @test solve(m, LinearProblem(Af, bf)) ≈ xf
    @test solve(m, Af, bf) ≈ xf

    # `ldiv!` transforms in place, so it has to tolerate `x === b`
    aliased = copy(bf)
    @test ldiv!(aliased, factorize!(LinearSolver(m, Af), Af), aliased) ≈ xf

    # and again on a rank-deficient matrix, which is a different code path: it is the branch
    # where a complete orthogonal factorization applies its second factor back
    bdef = A * randn(13)
    aliased_def = copy(bdef)
    @test ldiv!(aliased_def, factorize!(LinearSolver(m, A), A), aliased_def) ≈
          ldiv!(zeros(13), factorize!(LinearSolver(m, A), A), bdef)

    # using the factorization before it exists is an error, not a zero vector — an
    # unfactorized cache has `rank = 0`, which `ldiv!` would otherwise read as "all null"
    fresh = LinearSolver(m, Af)
    @test_throws ArgumentError ldiv!(zero(xf), fresh, bf)
    @test_throws ArgumentError rank(fresh)
    @test_throws ArgumentError singular_index(fresh)

    # a matrix of the wrong size is refused; which element types are, differs by method and
    # is the subject of its own testset below
    @test_throws DimensionMismatch factorize!(LinearSolver(m, Af), randn(2, 2))
    @test_throws DimensionMismatch LinearSolver(m, randn(3, 4))
end

# The four methods are two algorithms times two kernels, and the kernel is what decides which
# element types are available — exactly the split `LU` and `LapackLU` have.
@testset "the pure-Julia methods take any floating-point type, the LAPACK ones do not" begin
    for m in (PivotedQR(), SVDSolver())
        for M in (zeros(Float16, 2, 2), zeros(ComplexF16, 2, 2), zeros(Float32, 2, 2),
            [big(1.0) big(2.0); big(3.0) big(4.0)])
            @test LinearSolverCache(m, M) isa LinearSolverCache{eltype(M)}
        end
        # a square root is what a reflector and a rotation are built out of, so an exact type
        # is refused rather than silently promoted
        @test_throws ArgumentError LinearSolverCache(m, [1//1 0//1; 0//1 1//1])
        @test_throws ArgumentError LinearSolverCache(m, [1 0; 0 1])
    end

    for m in (LapackPivotedQR(), LapackSVDSolver())
        @test LinearSolverCache(m, zeros(Float32, 2, 2)) isa LinearSolverCache{Float32}
        for M in (zeros(Float16, 2, 2), [big(1.0) big(2.0); big(3.0) big(4.0)],
            [1//1 0//1; 0//1 1//1])
            @test_throws ArgumentError LinearSolverCache(m, M)
        end
    end
end

# Accepting a `BigFloat` into the cache is not the same as factorizing and solving in it, and
# `BigFloat` is the other element type LAPACK does not reach. This drives the prescaling, the
# Householder and Jacobi arithmetic, the rank and the solve end to end at a tolerance only an
# extended-precision type can meet. The reference is written out of the factors the matrix is
# built from, because `pinv` and `nullspace` both go through a `BlasFloat`-only `svd`.
#
# The bound is a multiple of `eps`, not of `sqrt(eps)`: these draws are well conditioned, and both
# methods land within `11 · eps(BigFloat)` of the reference, so `sqrt(eps)` would pass 37 orders of
# magnitude short of the accuracy the kernels actually reach.
@testset "the pure-Julia methods solve a rank-deficient BigFloat system" begin
    tol = 1000 * eps(BigFloat)

    @testset "$(nameof(typeof(m)))" for m in (PivotedQR(), SVDSolver())
        for (n, r) in ((7, 3), (9, 9))
            Random.seed!(20 + 1000n + r)
            U = Matrix(qr(randn(BigFloat, n, n)).Q)
            V = Matrix(qr(randn(BigFloat, n, n)).Q)
            s = zeros(BigFloat, n)
            s[1:r] .= exp.(range(big(0.0), big(-3.0), length = r))
            A = U * Diagonal(s) * V'
            b = A * randn(BigFloat, n)
            xref = V[:, 1:r] * ((U[:, 1:r]' * b) ./ s[1:r])

            lsolver = factorize!(LinearSolver(m, A), copy(A))
            @test rank(lsolver) == r
            x = ldiv!(zeros(BigFloat, n), lsolver, copy(b))
            @test norm(A * x - b) < tol * norm(b)          # consistent: solved exactly
            @test norm(x - xref) < tol * norm(xref)        # and it is the min-norm one
        end
    end
end

# The pure-Julia kernels are reimplementations, and every way one of them can be subtly wrong —
# the conjugation in `Hᴴ = I - τ̄ v vᴴ`, the order the reflectors of `Q` go in against those of
# `Z`, the phase a complex Jacobi rotation carries — shows up as a wrong answer on a complex
# matrix and nowhere else. So the reference they are checked against is LAPACK, and the two
# complex element types are the point of the sweep rather than extra coverage.
@testset "the pure-Julia kernels agree with LAPACK" begin
    for T in (Float64, Float32, ComplexF64, ComplexF32)
        tol = 1000 * sqrt(eps(real(T)))
        for (n, r) in ((13, 5), (16, 8), (13, 13), (9, 8), (20, 1), (4, 0), (1, 1))
            A = rank_deficient(T, n, r)
            b = A * randn(T, n)
            for (generic, lapack) in ((PivotedQR(), LapackPivotedQR()),
                (SVDSolver(), LapackSVDSolver()))
                lg = factorize!(LinearSolver(generic, A), A)
                ll = factorize!(LinearSolver(lapack, A), A)
                @test rank(lg) == rank(ll)
                @test ldiv!(zeros(T, n), lg, copy(b))≈ldiv!(zeros(T, n), ll, copy(b)) rtol=tol atol=tol
            end
        end
    end
end

# What the two decompositions claim about themselves, rather than about the system they solve.
# A `ldiv!` that is right can rest on factors that are not, and then the next element type or
# the next shape is where it shows.
@testset "the pure-Julia factorizations reconstruct their matrix" begin
    for T in (Float64, ComplexF64, Float32, ComplexF32)
        tol = 1000 * sqrt(eps(real(T)))
        for (n, r) in ((13, 5), (16, 8), (11, 11), (6, 1))
            A = rank_deficient(T, n, r)

            c = SimpleSolvers.cache(factorize!(LinearSolver(PivotedQR(), A), A))
            # Q, rebuilt from the reflectors the cache stores
            Q = Matrix{T}(I, n, n)
            for k in n:-1:1
                SimpleSolvers._reflect!(view(c.A, k:n, k), c.tau[k], view(Q, k:n, :))
            end
            @test opnorm(Q' * Q - I) < tol
            @test A[:, c.jpvt]≈(Q*triu(c.A)) ./ c.scale rtol=tol atol=tol
            # Column pivoting is what makes the rank a leading run of the diagonal. The
            # tolerance here is `eps`, not the loose `tol` the reconstructions use: after
            # `_prescale!` the diagonal is bounded by one, so `1000 * sqrt(eps(Float32))` is
            # `0.34` and would assert almost nothing.
            d = [abs(c.A[i, i]) for i in 1:n]
            @test all(d[i] ≥ d[i + 1] - 10 * eps(real(T)) for i in 1:(n - 1))

            s = SimpleSolvers.cache(factorize!(LinearSolver(SVDSolver(), A), A))
            @test s.A*Diagonal(T.(s.S))*s.V'≈A rtol=tol atol=tol
            @test opnorm(s.V' * s.V - I) < tol
            @test s.S≈LinearAlgebra.svdvals(A) rtol=tol atol=tol
            # `U` is orthonormal only where the rank reaches: beyond it the columns are zeroed
            # rather than completed into a basis, which `_decompose!` documents
            rk = s.rank
            rk == 0 || @test opnorm(view(s.A, :, 1:rk)' * view(s.A, :, 1:rk) - I) < tol
        end
    end
end

"""
A `Float16` or `ComplexF16` matrix of numerical rank `r`, built in double precision and rounded.

`rank_deficient` cannot build one directly — it calls `qr`, which LAPACK does not have for
`Float16`. Rounding a double-precision matrix is also what the assertions want: the reference
to compare a `Float16` answer against is the matrix that was actually factorized, not the exact
one it came from.

`decay` sets the spread of the retained spectrum, and the default here is flatter than
`rank_deficient`'s. At `exp(-3)` the effective condition number is 20, which multiplied into
`eps(Float16)` leaves an error bound so loose that it asserts nothing; at `exp(-1)` it is 2.7.
"""
function rank_deficient16(::Type{T}, n, r; decay = -1) where {T}
    W = T === Float16 ? Float64 : ComplexF64
    # the shape goes into the seed, so each `(n, r)` in a sweep draws its own stream and the
    # `randn` a caller takes afterwards is not the same one every time
    Random.seed!(4242 + 1000n + r)
    U = Matrix(qr(randn(W, n, n)).Q)
    V = Matrix(qr(randn(W, n, n)).Q)
    s = zeros(Float64, n)
    s[1:r] .= r == 1 ? [1.0] : exp.(range(0, decay, length = r))
    T.(U * Diagonal(W.(s)) * V')
end

# The reason the two pure-Julia methods exist. `LU` factorizes `Float16` and neither
# LAPACK-backed rank-revealing method can, and NonlinearIntegrators computes in it.
#
# Three claims, with three separately argued bounds rather than one tuned number. `eps(Float16)`
# is `9.8e-4`, so every one of them is coarse in absolute terms and none of them is slack: the
# measured values sit six to twenty times inside.
@testset "Float16" begin
    ε = eps(Float16)

    for T in (Float16, ComplexF16), (n, r) in ((13, 5), (20, 8), (13, 13), (30, 10))

        W = T === Float16 ? Float64 : ComplexF64
        A = rank_deficient16(T, n, r)
        Aw = W.(A)                                  # the reference: the *rounded* matrix
        tol = sqrt(ε)

        for m in (PivotedQR(), SVDSolver())
            ls = factorize!(LinearSolver(m, A), A)

            # the rank, which is what a rank-revealing method is for
            @test rank(ls) == LinearAlgebra.rank(Aw; rtol = tol) == r

            # backward stability, on a system consistent by construction. This is the claim
            # that does not depend on the conditioning of the problem.
            b = T.(Aw * randn(W, n))
            x = ldiv!(zeros(T, n), ls, copy(b))
            @test norm(Aw * W.(x) - W.(b)) ≤ 5 * n * ε * opnorm(Aw) * norm(x)

            # and the answer itself, against double precision. The bound is `n · eps` times
            # the effective condition number of the retained spectrum, which `decay = -1`
            # holds at 2.7 — see `rank_deficient16`.
            @test W.(x)≈pinv(Aw; rtol = tol)*W.(b) rtol=5*n*ε atol=5*n*ε
        end

        # the spectrum, absolutely and scaled by σ₁ — a *relative* bound per singular value is
        # not something backward stability promises, and asserting one would be asserting luck
        s = singular_values(factorize!(LinearSolver(SVDSolver(), A), A))
        @test maximum(abs, s - LinearAlgebra.svdvals(Aw)) ≤ 20 * n * ε * s[1]
    end

    # the LAPACK pair cannot be reached at this element type at all, which is the gap
    @test_throws ArgumentError LinearSolver(LapackPivotedQR(), zeros(Float16, 3, 3))
    @test_throws ArgumentError LinearSolver(LapackSVDSolver(), zeros(Float16, 3, 3))

    # a full-rank `Float16` system, against `LU` — the one other method here that reaches it
    Af = Float16[4 5 -2; 7 -1 2; 3 1 4]
    bf = Float16[-14, 42, 28]
    xlu = solve(LU(), Af, bf)
    for m in (PivotedQR(), SVDSolver())
        @test solve(m, Af, bf) ≈ xlu rtol=20 * ε
    end

    # every entry subnormal, which is the case `_prescale!` has to cap its exponent for: the
    # power of two that would bring the maximum to one is not representable, and an infinite
    # factor turns the matrix into `Inf`, the rank into zero and the solution into zero
    As = fill(Float16(1e-5), 2, 2)
    @test issubnormal(maximum(abs, As))
    for m in (PivotedQR(), SVDSolver())
        @test rank(factorize!(LinearSolver(m, As), As)) == 1
        @test solve(m, copy(As), Float16[1e-5, 1e-5]) ≈ Float16[0.5, 0.5] rtol=20 * ε
    end

    # the same matrix against a right-hand side of ones, where the reciprocal of the one
    # non-zero singular value overflows `Float16` although the solution does not: the exact
    # minimum-norm `x` is 4.99e4 against a `floatmax` of 6.55e4. `SVDSolver` divides by the
    # singular values of the *scaled* matrix and scales `x` afterwards for this reason; the
    # sum of the two entries is what the rank-one system actually determines.
    xref = Float64(1) / (2 * Float64(As[1, 1]))
    for m in (PivotedQR(), SVDSolver())
        x = solve(m, copy(As), Float16[1, 1])
        @test all(isfinite, x)
        @test sum(Float64.(x)) ≈ 2 * xref rtol=20 * ε
    end
end

# `factorize!` clears `factorized` before it decomposes, so a decomposition that throws leaves
# the cache unusable rather than answering for the matrix before it.
#
# The one throw site in a real `_decompose!` is the Jacobi sweep cap, and no constructible
# matrix reaches it — a non-finite entry stops the sweep rather than prolonging it, and the
# worst conditioning tried here converges in eight of the thirty. So the contract is asserted
# against a method that decomposes normally and then fails on demand.
struct ThrowingRankRevealing <: SimpleSolvers.RankRevealingMethod{Missing} end

function SimpleSolvers.LinearSolverCache(::ThrowingRankRevealing, A::AbstractMatrix)
    SimpleSolvers.LinearSolverCache(SVDSolver(), A)
end

function SimpleSolvers._decompose!(::ThrowingRankRevealing, c::SimpleSolvers.SVDCache)
    any(isnan, c.A) && error("this decomposition fails on demand")
    SimpleSolvers._decompose!(SVDSolver(), c)
end

@testset "a failed refactorization invalidates the cache" begin
    ls = LinearSolver(ThrowingRankRevealing(), zeros(2, 2))
    factorize!(ls, [2.0 0.0; 0.0 3.0])
    @test SimpleSolvers.cache(ls).factorized
    @test rank(ls) == 2

    @test_throws ErrorException factorize!(ls, [NaN 0.0; 0.0 3.0])

    # `checkfactorized` is what every reader goes through, so clearing the flag is what stops
    # `rank`, `singular_values` and `ldiv!` answering for the matrix before it
    @test !SimpleSolvers.cache(ls).factorized
    @test_throws ArgumentError rank(ls)
end

# The three inner products of a Jacobi rotation are summed in `Float32` for a `Float16` matrix,
# and this is the measurement that says they have to be. Summed in `Float16`, the sweep cannot
# drive the columns closer to orthogonal than 32 · eps — coarser than the rank tolerance that
# then has to read a rank off them.
@testset "the Float16 accumulator is load-bearing" begin
    @test SimpleSolvers._accumulator(Float16) === Float32
    @test SimpleSolvers._accumulator(ComplexF16) === ComplexF32
    @test SimpleSolvers._accumulator(Float64) === Float64
    @test SimpleSolvers._accumulator(BigFloat) === BigFloat

    x = fill(Float16(0.01), 64)
    @test SimpleSolvers._acc_dot(x, x) isa Float32
    @test SimpleSolvers._acc_norm(x) isa Float32

    # the whole claim, stated as a comparison rather than against a hand-computed constant:
    # summing the same products in `Float16` is four orders of magnitude worse
    exact = 64 * Float64(x[1])^2
    naive = let s = Float16(0)
        for xᵢ in x
            s += xᵢ * xᵢ
        end
        s
    end
    @test abs(Float64(naive) - exact) >
          100 * abs(Float64(SimpleSolvers._acc_dot(x, x)) - exact)
end

# Both are allocation-free, unlike either LAPACK-backed method, and unlike them that is a
# property callers can rely on rather than a note about the wrapper. Three shapes, because
# `PivotedQR` takes a different branch at each: the second factorization runs, is skipped at
# full rank, and is skipped again on a numerically zero matrix.
#
# Unguarded, as the `LapackLU` and `RecursiveLU` allocation tests above are: these are scalar
# loops over preallocated arrays with no closure whose escape analysis `--check-bounds=yes`
# could change. `BigFloat` is excluded because its arithmetic allocates per operation.
@testset "the pure-Julia factorizations allocate nothing" begin
    for T in (Float64, Float32, Float16, ComplexF64, ComplexF16)
        for (n, r) in ((30, 12), (30, 30), (6, 0))
            M = T.(rank_deficient16(T <: Complex ? ComplexF16 : Float16, n, r))
            rhs = rand(T, n)
            z = zeros(T, n)
            for m in (PivotedQR(), SVDSolver())
                ls = LinearSolver(m, M)
                factorize!(ls, M)                    # warm up both
                ldiv!(z, ls, rhs)
                @test (@allocated factorize!(ls, M)) == 0
                @test (@allocated ldiv!(z, ls, rhs)) == 0
            end
        end
    end
end

# A Jacobi sweep divides by a column norm and by |γ|, and the interesting inputs are the ones
# where those are zero or equal — the cases a formula derived on a generic matrix skips over.
@testset "the Jacobi sweep survives its degenerate inputs" begin
    for T in (Float64, Float16, ComplexF64)
        for (name, A) in (("zero", zeros(T, 5, 5)),
            ("identity", Matrix{T}(I, 5, 5)),
        # every pair has γ = 0 already, so no rotation may be attempted
            ("repeated singular values", Matrix{T}(Diagonal(T[2, 2, 2, 1, 1]))),
        # and here two columns are identical, so one of the pair is pure null space
            ("duplicated columns", T[1 1 0; 2 2 0; 3 3 1]),
            ("one by one", fill(T(3), 1, 1)),
            ("six decades of spread", Matrix{T}(Diagonal(T[1e3, 1, 1e-3, 1e-6, 0]))))
            n = size(A, 1)
            ls = factorize!(LinearSolver(SVDSolver(), A), A)
            x = ldiv!(zeros(T, n), ls, ones(T, n))
            @test all(isfinite, x)
            @test rank(ls) == LinearAlgebra.rank(Float64.(A); rtol = sqrt(eps(real(T))))
            @test issorted(singular_values(ls); rev = true)
        end
    end
end

@testset "the rank tolerance is an option with a per-element-type default" begin
    for M in (PivotedQR, LapackPivotedQR, SVDSolver, LapackSVDSolver)
        # `missing` resolves from the element type; an explicit value is used as given
        @test rank_tolerance(M(), Float64) == sqrt(eps(Float64))
        @test rank_tolerance(M(), Float32) == sqrt(eps(Float32))
        @test rank_tolerance(M(), ComplexF64) == sqrt(eps(Float64))
        @test rank_tolerance(M(; rtol = 1e-10), Float64) === 1e-10
        @test rank_tolerance(M(; rtol = 1e-10), Float32) === 1.0f-10

        # and it decides the rank. The spectrum spans three decades, so a tolerance inside
        # that range truncates it and a tolerance below it does not.
        A = rank_deficient(Float64, 13, 5)
        loose = factorize!(LinearSolver(M(; rtol = 0.1), A), A)
        tight = factorize!(LinearSolver(M(; rtol = 1e-14), A), A)
        @test rank(loose) < 5
        @test rank(tight) == 5

        # dropping a direction changes the answer rather than merely reporting it
        b = A * randn(13)
        @test !(solve!(zeros(13), loose, b) ≈ solve!(zeros(13), tight, b))
    end
end

@testset "$(nameof(typeof(m))) reports the spectrum" for m in (SVDSolver(), LapackSVDSolver())
    A = rank_deficient(Float64, 13, 5)
    ls = LinearSolver(m, A)
    @test_throws ArgumentError singular_values(ls)
    factorize!(ls, A)
    s = singular_values(ls)
    @test s ≈ LinearAlgebra.svdvals(A)
    @test issorted(s; rev = true)
    # the gap the rank was read off: five significant values, eight zero to roundoff
    @test s[5] / s[1] > 1e-2
    @test s[6] / s[1] < 1e-10

    # there is deliberately no such method for either pivoted QR
    for q in (PivotedQR(), LapackPivotedQR())
        @test_throws MethodError singular_values(factorize!(LinearSolver(q, A), A))
    end
end

# The one matrix where the two methods disagree, and the reason `rank`'s docstring stops short
# of promising that `PivotedQR` matches `LinearAlgebra.rank`. A Kahan matrix is the standard
# counterexample to rank-revealing QR: every |R_ii| stays above the tolerance, so the column
# pivoting never exposes the direction the spectrum does. Pinned because both docstrings quote
# these numbers.
#
# `n = 70, θ = 1.15` is chosen for margin, not for the effect — which is visible over a wide
# band of both. Here min|R_ii|/max|R_ii| sits 1.2e5× *above* the tolerance and σ_min/σ_1 sits
# 6.2e5× *below* it, so neither assertion is a near-miss that a different LAPACK build could
# tip. At the more obvious `n = 90, θ = 1.35` the QR margin is a factor of 1.2, which for a
# package whose issue #98 is a BLAS-dependent rank is not enough to pin in a test.
@testset "LapackPivotedQR can miss the rank a Kahan matrix hides" begin
    n, θ = 70, 1.15
    c, s = cos(θ), sin(θ)
    A = [j == i ? s^(i - 1) : j > i ? -c * s^(i - 1) : 0.0 for i in 1:n, j in 1:n]
    tol = sqrt(eps(Float64))

    qrls = factorize!(LinearSolver(LapackPivotedQR(), A), copy(A))
    svdls = factorize!(LinearSolver(LapackSVDSolver(), A), copy(A))

    @test rank(svdls) == LinearAlgebra.rank(A; rtol = tol) == n - 1
    @test rank(qrls) == n           # the documented shortfall, not an accident

    # and the solve inherits it: the SVD reaches the pseudoinverse solution, the QR does not
    b = A * ones(n)
    xref = pinv(A; rtol = tol) * b
    @test ldiv!(zeros(n), svdls, copy(b)) ≈ xref rtol = 1e-6
    @test norm(ldiv!(zeros(n), qrls, copy(b)) - xref) / norm(xref) > 0.1

    # The pure-Julia pivoted QR does *not* miss it, and that is the one place the two kernels
    # give different answers rather than the same one at different speeds. `geqp3` carries
    # running column norms and downdates them; `PivotedQR` recomputes them exactly at every
    # step, and here the drift is what costs LAPACK the pivot order that exposes the small
    # direction. Pinned because `rank`'s docstring tabulates these two rows.
    #
    # Only two, for the reason the paragraph above gives. The effect is visible over a wide
    # band, but the margin — how far min|R_ii| sits below the tolerance — collapses over most
    # of it: 1.2 at (90, 1.35) and 3.5 at (50, 1.15), against 2e4 here and 1e8 at (90, 1.15).
    # A factor of 1.2 is a rounding difference between two Julia versions, and pinning it
    # asserts the version rather than the algorithm.
    for (m, θ) in ((70, 1.15), (90, 1.15))
        cm, sm = cos(θ), sin(θ)
        K = [j == i ? sm^(i - 1) : j > i ? -cm * sm^(i - 1) : 0.0 for i in 1:m, j in 1:m]
        @test rank(factorize!(LinearSolver(PivotedQR(), K), K)) ==
              rank(factorize!(LinearSolver(SVDSolver(), K), K)) ==
              LinearAlgebra.rank(K; rtol = tol) == m - 1
        @test rank(factorize!(LinearSolver(LapackPivotedQR(), K), K)) == m
    end
end

# The shape of NonlinearIntegrators #98, reduced to two unknowns: a residual whose Jacobian is
# exactly rank deficient at every point, and consistent, so a minimum-norm Newton step solves
# it exactly while an LU cannot factorize it at all.
@testset "a rank-deficient Jacobian through a nonlinear solve" begin
    Fdeg(y, x, params) = (y[1] = x[1] + x[2] - 1; y[2] = x[1] + x[2] - 1; y)

    # the LU methods report the singularity, which for them is the correct behaviour
    @test_throws SingularException solve!([0.0, 0.0],
        NonlinearProblem(Fdeg, zeros(2)), Newton(); verbosity = 0,
        linear_solver_method = LapackLU())

    for lsm in (PivotedQR(), LapackPivotedQR(), SVDSolver(), LapackSVDSolver())
        for nlm in (Newton(), QuasiNewton(), DogLeg())
            x = [0.0, 0.0]
            solve!(x, NonlinearProblem(Fdeg, zeros(2)), nlm; verbosity = 0,
                linear_solver_method = lsm)
            y = zeros(2)
            Fdeg(y, x, nothing)
            @test maximum(abs, y) < 1e-12          # a root, ...
            @test x ≈ [0.5, 0.5]                   # ... and the minimum-norm one
        end
    end

    # and the same in `Float16`, which is the reason the pure-Julia pair exists: LAPACK has no
    # `Float16`, so the two `Lapack` methods cannot reach this problem at all
    Fdeg16(y, x, params) = (y[1] = x[1] + x[2] - Float16(1);
        y[2] = x[1] + x[2] - Float16(1);
        y)

    for lsm in (PivotedQR(), SVDSolver())
        for nlm in (Newton(), QuasiNewton(), DogLeg())
            x = Float16[0, 0]
            solve!(x, NonlinearProblem(Fdeg16, zeros(Float16, 2)), nlm; verbosity = 0,
                linear_solver_method = lsm)
            y = zeros(Float16, 2)
            Fdeg16(y, x, nothing)
            @test maximum(abs, y) == 0
            @test x == Float16[0.5, 0.5]
        end
    end

    # and they are interchangeable on a well-posed problem, like every other method here
    F(y, x, params) = y .= x .^ 3 .- 2
    for lsm in (PivotedQR(), SVDSolver())
        for nlm in (Newton(), QuasiNewton(), DogLeg())
            x = [1.5]
            solve!(x, NonlinearProblem(F, zeros(1)), nlm; verbosity = 0,
                linear_solver_method = lsm)
            @test x[1] ≈ cbrt(2.0)
        end
    end
end

# The addition is opt-in. Returning a minimum-norm step by default would turn a singular
# matrix — a bug in almost every caller — into a plausible wrong answer.
@testset "no rank-revealing method is ever a default" begin
    for A in (randn(3, 3), Float32.(randn(3, 3)), ComplexF64.(randn(3, 3)),
        big.(randn(3, 3)), sparse(banded_spd(8)))
        @test !(default_linear_solver_method(A) isa RankRevealingMethod)
    end
    @test default_linear_solver_method(randn(3, 3)) isa LapackLU
end
