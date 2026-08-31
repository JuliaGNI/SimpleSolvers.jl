using LinearAlgebra: LinearAlgebra, det, diag, ldiv!, I, SingularException
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
                     jacobianmatrix
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
