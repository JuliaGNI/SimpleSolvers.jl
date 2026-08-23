"""
    struct UmfpackLU <: SparseDirectMethod

A sparse LU solver backed by SuiteSparse's UMFPACK, meant to solve a sparse
[`LinearProblem`](@ref).

No extension and no new dependency: UMFPACK ships inside the `SparseArrays` standard library,
which this package depends on for the sparse-Jacobian plumbing anyway. Restricted to the
element types UMFPACK provides — `Float64` and `ComplexF64` — and it names the alternatives for
anything else. That includes `Float32` and `ComplexF32`: SuiteSparse converts them in `lu` and
`lu!` but has no 32-bit *solve*, so a 32-bit factorization would be a `ldiv!` `MethodError`
waiting to happen rather than a working narrow path. Use [`SparspakLU`](@ref) to keep a 32-bit
matrix sparse, or [`LapackLU`](@ref) to densify it.

This is the method to reach for when the Jacobian is genuinely sparse and its element type is
a standard one. It is also what [`default_linear_solver_method`](@ref) selects in that case.

# Constructor

```julia
UmfpackLU()
```

# Why sparse is worth it, and when it is not

Measured on an Apple M4 Max, periodic banded matrices of bandwidth 2, `factorize!` and
`ldiv!` in microseconds against a dense [`LapackLU`](@ref) on the same matrix:

| n | nnz | `UmfpackLU` factorize | `ldiv!` | dense `LapackLU` factorize |
|---:|---:|---:|---:|---:|
| 64 | 320 | 13.0 | 0.68 | **11.3** |
| 128 | 640 | **26.3** | 1.28 | 59.5 |
| 384 | 1920 | **76.2** | 3.52 | 525 |
| 1024 | 5120 | **207** | 8.6 | 2525 |
| 4096 | 20480 | **961** | 39.8 | — |

So the two are a wash around `n = 64` and sparse wins by roughly 7× at `n = 384` and 12× at
`n = 1024`. A dense matrix handed to this method is an error rather than a conversion — see
[`checksparse`](@ref) — because a `SparseMatrixCSC` with no structural zeros factorizes
*slower* than [`LapackLU`](@ref) does.

# Against [`SparspakLU`](@ref)

Sparspak's factorization is the faster of the two, by about 1.3–1.5×. Its triangular solve is
about 9× slower, which more than reverses that in a Newton loop, where one factorization is
followed by one or more solves. `UmfpackLU` is the better default; [`SparspakLU`](@ref) is for
the element types UMFPACK cannot do at all.

# Allocation

`ldiv!` is allocation-free. [`factorize!`](@ref) is not — `lu!` allocates about 374 kB at
`n = 384` inside SuiteSparse — and unlike the dense methods that cannot be fixed from here.
See [`SparseFactorizationCache`](@ref).

# Singularity

UMFPACK reports singularity as a status, not as a pivot index, so [`singular_index`](@ref) is
a flag: `0` on success, non-zero otherwise. `factorize!` records it and [`ldiv!`](@ref) raises
`SingularException`, matching [`LapackLU`](@ref)'s "report it when it is used" contract so
that a quasi-Newton method that factorizes speculatively is not interrupted.

!!! warning "It can be silently wrong on block-structured systems"
    A sparse direct solver relaxes pivoting to preserve sparsity, and UMFPACK's ordering plus
    threshold pivoting is not always up to a matrix whose *blocks* have very different norms —
    a saddle-point or mixed formulation, for instance. Measured on the `2N × 2N` Newton matrix
    of PoissonBrackets.jl's mixed two-field formulation, whose four banded blocks span seven
    orders of magnitude: from `n = 1536` upward the computed solution is wrong by a factor of
    150, with a linear residual `2000×` the right-hand side, while `issuccess` returns `true`
    and no exception is raised. Dense [`LapackLU`](@ref) on the same matrix is accurate to
    `1e-12`, and so is [`SparspakLU`](@ref), whose different ordering and pivoting handle it.

    This is not general ill-conditioning: on synthetic banded matrices UMFPACK is accurate to
    `1e-14` at condition numbers of `1e7`, well past where these failures start.

    So: for a block-structured Jacobian, check the residual of a solve before trusting it, and
    prefer [`SparspakLU`](@ref) if it does not hold up. `UmfpackLU` remains the right default
    for the banded and mesh-like patterns a discretisation usually produces, where it is both
    faster and accurate.
"""
struct UmfpackLU <: SparseDirectMethod end

function LinearSolverCache(method::UmfpackLU, A::AbstractMatrix{T}) where {T}
    T <: Union{Float64,ComplexF64} || throw(ArgumentError(
        "UmfpackLU is restricted to the element types UMFPACK provides, i.e. Float64 and " *
        "ComplexF64, but got $(T); use SparspakLU() to keep the matrix sparse, or LapackLU() " *
        "to densify it"))
    n = checksparse(method, A)
    # `lu` here is symbolic *and* numeric: `SparseArrays` exposes no symbolic-only entry point,
    # so unlike the `factorize=false` of `SparspakLU`'s counterpart this pays one numeric
    # factorization at construction. `factorized` stays `false` regardless — the cache is not
    # usable until `factorize!` has run — and every refactorization afterwards reuses the
    # ordering and symbolic factorization, which is where the saving is.
    #
    # `check=false`: a singular matrix is reported by `ldiv!`, not here — see the docstring.
    SparseFactorizationCache{T}(lu(A; check=false), n)
end

"""
    factorize!(lsolver::LinearSolver{T,UmfpackLU}, A)

Refactorize `A`, reusing the ordering and symbolic factorization already in the cache.

`SparseArrays`' `lu!` reuses the ordering and symbolic factorization the cache holds, which is
the [`SparseFactorizationCache`](@ref) contract, and raises
`ArgumentError: pattern of the matrix changed` if the pattern does not match the one they were
computed for — which is why the error worth reading comes from the backend and
[`checkpattern`](@ref) only checks the size. Keeping the pattern fixed is still the caller's
job, and the reason [`LinearSolver`](@ref) construction takes the prototype.
"""
function factorize!(lsolver::LinearSolver{T,UmfpackLU}, A::AbstractMatrix{T}) where {T}
    checkpattern(lsolver, A)
    c = cache(lsolver)
    lu!(c.F, A; check=false)
    # widened to the interface's "index or 0"; UMFPACK does not say which pivot vanished
    c.info = issuccess(c.F) ? 0 : 1
    c.factorized = true
    lsolver
end

# UMFPACK's three-argument `ldiv!` refuses aliased arrays, but every other method here
# tolerates `x === b` — `solve!(x, lsolver, x)` is a legitimate call — so route the aliased
# case through the two-argument form, which solves in place. Both are allocation-free.
function _sparse_ldiv!(::UmfpackLU, c::SparseFactorizationCache{T}, x::AbstractVector{T}, b::AbstractVector{T}) where {T}
    if x === b
        ldiv!(c.F, x)
    else
        ldiv!(x, c.F, b)
    end
    x
end
