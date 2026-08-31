"""
    struct LapackLU <: PivotedLUMethod

A LAPACK-backed LU solver, meant to solve a [`LinearProblem`](@ref).

Where [`LU`](@ref) is a self-contained implementation that works for any number type and for
static matrices, this method delegates the factorization to LAPACK's `getrf`. It is restricted
to the element types LAPACK handles — `Float32`, `Float64`, `ComplexF32` and `ComplexF64` —
and throws an `ArgumentError` naming the type when handed anything else. Any `AbstractMatrix`
storage is accepted, but the cache always holds a plain `Matrix`, because that is what LAPACK
can be pointed at.

Routines that use it are the same as for [`LU`](@ref): [`factorize!`](@ref), [`ldiv!`](@ref)
and [`solve!`](@ref), all shared with the other [`PivotedLUMethod`](@ref)s through
[`PivotedLUCache`](@ref).

# Constructor

```jldoctest; setup = :(using SimpleSolvers)
LapackLU()

# output

LapackLU()
```

# When to use which

[`LU`](@ref) is the better choice for very small systems, where its static-matrix cache avoids
allocation altogether, and it is the only choice for element types LAPACK does not know
about. `LapackLU` is the better choice everywhere else, and is the default for a dense matrix
of a LAPACK element type — see [`default_linear_solver_method`](@ref). Measured on an Apple M4
Max against OpenBLAS, in microseconds for `factorize!` including the copy-in:

| n | `LU(static=false)` | `LapackLU` | [`RecursiveLU`](@ref) |
|---:|---:|---:|---:|
| 12 | 0.24 | 0.63 | **0.14** |
| 64 | 22.9 | 10.8 | **6.65** |
| 128 | 182 | 59.6 | **42.5** |
| 384 | 6526 | **531** | 961 |
| 768 | 51109 | **1613** | 7349 |

`LapackLU`'s triangular solve is the faster one too, by 3.5–4.5× across that whole range:
21.9 µs against `LU`'s 77.3 µs at `n = 384`. See [`RecursiveLU`](@ref) for where the
crossover in the factorization sits and why it depends on the BLAS in use.

The effect on a real problem is large. Measured from
[PoissonBrackets.jl](https://github.com/JuliaGNI/PoissonBrackets.jl), where a Newton step
factorizes a dense ``384 \\times 384`` Jacobian, [`LU`](@ref) accounted for 74 % of the cost
of one implicit time step — about 17 ms against 0.6 ms for the same factorization through
LAPACK.

Allocation is *not* part of the trade-off. Like [`LU`](@ref), and unlike a bare
`LinearAlgebra.lu!`, both [`factorize!`](@ref) and [`ldiv!`](@ref) are allocation-free after
the [`LinearSolver`](@ref) is built: the working matrix and the pivot vector are allocated
once and reused: `getrf!` writes the pivots straight into the cached vector and `getrs!`
reads them back out of it. See [`PivotedLUCache`](@ref).

# Example

```jldoctest; setup = :(using SimpleSolvers, Random; using SimpleSolvers: inv; Random.seed!(123))
A = [1. 2. 3.; 5. 7. 11.; 13. 17. 19.]
v = rand(3)
ls = LinearProblem(A, v)

solve(LapackLU(), ls) ≈ inv(A) * v

# output

true
```
"""
struct LapackLU <: PivotedLUMethod end

function LinearSolverCache(::LapackLU, A::AbstractMatrix{T}) where {T}
    T <: LinearAlgebra.BlasFloat || throw(ArgumentError(
        "LapackLU is restricted to the element types LAPACK provides, i.e. Float32, " *
        "Float64, ComplexF32 and ComplexF64, but got $(T); use LU() instead"))
    _pivoted_lu_cache(A)
end

"""
    _getrf!(method, cache)

Compute the LU factors of `cache.A` in place and record the zero-pivot index in `cache.info`.

This is the *only* thing that differs between the [`PivotedLUMethod`](@ref)s; everything else
— the cache, [`ldiv!`](@ref), [`singular_index`](@ref) and every [`solve!`](@ref) form — is
shared. See [`PivotedLUCache`](@ref).
"""
function _getrf! end

# `getrf!` rather than `lu!`: it takes the pre-allocated `ipiv`, and without the `LU` object
# `lu!` would allocate to wrap the result. See `PivotedLUCache`.
function _getrf!(::LapackLU, c::PivotedLUCache)
    _, _, c.info = LinearAlgebra.LAPACK.getrf!(c.A, c.ipiv; check = false)
    c
end
