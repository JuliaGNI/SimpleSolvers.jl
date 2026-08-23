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
once and reused. See [`PivotedLUCache`](@ref).

The one exception is [`factorize!`](@ref) on a Julia too old for `LAPACK.getrf!(A, ipiv)` —
the 1.10 LTS — where the pivot vector costs one `O(n)` allocation per call. The `O(n^2)`
working matrix is reused on every version, and [`ldiv!`](@ref) is allocation-free on every
version. See [`HAS_PREALLOCATED_GETRF`](@ref).

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
    HAS_PREALLOCATED_GETRF

Whether this Julia's `LinearAlgebra.LAPACK.getrf!` accepts a pre-allocated pivot vector.

`getrf!(A, ipiv)` arrived after the 1.10 LTS, and it is the whole reason [`factorize!`](@ref)
can be allocation-free for [`LapackLU`](@ref). Where it is missing, the one-argument
`getrf!(A)` is used and its pivot vector copied into the cache, which costs one `O(n)`
allocation per factorization — 3.3 kB at `n = 384`. The `O(n^2)` working matrix is reused on
every version, and [`ldiv!`](@ref) is allocation-free on every version, because `getrs!` has
always taken the pivot vector as an argument.

Since [`LapackLU`](@ref) is the default (see [`default_linear_solver_method`](@ref)), on the
LTS that allocation lands once per factorization in a nonlinear solve. Closing it would mean
calling `getrf` through a hand-written `ccall` to pass a pre-allocated pivot vector — the only
place in the package reaching past the stdlib — for a benefit confined to a Julia version with
a finite lifetime. Not worth the maintenance; when the compat floor reaches 1.11 this constant
and its `else` branch delete cleanly.

This is a feature check rather than a `VERSION` comparison, so the exact release it arrived in
does not have to be tracked here.
"""
const HAS_PREALLOCATED_GETRF =
    hasmethod(LinearAlgebra.LAPACK.getrf!, Tuple{Matrix{Float64},Vector{LinearAlgebra.BlasInt}})

"""
    _getrf!(method, cache)

Compute the LU factors of `cache.A` in place and record the zero-pivot index in `cache.info`.

This is the *only* thing that differs between the [`PivotedLUMethod`](@ref)s; everything else
— the cache, [`ldiv!`](@ref), [`singular_index`](@ref) and every [`solve!`](@ref) form — is
shared. See [`PivotedLUCache`](@ref).
"""
function _getrf! end

@static if HAS_PREALLOCATED_GETRF
    # `getrf!` rather than `lu!`: with a pre-allocated `ipiv`, and without the `LU` object
    # `lu!` would allocate to wrap the result. See `PivotedLUCache`.
    function _getrf!(::LapackLU, c::PivotedLUCache)
        _, _, c.info = LinearAlgebra.LAPACK.getrf!(c.A, c.ipiv; check=false)
        c
    end
else
    function _getrf!(::LapackLU, c::PivotedLUCache)
        _, ipiv, info = LinearAlgebra.LAPACK.getrf!(c.A; check=false)
        copyto!(c.ipiv, ipiv)
        c.info = info
        c
    end
end
