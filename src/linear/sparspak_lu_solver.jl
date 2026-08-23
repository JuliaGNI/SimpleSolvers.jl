"""
    struct SparspakLU <: SparseDirectMethod

A sparse LU solver backed by [Sparspak.jl](https://github.com/PetrKryslUCSD/Sparspak.jl),
meant to solve a sparse [`LinearProblem`](@ref).

Only available once Sparspak is loaded; the constructor exists either way, and building a
[`LinearSolver`](@ref) with it says what to load if it is not.

# Constructor

```julia
SparspakLU()
```

# Why this exists alongside [`UmfpackLU`](@ref)

**Element types.** Sparspak is generic in the element type where UMFPACK is not. Probed on a
periodic banded matrix:

| element type | `SparspakLU` | residual | [`UmfpackLU`](@ref) |
|---|---|---|---|
| `Float64` | works | 1.1e-16 | works |
| `Float32` | works | 1.2e-7 | **unsupported** |
| `ComplexF64` | works | 1.1e-16 | works |
| `BigFloat` | works | 1.7e-77 | **unsupported** |
| `Rational{BigInt}` | works | **0.0 — exact** | **unsupported** |

That last row is the point: a sparse solve over ℚ with no rounding at all, which nothing else
here can do. It is also a pure-Julia stack, with no SuiteSparse binary. The `Float32` row is
not a typo either: UMFPACK converts a 32-bit matrix in `lu`/`lu!` but has no 32-bit *solve*, so
[`UmfpackLU`](@ref) refuses those element types at construction rather than failing later
inside `ldiv!`, and this is one of the two methods that cover them.

!!! note "An exact solve goes through `factorize!` and `ldiv!`"
    The allocating convenience forms — [`solve`](@ref), and the `solve!(lsolver, args...)` that
    returns a fresh vector — fill their solution with `NaN`s, which `Rational` and `Integer`
    element types cannot represent, so they raise for exactly the types this method exists to
    serve. Build the [`LinearSolver`](@ref), call [`factorize!`](@ref), and pass your own
    solution vector to `LinearAlgebra.ldiv!`. This is package-wide `SimpleSolvers._nan` policy
    rather than anything specific to `SparspakLU` — dense `solve(LU(), A, b)` raises the same
    way for a `Rational` system.

**Not for speed on `Float64`.** Its factorization is in fact the faster of the two, by
1.3–1.5×, but its triangular solve is about 9× slower, and a Newton loop does at least one
solve per factorization. Measured at `n = 384` on a periodic banded matrix: 59.5 µs
factorize + 32.7 µs solve against UMFPACK's 76.2 + 3.5. Prefer [`UmfpackLU`](@ref) for the
element types it handles.

# Allocation

Neither [`factorize!`](@ref) nor [`ldiv!`](@ref) is allocation-free — about 11 kB and 10 kB
respectively at `n = 384` — and that is inside Sparspak rather than in this wrapper.

# Singularity: reported late

!!! warning
    Sparspak has no zero-pivot index and no status field: a singular matrix **factorizes
    without complaint**, and the solve then returns non-finite numbers. This wrapper closes
    that hole by checking the solution in [`ldiv!`](@ref) and raising `SingularException`
    itself — an `O(n)` check against an `O(nnz · fill)` factorization, so affordable — but two
    consequences cannot be papered over:

    - [`singular_index`](@ref) returns `0` until a solve has actually failed, and is a flag
      rather than a pivot position even then.
    - [`DogLegSolver`](@ref) reads [`singular_index`](@ref) *before* solving, to decide whether
      the Newton leg is available (see [`SimpleSolvers.directions!`](@ref)). With this method
      that check cannot fire, so a singular Jacobian surfaces as a `SingularException` out of
      the solve rather than as a fallback to the steepest-descent leg. Use
      [`UmfpackLU`](@ref) with [`DogLeg`](@ref) where the element type allows it.
"""
struct SparspakLU <: SparseDirectMethod end

# On `AbstractArray` rather than `AbstractMatrix` so the extension's `AbstractMatrix{T}` is
# strictly more specific and adds a method instead of overwriting this one; see
# `RecursiveLU`'s counterpart.
function LinearSolverCache(::SparspakLU, ::AbstractArray)
    error("SparspakLU needs Sparspak.jl to be loaded: add it to your environment and " *
          "`import Sparspak`.")
end
