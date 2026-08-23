"""
    struct RecursiveLU <: PivotedLUMethod

A [RecursiveFactorization.jl](https://github.com/JuliaLinearAlgebra/RecursiveFactorization.jl)-backed
LU solver, meant to solve a [`LinearProblem`](@ref).

Only available once RecursiveFactorization is loaded; the constructor exists either way, and
building a [`LinearSolver`](@ref) with it says what to load if it is not. It is a package
extension because that dependency is heavy — LoopVectorization, Polyester, StrideArraysCore,
TriangularSolve and VectorizedRNG — and the win is confined to a middle range of sizes.

Restricted to `Float32` and `Float64`, and to `Matrix` storage. RecursiveFactorization has no
complex support, so unlike [`LapackLU`](@ref) this covers two of the four BLAS element types;
the cache constructor says so by name.

Everything except the factorization itself is shared with [`LapackLU`](@ref) — the cache, the
triangular solve, [`singular_index`](@ref) and every [`solve!`](@ref) form — because
RecursiveFactorization writes LAPACK-layout factors with LAPACK's pivot convention, so
`getrs` applies unchanged. See [`PivotedLUMethod`](@ref) and [`PivotedLUCache`](@ref).

Like [`LapackLU`](@ref), [`factorize!`](@ref) and [`ldiv!`](@ref) are allocation-free once the
[`LinearSolver`](@ref) exists, on every Julia version — the pivot vector is pre-allocated in
the cache and RecursiveFactorization takes it as an argument.

# Constructor

```julia
RecursiveLU()
```

# When to use it

For a *middle range* of sizes, where a blocked pure-Julia kernel beats the BLAS's own but the
`O(n^3)` term has not yet taken over. Measured on an Apple M4 Max, `factorize!` in
microseconds including the copy-in:

| n | `LU(static=false)` | [`LapackLU`](@ref) (OpenBLAS) | `RecursiveLU` |
|---:|---:|---:|---:|
| 12 | 0.24 | 0.63 | **0.14** |
| 32 | 3.47 | 3.36 | **1.40** |
| 64 | 22.9 | 10.8 | **6.65** |
| 128 | 182 | 59.6 | **42.5** |
| 256 | 1912 | **169** | 287 |
| 384 | 6526 | **531** | 961 |
| 768 | 51109 | **1613** | 7349 |

!!! warning "The crossover depends on the BLAS, not just on `n`"
    Against OpenBLAS — the default, and what most callers will have — `RecursiveLU` wins for
    roughly `10 < n ≲ 200`. Against a faster `getrf` the window shrinks sharply: with
    AppleAccelerate loaded on the same machine, `LapackLU` factorizes a `384 × 384` in 285 µs
    rather than 531, and a `128 × 128` in 26.5 µs rather than 59.6 — so it wins from about
    `n = 64` upward. Measure on the machine that matters before choosing this.

Below `n ≈ 10` the default `LU()` uses a static-matrix cache and allocates nothing, which is
the better trade there. Above the crossover use [`LapackLU`](@ref). For element types neither
LAPACK nor RecursiveFactorization handles, [`LU`](@ref) remains the only option.

!!! note "No threading option, deliberately"
    RecursiveFactorization's threaded path is not exposed. Measured under `julia -t12` on
    Julia 1.13 it reproduced every sequential timing up to `n = 384` — so it does not
    parallelize at these sizes — and then stalled at `n = 512`, with the worker threads parked
    in a condition wait inside LoopVectorization, on a factorization that takes 2.5 ms
    sequentially. `Val(false)` is hardcoded.
"""
struct RecursiveLU <: PivotedLUMethod end

# Deliberately on `AbstractArray`, one step less specific than the extension's
# `AbstractMatrix{T}`, so that loading the extension *adds* a method instead of overwriting
# this one — method overwriting is an error during precompilation. It is a targeted message
# rather than the generic `LinearSolverCache(method, ::AbstractArray)` fallback, which would
# only say that no method is implemented — true, but not actionable.
function LinearSolverCache(::RecursiveLU, ::AbstractArray)
    error("RecursiveLU needs RecursiveFactorization.jl to be loaded: add it to your " *
          "environment and `import RecursiveFactorization`.")
end
