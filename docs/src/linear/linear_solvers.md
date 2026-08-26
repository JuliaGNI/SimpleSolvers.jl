# Linear Solvers

Objects of type [`LinearSolver`](@ref) are used to solve [`LinearProblem`](@ref)s, i.e. we want to find ``x`` for given ``A`` and ``y`` such that

```math
    Ax = y
```

is satisfied. 

A linear system can be created with:

```@example linear_system
using SimpleSolvers

A = [(0. + 1e-6) 1. 2.; 3. 4. 5.; 6. 7. 8.]
y = [1., 2., 3.]
ls = LinearProblem(A, y)
nothing # hide
```

Note that we here use the matrix:

```math
A = \begin{pmatrix} 0 + \varepsilon & 1 & 2 \\ 3 & 4 & 5 \\ 6 & 7 & 8 \end{pmatrix}.
```

This matrix would be singular if we had ``\varepsilon = 0`` because ``2\cdot\begin{pmatrix} 3 \\ 4 \\ 5 \end{pmatrix} - \begin{pmatrix} 6 \\ 7 \\ 8 \end{pmatrix} = \begin{pmatrix} 0 \\ 1 \\ 2 \end{pmatrix}.`` So by choosing ``\varepsilon = 10^{-6}`` the matrix is *ill-conditioned*.

We first solve [`LinearProblem`](@ref) with an lu solver (using [`LU`](@ref) and [`solve`](@ref)) in double precision and without pivoting:

```@example linear_system
lu = LU(; pivot = false)
y¹ = solve(lu, ls)
```

We check the result:

```@example linear_system
A * y¹
```

We now do the same in single precision:

```@example linear_system
Aˢ = Float32.(A)
yˢ = Float32.(y)
lsˢ = LinearProblem(Aˢ, yˢ)
y² = solve(lu, lsˢ)
```

and again check the result:

```@example linear_system
Aˢ * y²
```

As we can see the computation of the factorization returns a wrong solution. If we use pivoting however, the problem can also be solved with single precision:

```@example linear_system
lu = LU(; pivot = true)
y³ = solve(lu, lsˢ)
```

```@example linear_system
Aˢ * y³
```

## Solving the System with Built-In Functionality from the `LinearAlgebra` Package

We further try to solve the system with the `inv` operator from the `LinearAlgebra` package. First in double precision:

```@example linear_system
inv(A) * y
```

And also in single precision

```@example linear_system
inv(Aˢ) * yˢ
```

In single precision the result is completely wrong as can also be seen by computing:

```@example linear_system
inv(Aˢ) * Aˢ
```

If we however write:

```@example linear_system
Aˢ \ yˢ
```

we again obtain a correct-looking result, as `LinearAlgebra.\` uses an algorithm very similar to [`factorize!`](@ref) in `SimpleSolvers`.

## Delegating the Factorization to LAPACK

[`LU`](@ref) is a self-contained scalar implementation. That is what makes the comparison
above possible — the pivoting strategy is ours to choose — and for small systems its
static-matrix cache means a factorization allocates nothing at all. It does not scale,
though: the factorization is ``\mathcal{O}(n^3)`` scalar operations with no blocking, so for
a large dense matrix it is an order of magnitude slower than a LAPACK kernel.

[`LapackLU`](@ref) is the same interface with LAPACK's `getrf` underneath:

```@example linear_system
solve(LapackLU(), ls)
```

It is restricted to the element types LAPACK provides (`Float32`, `Float64`, `ComplexF32`
and `ComplexF64`) and throws an `ArgumentError` naming the type for anything else, so
`LU()` remains the only *dense* option for e.g. `BigFloat` (a sparse one of those goes to
[`SparspakLU`](@ref)). What it is *not* is a
trade of allocation for speed: like [`LU`](@ref), it allocates nothing per factorization or
solve once the [`LinearSolver`](@ref) has been built. Everything else is
interchangeable — [`factorize!`](@ref), `LinearAlgebra.ldiv!`, [`solve!`](@ref) and
[`solve`](@ref) behave the same way, and either method can be handed to a nonlinear solver
as its `linear_solver_method`:

```@example linear_system
F(y, x, params) = y .= x .^ 3 .- 2
x = [1.5]
solve!(x, NonlinearProblem(F, zeros(1)), Newton(); linear_solver_method = LapackLU())
```

## Choosing a Method

Five methods, and the choice is made by two things: whether the matrix is sparse, and whether
its element type is one LAPACK knows. `SimpleSolvers.default_linear_solver_method` encodes the
answer, and it is what a nonlinear solver uses when no `linear_solver_method` is given:

| matrix | element type | method |
|---|---|---|
| dense | `Float32`/`Float64`/`ComplexF32`/`ComplexF64` | [`LapackLU`](@ref) |
| dense | anything else (`BigFloat`, `Rational`, …) | [`LU`](@ref) |
| sparse | `Float64`/`ComplexF64` | [`UmfpackLU`](@ref) |
| sparse | anything else | none — an `ArgumentError`; see below |

[`RecursiveLU`](@ref) is never chosen automatically; see below.

### Dense

Measured on an Apple M4 Max against OpenBLAS, `factorize!` in microseconds including the
copy-in:

| n | [`LU`](@ref)`(static=false)` | [`LapackLU`](@ref) | [`RecursiveLU`](@ref) |
|---:|---:|---:|---:|
| 12 | 0.24 | 0.63 | **0.14** |
| 32 | 3.47 | 3.36 | **1.40** |
| 64 | 22.9 | 10.8 | **6.65** |
| 128 | 182 | 59.6 | **42.5** |
| 256 | 1912 | **169** | 287 |
| 384 | 6526 | **531** | 961 |
| 768 | 51109 | **1613** | 7349 |

[`LU`](@ref)'s `MMatrix` path stops at `SimpleSolvers.N_STATIC_THRESHOLD` `= 10`; above that it is a
scalar triple loop, and the triangular solve is a further 3.5–4.5× behind `getrs` throughout.
That is why [`LapackLU`](@ref) rather than [`LU`](@ref) is the default for the element types
it covers.

[`RecursiveLU`](@ref) wins in the middle — but only against OpenBLAS. With AppleAccelerate
loaded on the same machine, [`LapackLU`](@ref) factorizes a `384 × 384` in 285 µs rather than
531 and a `128 × 128` in 26.5 µs rather than 59.6, which moves the crossover down from
`n ≈ 200` to `n ≈ 64`. It also needs a package extension and a heavy dependency, and covers
only `Float32`/`Float64`. Hence: opt in explicitly, after measuring on the machine that
matters.

### Sparse

A sparse method needs the sparsity pattern up front, so it is fixed when the
[`LinearSolver`](@ref) is built — which is exactly what makes the ordering and symbolic
factorization reusable across refactorizations, and where the saving comes from. A dense
matrix is refused rather than converted.

Periodic banded matrices of bandwidth 2, same machine, against a dense [`LapackLU`](@ref) on
the same matrix:

| n | nnz | [`UmfpackLU`](@ref) factorize | `ldiv!` | [`SparspakLU`](@ref) factorize | `ldiv!` | dense [`LapackLU`](@ref) |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 320 | 13.0 | **0.68** | **9.6** | 5.2 | 11.3 |
| 128 | 640 | 26.3 | **1.28** | **19.8** | 11.0 | 59.5 |
| 384 | 1920 | 76.2 | **3.52** | **59.5** | 32.7 | 525 |
| 1024 | 5120 | 207 | **8.6** | **153** | 85.6 | 2525 |
| 4096 | 20480 | 961 | **39.8** | **669** | 348 | — |

Two things to read off. Sparse and dense are a wash around `n = 64` and sparse wins by ~7× at
`n = 384`, so sparsity is worth exploiting only once the matrix is big enough. And
[`SparspakLU`](@ref) has the faster factorization but a ~9× slower solve, which reverses the
comparison in a nonlinear solve — where one factorization is followed by one or more solves.
So [`UmfpackLU`](@ref) is the default.

What [`SparspakLU`](@ref) is for is element types UMFPACK cannot do at all:

| element type | [`SparspakLU`](@ref) | [`UmfpackLU`](@ref) |
|---|---|---|
| `Float64`, `ComplexF64` | works | works |
| `Float32`, `ComplexF32` | works | **unsupported** |
| `BigFloat` | works | **unsupported** |
| `Rational{BigInt}` | works, **exactly** | **unsupported** |

So every element type outside `Float64`/`ComplexF64` has no *default* at all, and that is
deliberate: a sparse matrix is never densified for you.
`SimpleSolvers.default_linear_solver_method` raises an `ArgumentError` naming the two things you
might have meant — [`SparspakLU`](@ref), which keeps the matrix sparse, or a dense method
([`LapackLU`](@ref) for a 32-bit float, [`LU`](@ref) otherwise), which discards the sparsity.
Both are legitimate; which one is right depends on how large the matrix is and whether you can
depend on the Sparspak extension, and neither is something a fallback should decide. Pass one
as `linear_solver_method`.

An exact `Rational` solve goes through [`factorize!`](@ref) and `LinearAlgebra.ldiv!` rather
than the allocating [`solve`](@ref), whose `NaN`-filled solution vector those element types
cannot represent.

Neither sparse method is allocation-free, and that is inside the backends rather than in the
wrapper: [`UmfpackLU`](@ref) allocates ~374 kB per factorization but nothing per solve;
[`SparspakLU`](@ref) allocates ~11 kB and ~10 kB respectively.

!!! warning "Check the residual on a block-structured system"
    Sparse direct solvers relax pivoting to preserve sparsity, and that has a failure mode
    dense factorizations do not. On a matrix whose *blocks* have very different norms — a
    saddle-point or mixed formulation — [`UmfpackLU`](@ref) can return a badly wrong solution
    while reporting success. See its docstring for the measured case. [`SparspakLU`](@ref)
    handled the same matrices; so did dense [`LapackLU`](@ref). It is worth checking
    `norm(A * x - b)` once on a new problem class rather than assuming.

### Sparse Jacobians in a nonlinear solve

To run a sparse Jacobian through a [`NewtonSolver`](@ref) or [`DogLegSolver`](@ref), pass the
pattern as `jacobian_prototype` together with a `DF!` that assembles into it:

```julia
solver = NewtonSolver(x, y; F = F!, DF! = DF!, jacobian_prototype = J0)
```

The prototype's storage is adopted by the Jacobian, the [`LinearProblem`](@ref) and the
[`LinearSolver`](@ref)'s cache, and `SimpleSolvers.default_linear_solver_method` then selects
[`UmfpackLU`](@ref). `DF!` is required: [`JacobianAutodiff`](@ref) and
[`JacobianFiniteDifferences`](@ref) produce dense matrices and would write to
structurally-zero positions, so that combination is refused at construction.

There is no `linear_solver_method` to pass for a `Float64`/`ComplexF64` pattern — that is the
one case with a default — but every other element type needs one, since a sparse matrix is
never densified on your behalf.

The pattern must not change from iteration to iteration — the symbolic factorization was
built for one — and `DF!` writing into positions outside it is an error rather than a silent
reallocation.
