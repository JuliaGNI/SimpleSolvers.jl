# Cost and allocation of the two RankRevealingMethods against LapackLU.
#
# This is the script behind the tables in the `PivotedQR` and `SVDSolver` docstrings and in
# `docs/src/linear/linear_solvers.md`. Run it on the machine whose numbers are being quoted:
#
#     julia --startup-file=no --project=. scripts/benchmark_rank_revealing.jl
#
# The matrices are rank deficient by construction, at rank ⌊n/2⌋ — the case these methods
# exist for, and the one where `PivotedQR` does its `tzrzf` step and `LapackLU` would throw.
# `LapackLU` is therefore timed on a *full-rank* matrix of the same size: it has no answer for
# the deficient one, so its row is a scale rather than a comparison.
#
# The timing is a minimum over repetitions after a warm-up call, rather than BenchmarkTools:
# this package does not depend on it, and a script in `scripts/` that needs an environment of
# its own is a script nobody re-runs. See `Packages/CLAUDE.md`, *Measurement* — everything
# here is measured on an already-built `LinearSolver`, which is how a nonlinear solve uses one.

using LinearAlgebra
using Printf
using Random
using SimpleSolvers
using SimpleSolvers: factorize!

const REPETITIONS = 200

"""
A square `n × n` matrix of element type `T` with exactly `r` non-zero singular values,
logarithmically spread over three decades.
"""
function rank_deficient(T, n, r)
    Random.seed!(4242)
    U = Matrix(qr(randn(T, n, n)).Q)
    V = Matrix(qr(randn(T, n, n)).Q)
    s = zeros(real(T), n)
    s[1:r] .= exp.(range(0, -3, length = r))
    U * Diagonal(T.(s)) * V'
end

"Minimum wall time in microseconds of `f()` over `REPETITIONS` calls, after one warm-up."
function best(f)
    f()
    t = Inf
    for _ in 1:REPETITIONS
        t = min(t, @elapsed f())
    end
    1e6 * t
end

"Time and allocation of `factorize!` and of `ldiv!`, for an already-built `LinearSolver`."
function measure(method, A, b)
    ls = LinearSolver(method, A)
    x = zero(b)
    factorize!(ls, A)
    ldiv!(x, ls, b)
    tf = best(() -> factorize!(ls, A))
    af = @allocated factorize!(ls, A)
    tl = best(() -> ldiv!(x, ls, b))
    al = @allocated ldiv!(x, ls, b)
    (tf, af, tl, al)
end

const T = Float64

println("Julia $(VERSION)")
println(BLAS.get_config())
println()
@printf("%5s %12s %10s %9s %10s %9s\n",
    "n", "method", "fact µs", "fact B", "ldiv µs", "ldiv B")

for n in (13, 40, 128, 384)
    b = randn(T, n)
    Adef = rank_deficient(T, n, n ÷ 2)
    Afull = rank_deficient(T, n, n)

    for (method, A, label) in ((LapackLU(), Afull, "LapackLU*"),
        (PivotedQR(), Adef, "PivotedQR"),
        (SVDSolver(), Adef, "SVDSolver"))
        tf, af, tl, al = measure(method, A, b)
        @printf("%5d %12s %10.2f %9d %10.2f %9d\n", n, label, tf, af, tl, al)
    end
    println()
end

println("* LapackLU is timed on a full-rank matrix: it throws a SingularException on the")
println("  rank-deficient one the other two rows are measured against.")
