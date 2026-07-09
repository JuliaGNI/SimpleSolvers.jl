using SimpleSolvers

# This example is taken from (Powell, 1970) (the dogleg paper)

function F(y::AbstractVector{T}, x::AbstractVector{T}, params) where {T}
    @assert length(y) == length(x) == 2
    y[1] = x[1]
    y[2] = 10x[1] / (x[1] + one(T) / 10) + 2(x[2]^2)
end

ics(::Type{T}) where {T} = T[3one(T), one(T)]
root(::Type{T}) where {T} = zeros(T, 2)
tol(::Type{T}) where {T} = T == Float64 ? eps(T) : eps(T)

function try_different_solvers(T::DataType)
    # NewtonSolver: with the Phase 2 fixes this now converges on the Powell
    # problem.  Previously it *stagnated* at x ≈ [1.108, 0] and that stalled
    # iterate was falsely reported as converged (bugs.md §3): the backtracking
    # line search shrank α to a denormal (§1.3) and the successive-change
    # convergence criteria treated the resulting zero step as convergence.
    # Fixing the backtracking stall (2.1) and requiring a small residual for
    # convergence (2.2) lets Newton escape the stagnation point and reach the
    # true root.
    x0 = ics(T)
    _root = root(T)
    solver = NewtonSolver(x0, F, copy(x0))

    solve!(x0, solver)
    @test ≈(x0, _root; atol=tol(T))

    # PicardSolver cannot solve this problem, but for a principled reason: since
    # Phase 5 it is a proper (residual-safeguarded) fixed-point iteration
    # x ← x + α(-F(x)), and the Powell map is not a contraction here, so it stalls
    # at a non-root instead of converging.  Crucially it does *not* diverge to NaN
    # or falsely report convergence (Phase 2 residual gate) — it simply runs out of
    # iterations, so the equality assertion below fails as expected.
    x0 = ics(T)
    solver = PicardSolver(x0, F, copy(x0))

    solve!(x0, solver)
    @test_throws AssertionError @assert ≈(x0, _root; atol=tol(T))

    x0 = ics(T)
    solver = DogLegSolver(x0, F, copy(x0))#; verbosity=2

    solve!(x0, solver)
    @test ≈(x0, _root; atol=tol(T))
end

try_different_solvers(Float64)
try_different_solvers(Float32)
