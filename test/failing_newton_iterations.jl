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

    # PicardSolver still fails: the Picard direction d = -F(x) is not generally a
    # descent direction for the ‖F‖² merit used by the default line search, so it
    # cannot solve this problem (a separate issue, deferred to Phase 5).
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
