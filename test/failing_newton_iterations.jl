using SimpleSolvers

# This example is taken from (Powell, 1970) (the dogleg paper)

function F(y::AbstractVector{T}, x::AbstractVector{T}, params) where {T}
    @assert length(y) == length(x) == 2
    y[1] = x[1]
    y[2] = 10x[1] / (x[1] + one(T) / 10) + 2(x[2] ^ 2)
end

ics(::Type{T}) where {T} = T[3one(T), one(T)]
root(::Type{T}) where {T} = zeros(T, 2)

function try_different_solvers(T::DataType)
    x0 = ics(T)
    _root = root(T)
    solver = NewtonSolver(x0, F, copy(x0))

    solve!(x0, solver)
    @test_throws AssertionError @assert x0 ≈ _root

    x0 = ics(T)
    solver = PicardSolver(x0, F, copy(x0))

    solve!(x0, solver)
    @test_throws AssertionError @assert x0 ≈ _root
end

try_different_solvers(Float64)
try_different_solvers(Float32)
