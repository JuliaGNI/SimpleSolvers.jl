# `outer!` on a device backend.
#
# `outer!` forms `O` without indexing it one entry at a time, because a scalar index fails on a
# device array. `allowscalar(false)` is what makes this a test rather than a description: without
# it a scalar index on a `JLArray` merely warns. `JLArrays` stands in for
# the device, as it does in `GeometricOptimizers`' own device tests.

using GPUArraysCore: allowscalar
using JLArrays: JLArray
using Random
using SimpleSolvers: outer!
using Test

Random.seed!(2718)

allowscalar(false)

@testset "`outer!` runs on the device" begin
    for T in (Float32, Float64)
        m, n = 4, 3
        x = JLArray(rand(T, m))
        y = JLArray(rand(T, n))
        O = JLArray(zeros(T, m, n))
        outer!(O, x, y)
        @test Array(O) ≈ Array(x) * Array(y)'
    end
end

@testset "`outer!` rejects a mismatched shape" begin
    x = rand(4)
    y = rand(3)
    O = zeros(4, 4)
    @test_throws DimensionMismatch outer!(O, x, y)
end
