using SimpleSolvers
using Test

n = 1
T = Float64
x = [T(π),]
j = reshape(2x, 1, 1)


function F!(f::AbstractVector, x::AbstractVector, params)
    f .= x .^ 2
end

function J!(g::AbstractMatrix, x::AbstractVector, params)
    g .= 0
    for i in eachindex(x)
        g[i, i] = 2x[i]
    end
    g
end


JPAD = JacobianAutodiff{T}(F!, n, n)
JPFD = JacobianFiniteDifferences{T}(F!, n, n)
JPUS = JacobianFunction{T}(F!, J!)

@test typeof(JPAD) <: JacobianAutodiff
@test typeof(JPFD) <: JacobianFiniteDifferences
@test typeof(JPUS) <: JacobianFunction

@test JPAD == JacobianAutodiff{T}(F!, n)
@test JPAD == JacobianAutodiff(F!, x)
@test JPFD == JacobianFiniteDifferences{T}(F!, n)
@test JPFD == JacobianFiniteDifferences(F!, x)


jad = zero(j)
jfd = zero(j)
jus = zero(j)

JPAD(jad, x, nothing)
JPFD(jfd, x, nothing)
JPUS(jus, x, nothing)

@test jad ≈ j atol = eps()
@test jfd ≈ j atol = 1E-7
@test jus == j


jad1 = zero(j)
jfd1 = zero(j)
jus1 = zero(j)

JPAD(jad1, x, nothing)
JPFD(jfd1, x, nothing)
JPUS(jus1, x, nothing)

@test jad1 == jad
@test jfd1 == jfd
@test jus1 == jus


jad2 = zero(j)
jfd2 = zero(j)
jus2 = zero(j)

JPAD(jad2, x, nothing)
JPFD(jfd2, x, nothing)
JPUS(jus2, x, nothing)

@test jad2 == jad
@test jfd2 == jfd
@test jus2 == jus
