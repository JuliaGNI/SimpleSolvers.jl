
using SimpleSolvers
using Test

n = 1
T = Float64
x = [T(π),]
j = reshape(2x, 1, 1)


function F!(f::AbstractVector, x::AbstractVector)
    f .= x.^2
end

function J!(g::AbstractMatrix, x::AbstractVector)
    g .= 0
    for i in eachindex(x)
        g[i,i] = 2x[i]
    end
end


JPAD = Jacobian{T}(F!, n; mode = :autodiff)
JPFD = Jacobian{T}(F!, n; mode = :finite)
JPUS = Jacobian{T}(J!, n; mode = :user)

@test typeof(JPAD) <: JacobianAutodiff
@test typeof(JPFD) <: JacobianFiniteDifferences
@test typeof(JPUS) <: JacobianFunction


jad = zero(j)
jfd = zero(j)
jus = zero(j)

JPAD(jad, x)
JPFD(jfd, x)
JPUS(jus, x)

@test jad ≈ j  atol = eps()
@test jfd ≈ j  atol = 1E-7
@test jus ≈ j  atol = 0


jad1 = zero(j)
jfd1 = zero(j)
jus1 = zero(j)

compute_jacobian!(jad1, x, JPAD)
compute_jacobian!(jfd1, x, JPFD)
compute_jacobian!(jus1, x, JPUS)

@test jad1 == jad
@test jfd1 == jfd
@test jus1 == jus


jad2 = zero(j)
jfd2 = zero(j)
jus2 = zero(j)

JPAD = Jacobian{T}(F!, n; mode = :autodiff)
JPFD = Jacobian{T}(F!, n; mode = :finite)
JPUS = Jacobian{T}(J!, n; mode = :user)

compute_jacobian!(jad2, x, JPAD)
compute_jacobian!(jfd2, x, JPFD)
compute_jacobian!(jus2, x, JPUS)

@test jad2 == jad
@test jfd2 == jfd
@test jus2 == jus


jad3 = zero(j)
jfd3 = zero(j)
jus3 = zero(j)

compute_jacobian!(jad3, x, F!; mode = :autodiff)
compute_jacobian!(jfd3, x, F!; mode = :finite)
compute_jacobian!(jus3, x, J!; mode = :user)

@test jad3 == jad
@test jfd3 == jfd
@test jus3 == jus
