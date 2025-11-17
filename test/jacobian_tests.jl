using SimpleSolvers
using Test

n = 1
T = Float64
x = [T(π),]
j = reshape(2x, 1, 1)


function F!(f::AbstractVector, x::AbstractVector, params)
    f .= x.^2
end

function J!(g::AbstractMatrix, x::AbstractVector, params)
    g .= 0
    for i in eachindex(x)
        g[i,i] = 2x[i]
    end
    g
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

JPAD(jad, x, nothing)
JPFD(jfd, x, nothing)

@test jad ≈ j  atol = eps()
@test jfd ≈ j  atol = 1E-7
@test jus != j


jad1 = zero(j)
jfd1 = zero(j)
jus1 = zero(j)

compute_jacobian!(jad1, x, JPAD, nothing)
compute_jacobian!(jfd1, x, JPFD, nothing)
@test_throws "You have to provide a `NonlinearProblem` when using `JacobianFunction`!" compute_jacobian!(jus1, x, JPUS, nothing)

@test jad1 == jad
@test jfd1 == jfd
@test jus1 == jus


jad2 = zero(j)
jfd2 = zero(j)
jus2 = zero(j)

JPAD = Jacobian{T}(F!, n; mode = :autodiff)
JPFD = Jacobian{T}(F!, n; mode = :finite)
JPUS = Jacobian{T}(J!, n; mode = :user)

compute_jacobian!(jad2, x, JPAD, nothing)
compute_jacobian!(jfd2, x, JPFD, nothing)
@test_throws "You have to provide a `NonlinearProblem` when using `JacobianFunction`!" compute_jacobian!(jus2, x, JPUS, nothing)

@test jad2 == jad
@test jfd2 == jfd
@test jus2 == jus


jad3 = zero(j)
jfd3 = zero(j)
jus3 = zero(j)

compute_jacobian!(jad3, x, JacobianAutodiff{Float64}(F!, n, n), nothing)
compute_jacobian!(jfd3, x, JacobianFiniteDifferences{Float64}(F!, n, n), nothing)
@test_throws "You have to provide a `NonlinearProblem` when using `JacobianFunction`!" compute_jacobian!(jus3, x, JacobianFunction{Float64}(), nothing)

@test jad3 == jad
@test jfd3 == jfd
@test jus3 == jus != jfd3
