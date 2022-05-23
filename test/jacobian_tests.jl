
using SimpleSolvers
using Test


n = 1
T = Float64
x = [T(π),]
j = reshape(2x, 1, 1)


function F!(f::Vector, x::Vector)
    f .= x.^2
end

function J!(g::Matrix, x::Vector)
    g .= 0
    for i in eachindex(x)
        g[i,i] = 2x[i]
    end
end


JPAD = Jacobian{T}(F!, n; mode = :autodiff, diff_type = :forward)
JPFD = Jacobian{T}(F!, n; mode = :autodiff, diff_type = :finite)
JPUS = Jacobian{T}(J!, n; mode = :user)

@test typeof(JPAD) <: JacobianAutodiff
@test typeof(JPFD) <: JacobianFiniteDifferences
@test typeof(JPUS) <: JacobianFunction


function test_jac(j1, j2, atol)
    for i in eachindex(j1,j2)
        @test j1[i] ≈ j2[i] atol=atol
    end
end


jad = zero(j)
jfd = zero(j)
jus = zero(j)

compute_jacobian!(jad, x, JPAD)
compute_jacobian!(jfd, x, JPFD)
compute_jacobian!(jus, x, JPUS)

test_jac(jad, j, eps())
test_jac(jfd, j, 1E-7)
test_jac(jus, j, 0)


jad1 = zero(j)
jfd1 = zero(j)
jus1 = zero(j)

compute_jacobian!(jad1, x, F!; mode = :autodiff, diff_type = :forward)
compute_jacobian!(jfd1, x, F!; mode = :autodiff, diff_type = :finite)
compute_jacobian!(jus1, x, J!; mode = :user)

test_jac(jad, jad1, 0)
test_jac(jfd, jfd1, 0)
test_jac(jus, jus1, 0)


jad2 = zero(j)
jfd2 = zero(j)

compute_jacobian_ad!(jad2, x, F!)
compute_jacobian_fd!(jfd2, x, F!)

test_jac(jad, jad2, 0)
test_jac(jfd, jfd2, 0)
