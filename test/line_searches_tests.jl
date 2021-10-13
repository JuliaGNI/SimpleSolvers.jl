
using SimpleSolvers
using Test

T = Float64

function F(x::Vector, b::Vector)
    b .= x.^2
end

function J(x::Vector, A::Matrix)
    A .= 0
    for i in eachindex(x)
        A[i,i] = 2x[i]
    end
end


# for Solver in (Armijo,ArmijoQuadratic,ArmijoCubic)
for Solver in (Armijo,)
    n = 1
    x0 = -0.5*ones(T, n)
    x1 =  1.0*ones(T, n)
    ls = Solver(F, x0)

    # @test params(nl) == nl.params
    # @test status(nl) == nl.status

    x  = copy(x0)
    δx = x1 .- x0
    y0 = zeros(n)
    g0 = zeros(n,n)
    F(x,y0)
    J(x,g0)

    solve!(x, δx, x0, y0, g0, ls)
    # println(nl.status.i, ", ", nl.status.rₐ,", ",  nl.status.rᵣ,", ",  nl.status.rₛ)
    b = zero(x)
    F(x,b)
    for xi in x
        @test xi ≈ 0 atol=2E-1
    end
    for bi in b
        @test bi ≈ 0 atol=4E-2
    end


    n = 3
    x0 = -ones(T, n)
    x1 = +ones(T, n)
    ls = Solver(F, x0)

    x  = copy(x0)
    δx = x1 .- x0
    y0 = zeros(n)
    g0 = zeros(n,n)
    F(x,y0)
    J(x,g0)

    solve!(x, δx, x0, y0, g0, ls)
    b = zero(x)
    F(x,b)

    @test x == zeros(n)
    @test b == zeros(n)

end
