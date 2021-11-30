
using SimpleSolvers
using Test


function F!(f, x)
    f .= x.^2
end

function J!(g, x)
    g .= 0
    for i in eachindex(x)
        g[i,i] = 2x[i]
    end
end



@testset "$(rpad("NoLineSearch",80))" begin
    for n ∈ (1,3)
        x₀ = rand(n)
        x₁ = rand(n)
        x  = copy(x₀)
        f  = zeros(n)
        g  = zeros(n,n)

        ls = NoLineSearch()

        @test ls == NoLineSearch(F!, x, f)
        @test solve!(x, f, g, x₀, x₁, ls) == x₁
        @test solve!(x, f, g, x₀, x₁, ls) == ls(x, f, g, x₀, x₁)
    end
end



@testset "$(rpad("Armijo",80))" begin

    # TODO Test scalars!
        
    # for Solver in (Armijo,ArmijoQuadratic,ArmijoCubic)
    for (Solver,SFunc) in ((Armijo, armijo),
                           (ArmijoQuadratic, armijo_quadratic))

        n = 1
        x₀ = -0.5*ones(n)
        x₁ = +1.0*ones(n)
        x  = copy(x₀)
        f  = zeros(n)
        g  = zeros(n,n)
        ls = Solver(F!, x, f)
        
        F!(f,x)
        J!(g,x)

        solve!(x, f, g, x₀, x₁, ls)

        F!(f,x)

        @test x ≈ zero(x) atol=4E-1
        @test f ≈ zero(f) atol=8E-2

        @test solve!(x, f, g, x₀, x₁, ls) == ls(x, f, g, x₀, x₁) == SFunc(F!, x, f, g, x₀, x₁)


        n = 3
        x₀ = -ones(n)
        x₁ = +ones(n)
        x  = copy(x₀)
        f  = zeros(n)
        g  = zeros(n,n)
        ls = Solver(F!, x, f)
        
        F!(f,x)
        J!(g,x)

        solve!(x, f, g, x₀, x₁, ls)

        F!(f,x)

        @test x == zero(x)
        @test f == zero(f)

        @test solve!(x, f, g, x₀, x₁, ls) == ls(x, f, g, x₀, x₁) == SFunc(F!, x, f, g, x₀, x₁)

    end

end

@testset "$(rpad("Bracketing",80))" begin
    @test bracket_minimum(x -> x^2) == (-SimpleSolvers.DEFAULT_BRACKETING_s, +SimpleSolvers.DEFAULT_BRACKETING_s)
    @test bracket_minimum(x -> (x-1)^2) == (0.64, 2.56)
end


function F(x)
    x^2-1
end


@testset "$(rpad("Bisection",80))" begin

    n  = 1
    x₀ = -π
    x₁ = one(x₀)
    x  = copy(x₀)
    f  = zeros(n)
    g  = zeros(n,n)
    
    ls = Bisection(F; xtol=0.)

    x1 = ls(x₀, x₁)
    x2 = ls(x)
    x3 = solve!(x, f, g, x₀, x₁, ls)
    x4 = solve!(x₀, x₁, ls)
    x5 = solve!(x, f, g, ls)
    x6 = solve!(x, ls)
    x7 = bisection(F, x₀, x₁; xtol=0.)
    x8 = bisection(F, x; xtol=0.)

    @test x1 ≈ -1  atol=∛(2eps())
    @test x2 ≈ -1  atol=∛(2eps())
    @test x3 ≈ -1  atol=∛(2eps())
    @test x4 ≈ -1  atol=∛(2eps())
    @test x5 ≈ -1  atol=∛(2eps())
    @test x6 ≈ -1  atol=∛(2eps())
    @test x7 ≈ -1  atol=∛(2eps())
    @test x8 ≈ -1  atol=∛(2eps())

    @test F(x1) ≈ 0  atol=2eps()
    @test F(x2) ≈ 0  atol=2eps()
    @test F(x3) ≈ 0  atol=2eps()
    @test F(x4) ≈ 0  atol=2eps()
    @test F(x5) ≈ 0  atol=2eps()
    @test F(x6) ≈ 0  atol=2eps()
    @test F(x7) ≈ 0  atol=2eps()
    @test F(x8) ≈ 0  atol=2eps()


    ls = Bisection(F; ftol=0.)

    x1 = ls(x₀, x₁)
    x2 = ls(x)
    x3 = solve!(x, f, g, x₀, x₁, ls)
    x4 = solve!(x₀, x₁, ls)
    x5 = solve!(x, f, g, ls)
    x6 = solve!(x, ls)
    x7 = bisection(F, x₀, x₁; ftol=0.)
    x8 = bisection(F, x; ftol=0.)

    @test x1 ≈ -1  atol=2eps()
    @test x2 ≈ -1  atol=2eps()
    @test x3 ≈ -1  atol=2eps()
    @test x4 ≈ -1  atol=2eps()
    @test x5 ≈ -1  atol=2eps()
    @test x6 ≈ -1  atol=2eps()
    @test x7 ≈ -1  atol=2eps()
    @test x8 ≈ -1  atol=2eps()

    @test F(x1) ≈ 0  atol=(2eps())^3
    @test F(x2) ≈ 0  atol=(2eps())^3
    @test F(x3) ≈ 0  atol=(2eps())^3
    @test F(x4) ≈ 0  atol=(2eps())^3
    @test F(x5) ≈ 0  atol=(2eps())^3
    @test F(x6) ≈ 0  atol=(2eps())^3
    @test F(x7) ≈ 0  atol=(2eps())^3
    @test F(x8) ≈ 0  atol=(2eps())^3

end
