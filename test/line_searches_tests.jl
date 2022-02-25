
using SimpleSolvers
using SimpleSolvers: LinesearchState, StaticState
using Test


F(x) = x^2-1
D(x) = 2x


@testset "$(rpad("Static",80))" begin
    x₀ = -π
    x₁ = one(x₀)
    x  = copy(x₀)
    f  = F(x)
    g  = D(x)


    o  = UnivariateObjective(F, x)
    ls = StaticState()

    @test ls == LinesearchState(Static(), o)
    @test ls == LinesearchState(Static(), F, x)
    @test ls == LinesearchState(Static(), F, x; D = D)
    @test ls == LinesearchState(Static(1.0), o)
    @test ls == LinesearchState(Static(1.0), F, x)
    @test ls == LinesearchState(Static(1.0), F, x; D = D)

    @test ls == StaticState(o)
    @test ls == StaticState(F, x)
    @test ls == StaticState(F, x; D = D)

    @test solve!(x₁, ls) == ls(x₁) == x₁
    @test solve!(x₀, x₁, ls) == ls(x₀, x₁) == x₁


    o1  = UnivariateObjective(F, x)
    o2  = UnivariateObjective(F, D, x)
    ls1 = Linesearch(x, o1; algorithm = Static())
    ls2 = Linesearch(x, o2; algorithm = Static())
    ls3 = Linesearch(x, o1; algorithm = Static(1.0))

    # @test ls1 == Linesearch(x, F; algorithm = Static())
    # @test ls2 == Linesearch(x, F; algorithm = Static(), D = D)

    @test solve!(x₁, ls1) == ls1(x₁) == x₁
    @test solve!(x₁, ls2) == ls2(x₁) == x₁
    @test solve!(x₁, ls3) == ls3(x₁) == x₁
    @test solve!(x₀, x₁, ls1) == ls1(x₀, x₁) == x₁
    @test solve!(x₀, x₁, ls2) == ls2(x₀, x₁) == x₁
    @test solve!(x₀, x₁, ls3) == ls3(x₀, x₁) == x₁

end



# @testset "$(rpad("Armijo",80))" begin

#     # TODO Test scalars!
        
#     # for Solver in (Armijo,ArmijoQuadratic,ArmijoCubic)
#     for (Solver,SFunc) in ((Armijo, armijo),
#                            (ArmijoQuadratic, armijo_quadratic))

#         n = 1
#         x₀ = -0.5*ones(n)
#         x₁ = +1.0*ones(n)
#         x  = copy(x₀)
#         f  = zeros(n)
#         g  = zeros(n,n)
#         ls = Solver(F!, x, f)
        
#         F!(f,x)
#         J!(g,x)

#         solve!(x, f, g, x₀, x₁, ls)

#         F!(f,x)

#         @test x ≈ zero(x) atol=4E-1
#         @test f ≈ zero(f) atol=8E-2

#         @test solve!(x, f, g, x₀, x₁, ls) == ls(x, f, g, x₀, x₁) == SFunc(F!, x, f, g, x₀, x₁)


#         n = 3
#         x₀ = -ones(n)
#         x₁ = +ones(n)
#         x  = copy(x₀)
#         f  = zeros(n)
#         g  = zeros(n,n)
#         ls = Solver(F!, x, f)
        
#         F!(f,x)
#         J!(g,x)

#         solve!(x, f, g, x₀, x₁, ls)

#         F!(f,x)

#         @test x == zero(x)
#         @test f == zero(f)

#         @test solve!(x, f, g, x₀, x₁, ls) == ls(x, f, g, x₀, x₁) == SFunc(F!, x, f, g, x₀, x₁)

#     end

# end

@testset "$(rpad("Bracketing",80))" begin
    @test bracket_minimum(x -> x^2) == (-SimpleSolvers.DEFAULT_BRACKETING_s, +SimpleSolvers.DEFAULT_BRACKETING_s)
    @test bracket_minimum(x -> (x-1)^2) == (0.64, 2.56)
end


@testset "$(rpad("Bisection",80))" begin

    n  = 1
    x₀ = -π
    x₁ = one(x₀)
    x  = copy(x₀)
    f  = F(x)
    g  = zeros(n)
    
    ls = Linesearch(x, F; algorithm = Bisection(), config = Options(x_abstol = zero(x)))

    x1 = ls(x₀, x₁)
    x2 = ls(x)
    # x3 = solve!(x, f, g, x₀, x₁, ls)
    x4 = solve!(x₀, x₁, ls)
    # x5 = solve!(x, f, g, ls)
    x6 = solve!(x, ls)
    x7 = bisection(F, x₀, x₁; config = Options(x_abstol = zero(x)))
    x8 = bisection(F, x; config = Options(x_abstol = zero(x)))

    # x1, y1 = ls(x₀, x₁)
    # x2, y2 = ls(x)
    # # x3, y3 = solve!(x, f, g, x₀, x₁, ls)
    # x4, y4 = solve!(x₀, x₁, ls)
    # # x5, y5 = solve!(x, f, g, ls)
    # x6, y6 = solve!(x, ls)
    # x7, y7 = bisection(F, x₀, x₁; config = Options(x_abstol = zero(x)))
    # x8, y8 = bisection(F, x; config = Options(x_abstol = zero(x)))

    @test x1 ≈ -1  atol=∛(2eps())
    @test x2 ≈ -1  atol=∛(2eps())
    # @test x3 ≈ -1  atol=∛(2eps())
    @test x4 ≈ -1  atol=∛(2eps())
    # @test x5 ≈ -1  atol=∛(2eps())
    @test x6 ≈ -1  atol=∛(2eps())
    @test x7 ≈ -1  atol=∛(2eps())
    @test x8 ≈ -1  atol=∛(2eps())

    # @test y1 ≈ 0  atol=2eps()
    # @test y2 ≈ 0  atol=2eps()
    # # @test y3 ≈ 0  atol=2eps()
    # @test y4 ≈ 0  atol=2eps()
    # # @test y5 ≈ 0  atol=2eps()
    # @test y6 ≈ 0  atol=2eps()
    # @test y7 ≈ 0  atol=2eps()
    # @test y8 ≈ 0  atol=2eps()


    ls = Linesearch(x, F; algorithm = Bisection(), config = Options(f_abstol = zero(f)))

    x1 = ls(x₀, x₁)
    x2 = ls(x)
    # x3 = solve!(x, f, g, x₀, x₁, ls)
    x4 = solve!(x₀, x₁, ls)
    # x5 = solve!(x, f, g, ls)
    x6 = solve!(x, ls)
    x7 = bisection(F, x₀, x₁; config = Options(f_abstol = zero(f)))
    x8 = bisection(F, x; config = Options(f_abstol = zero(f)))

    # x1, y1 = ls(x₀, x₁)
    # x2, y2 = ls(x)
    # # x3, y3 = solve!(x, f, g, x₀, x₁, ls)
    # x4, y4 = solve!(x₀, x₁, ls)
    # # x5, y5 = solve!(x, f, g, ls)
    # x6, y6 = solve!(x, ls)
    # x7, y7 = bisection(F, x₀, x₁; config = Options(f_abstol = zero(f)))
    # x8, y8 = bisection(F, x; config = Options(f_abstol = zero(f)))

    @test x1 ≈ -1  atol=2eps()
    @test x2 ≈ -1  atol=2eps()
    # @test x3 ≈ -1  atol=2eps()
    @test x4 ≈ -1  atol=2eps()
    # @test x5 ≈ -1  atol=2eps()
    @test x6 ≈ -1  atol=2eps()
    @test x7 ≈ -1  atol=2eps()
    @test x8 ≈ -1  atol=2eps()

    # @test y1 ≈ 0  atol=(2eps())^3
    # @test y2 ≈ 0  atol=(2eps())^3
    # # @test y3 ≈ 0  atol=(2eps())^3
    # @test y4 ≈ 0  atol=(2eps())^3
    # # @test y5 ≈ 0  atol=(2eps())^3
    # @test y6 ≈ 0  atol=(2eps())^3
    # @test y7 ≈ 0  atol=(2eps())^3
    # @test y8 ≈ 0  atol=(2eps())^3

end
