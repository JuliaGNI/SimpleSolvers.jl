
using SimpleSolvers
using SimpleSolvers: LinesearchState, StaticState
using Test

include("optimizers_problems.jl")

f(x) = x^2-1
g(x) = 2x

function F(x)
    # sum(x.^2) - 1
    y = - one(eltype(x))
    for _x in x
        y += _x^2
    end
    return y
end


@testset "$(rpad("Bracketing",80))" begin
    @test bracket_minimum(x -> x^2) == (-SimpleSolvers.DEFAULT_BRACKETING_s, +SimpleSolvers.DEFAULT_BRACKETING_s)
    @test bracket_minimum(x -> (x-1)^2) == (0.64, 2.56)
end


@testset "$(rpad("Static",80))" begin
    # xᵤ = 3.
    x₀ = [-3.]
    x₁ = [+1.]
    δx = x₁ - x₀
    x  = copy(x₀)


    o  = MultivariateObjective(f, x)
    ls = StaticState()

    @test ls == LinesearchState(Static(), o)
    @test ls == LinesearchState(Static(), f, x)
    @test ls == LinesearchState(Static(), f, x; D = g)
    @test ls == LinesearchState(Static(1.0), o)
    @test ls == LinesearchState(Static(1.0), f, x)
    @test ls == LinesearchState(Static(1.0), f, x; D = g)

    @test ls == StaticState(o)
    @test ls == StaticState(f, x)
    @test ls == StaticState(f, x; D = g)

    x1 = copy(x₀); x2 = copy(x₀); @test solve!(x1, δx, ls) == ls(x2, δx) == x₁


    o1  = MultivariateObjective(f, x)
    o2  = MultivariateObjective(f, g, x)
    ls1 = Linesearch(x, o1; algorithm = Static())
    ls2 = Linesearch(x, o2; algorithm = Static())
    ls3 = Linesearch(x, o1; algorithm = Static(1.0))

    # @test ls1 == Linesearch(x, F; algorithm = Static())
    # @test ls2 == Linesearch(x, F; algorithm = Static(), D = D)

    x1 = copy(x₀); x2 = copy(x₀); @test solve!(x1, δx, ls1) == ls1(x2, δx) == x₁
    x1 = copy(x₀); x2 = copy(x₀); @test solve!(x1, δx, ls2) == ls2(x2, δx) == x₁
    x1 = copy(x₀); x2 = copy(x₀); @test solve!(x1, δx, ls3) == ls3(x2, δx) == x₁

end



@testset "$(rpad("Bisection",80))" begin

    x₀ = -3.0
    x₁ = +0.5
    δx = x₁ .- x₀
    x  = copy(x₀)


    x1 = bisection(f, x₀, x₁; config = Options(x_abstol = zero(x)))
    x2 = bisection(f, x; config = Options(x_abstol = zero(x)))

    @test x1 ≈ -1  atol=∛(2eps())
    @test x2 ≈ -1  atol=∛(2eps())

    x1 = bisection(f, x₀, x₁; config = Options(f_abstol = zero(f(x))))
    x2 = bisection(f, x; config = Options(f_abstol = zero(f(x))))
    
    @test x1 ≈ -1  atol=2eps()
    @test x2 ≈ -1  atol=2eps()


    x₀ = [-3.0]
    x₁ = [+0.5]
    xₛ = [-1.0]
    δx = x₁ .- x₀
    x  = copy(x₀)

    x_abstol = zero(eltype(x))
    f_abstol = zero(F(x))

    ls = Linesearch(x, F; algorithm = Bisection(), config = Options(x_abstol = x_abstol))

    x1 = copy(x₀)
    x2 = copy(x₀)
    x3 = copy(x₀)

    ls(x1, δx)
    solve!(x2, δx, ls)
    solve!(x3, δx, ls.state)

    @test x1 ≈ xₛ  atol=∛(2eps())
    @test x2 ≈ xₛ  atol=∛(2eps())
    @test x3 ≈ xₛ  atol=∛(2eps())


    ls = Linesearch(x, F; algorithm = Bisection(), config = Options(f_abstol = f_abstol))

    x1 = copy(x₀)
    x2 = copy(x₀)
    x3 = copy(x₀)

    ls(x1, δx)
    solve!(x2, δx, ls)
    solve!(x3, δx, ls.state)

    @test x1 ≈ xₛ  atol=2eps()
    @test x2 ≈ xₛ  atol=2eps()
    @test x3 ≈ xₛ  atol=2eps()

end


@testset "$(rpad("Backtracking",80))" begin

    for (lsalg,lsfunc) in ((Backtracking, backtracking),)
    # for (lsalg,lsfunc) in ((Backtracking, backtracking)
    #                     (Quadratic, quadratic))

        x₀ = [-3.0]
        x₁ = [+3.0]
        xₛ = [ 0.0]
        δx = x₁ .- x₀
        x  = copy(x₀)

        x1 = copy(x₀)
        x2 = copy(x₀)
        x3 = copy(x₀)
    
        ls = Linesearch(x, F; algorithm = lsalg(), config = Options(x_abstol = zero(eltype(x))))

        ls(x1, δx)
        solve!(x2, δx, ls)
        lsfunc(F, x3, δx; config = Options(x_abstol = zero(eltype(x))))

        @test x1 ≈ xₛ  atol=∛(2eps())
        @test x2 ≈ xₛ  atol=∛(2eps())
        @test x3 ≈ xₛ  atol=∛(2eps())
    

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

    end

end

