using SimpleSolvers
using SimpleSolvers: LinesearchState, StaticState, compute_new_iterate
using Test

include("optimizers_problems.jl")

f(x) = x^2 - 1
g(x) = 2x

function test_linesearch(algorithm, method; kwargs...)

    x₀ = -3.0
    x₁ = +3.0
    xₛ =  0.0
    δx = x₁ .- x₀

    _f(α) = f(compute_new_iterate(x₀, α, δx))
    _d(α) = g(compute_new_iterate(x₀, α, δx))

    obj1 = UnivariateObjective(_f, xₛ; mode = :autodiff)
    obj2 = UnivariateObjective(_f, _d, xₛ)

    options = Options(x_abstol = zero(x₀))
    
    ls = LinesearchState(algorithm; config = options, kwargs...)

    α1 = ls(obj1)
    α2 = ls(obj2)
    α3 = ls(_f, _d)

    @test compute_new_iterate(x₀, α1, δx) ≈ xₛ  atol=∛(2eps())
    @test compute_new_iterate(x₀, α2, δx) ≈ xₛ  atol=∛(2eps())
    @test compute_new_iterate(x₀, α3, δx) ≈ xₛ  atol=∛(2eps())

    # start with a different initial guess
    α1 = ls(obj1, 2 * one(xₛ))
    α2 = ls(obj2, 2 * one(xₛ))
    α3 = ls(_f, _d, 2 * one(xₛ))

    @test compute_new_iterate(x₀, α1, δx) ≈ xₛ  atol=∛(2eps())
    @test compute_new_iterate(x₀, α2, δx) ≈ xₛ  atol=∛(2eps())
    @test compute_new_iterate(x₀, α3, δx) ≈ xₛ  atol=∛(2eps())

    # use an externally supplied method to compute the step size α
    α1 = method(obj1; config = config = options, kwargs...)
    α2 = method(obj2; config = config = options, kwargs...)
    α3 = method(_f, _d; config = options, kwargs...)

    @test compute_new_iterate(x₀, α1, δx) ≈ xₛ  atol=∛(2eps())
    @test compute_new_iterate(x₀, α2, δx) ≈ xₛ  atol=∛(2eps())
    @test compute_new_iterate(x₀, α3, δx) ≈ xₛ  atol=∛(2eps())

    # start with a different step size
    α1 = method(obj1, 2 * one(xₛ); config = options, kwargs...)
    α2 = method(obj2, 2 * one(xₛ); config = options, kwargs...)
    α3 = method(_f, _d, 2 * one(xₛ); config = options, kwargs...)

    @test compute_new_iterate(x₀, α1, δx) ≈ xₛ  atol=∛(2eps())
    @test compute_new_iterate(x₀, α2, δx) ≈ xₛ  atol=∛(2eps())
    @test compute_new_iterate(x₀, α3, δx) ≈ xₛ  atol=∛(2eps())

end

@testset "$(rpad("Bracketing",80))" begin
    @test bracket_minimum(x -> x^2) == (-SimpleSolvers.DEFAULT_BRACKETING_s, +SimpleSolvers.DEFAULT_BRACKETING_s)
    @test bracket_minimum(x -> (x-1)^2) == (0.64, 2.56)
end


@testset "$(rpad("Static",80))" begin
    x₀ = -3.
    x₁ = +1.
    δx = x₁ - x₀
    x  = copy(x₀)


    o  = UnivariateObjective(f, x)
    ls = StaticState()

    @test ls == LinesearchState(Static())
    @test ls == LinesearchState(Static(1.0))

    # x1 = copy(x₀); x2 = copy(x₀); @test solve!(x1, δx, ls) == ls(x2, δx) == x₁
    @test ls() == 1.


    o1  = UnivariateObjective(f, x)
    o2  = UnivariateObjective(f, g, x)
    
    ls1 = Linesearch(algorithm = Static())
    ls2 = Linesearch(algorithm = Static(1.0))
    ls3 = Linesearch(algorithm = Static(0.8))

    # @test ls1 == Linesearch(x, F; algorithm = Static())
    # @test ls2 == Linesearch(x, F; algorithm = Static(), D = D)

    # x1 = copy(x₀); x2 = copy(x₀); @test solve!(x1, δx, ls1) == ls1(x2, δx) == x₁
    # x1 = copy(x₀); x2 = copy(x₀); @test solve!(x1, δx, ls2) == ls2(x2, δx) == x₁
    # x1 = copy(x₀); x2 = copy(x₀); @test solve!(x1, δx, ls3) == ls3(x2, δx) == x₁

    @test ls1(o1) == ls1(o2) == ls1(f,g) == 1
    @test ls2(o1) == ls2(o2) == ls2(f,g) == 1
    @test ls3(o1) == ls3(o2) == ls3(f,g) == 0.8

end



@testset "$(rpad("Bisection",80))" begin

    x₀ = -3.0
    x₁ = +0.5
    δx = x₁ .- x₀
    x  = copy(x₀)
    o  = UnivariateObjective(f, x)

    x1 = bisection(f, x₀, x₁; config = Options(x_abstol = 10eps(x)))
    x2 = bisection(f, x; config = Options(x_abstol = 10eps(x)))

    @test x1 ≈ -1  atol=∛(10eps())
    @test x2 ≈ -1  atol=∛(10eps())

    x1 = bisection(f, x₀, x₁; config = Options(f_abstol = 2eps(f(x))))
    x2 = bisection(f, x; config = Options(f_abstol = 2eps(f(x))))
    
    @test x1 ≈ -1  atol=10eps()
    @test x2 ≈ -1  atol=10eps()


    # x₀ = [-3.0]
    # x₁ = [+0.5]
    # xₛ = [-1.0]
    # δx = x₁ .- x₀
    # x  = copy(x₀)

    x_abstol = 10eps(eltype(x))
    f_abstol = 10eps(F(x))

    ls = Linesearch(algorithm = Bisection(), config = Options(x_abstol = x_abstol))

    @test ( (ls(o) ≈ ls(f, g) ≈ 1.0) || (ls(o) ≈ ls(f, g) ≈ -1.0) )

    # x1 = copy(x₀)
    # x2 = copy(x₀)
    # x3 = copy(x₀)

    # ls(x1, δx)
    # solve!(x2, δx, ls)
    # solve!(x3, δx, ls.state)

    # @test x1 ≈ xₛ  atol=∛(10eps())
    # @test x2 ≈ xₛ  atol=∛(10eps())
    # @test x3 ≈ xₛ  atol=∛(10eps())


    ls = Linesearch(algorithm = Bisection(), config = Options(f_abstol = f_abstol))

    @test ( (ls(o) ≈ ls(f, g) ≈ 1.0) || (ls(o) ≈ ls(f, g) ≈ -1.0) )

    # x1 = copy(x₀)
    # x2 = copy(x₀)
    # x3 = copy(x₀)

    # ls(x1, δx)
    # solve!(x2, δx, ls)
    # solve!(x3, δx, ls.state)

    # @test x1 ≈ xₛ  atol=10eps()
    # @test x2 ≈ xₛ  atol=10eps()
    # @test x3 ≈ xₛ  atol=10eps()

end


@testset "$(rpad("Backtracking",80))" begin

    # test_linesearch(Backtracking(), backtracking)


    # x₀ = [-3.0]
    # x₁ = [+3.0]
    # xₛ = [ 0.0]
    # δx = x₁ .- x₀
    # x  = copy(x₀)

    # x1 = copy(x₀)
    # x2 = copy(x₀)
    # x3 = copy(x₀)

    # ls = Linesearch(x, F; algorithm = Backtracking(), config = Options(x_abstol = zero(eltype(x))))

    # ls(x1, δx)
    # solve!(x2, δx, ls)
    # backtracking(F, x3, δx; config = Options(x_abstol = zero(eltype(x))))

    # @test x1 ≈ xₛ  atol=∛(10eps())
    # @test x2 ≈ xₛ  atol=∛(10eps())
    # @test x3 ≈ xₛ  atol=∛(10eps())
    

    # n = 1
    # x₀ = -0.5*ones(n)
    # x₁ = +1.0*ones(n)
    # x  = copy(x₀)
    # f  = zeros(n)
    # g  = zeros(n,n)
    # ls = Solver(F!, x, f)
    
    # F!(f,x)
    # J!(g,x)

    # solve!(x, f, g, x₀, x₁, ls)

    # F!(f,x)

    # @test x ≈ zero(x) atol=4E-1
    # @test f ≈ zero(f) atol=8E-2

    # @test solve!(x, f, g, x₀, x₁, ls) == ls(x, f, g, x₀, x₁) == SFunc(F!, x, f, g, x₀, x₁)


    # n = 3
    # x₀ = -ones(n)
    # x₁ = +ones(n)
    # x  = copy(x₀)
    # f  = zeros(n)
    # g  = zeros(n,n)
    # ls = Solver(F!, x, f)
    
    # F!(f,x)
    # J!(g,x)

    # solve!(x, f, g, x₀, x₁, ls)

    # F!(f,x)

    # @test x == zero(x)
    # @test f == zero(f)

    # @test solve!(x, f, g, x₀, x₁, ls) == ls(x, f, g, x₀, x₁) == SFunc(F!, x, f, g, x₀, x₁)

end


@testset "$(rpad("Polynomial Linesearch",80))" begin

    test_linesearch(Quadratic(), quadratic; σ₀=0.5, σ₁=0.8)


    # x₀ = [-3.0]
    # x₁ = [+3.0]
    # xₛ = [ 0.0]
    # δx = x₁ .- x₀
    # x  = copy(x₀)

    # x1 = copy(x₀)
    # x2 = copy(x₀)
    # x3 = copy(x₀)

    # ls = Linesearch(x, F; algorithm = Quadratic(), config = Options(x_abstol = zero(eltype(x))))

    # ls(x1, δx)
    # solve!(x2, δx, ls)
    # quadratic(F, x3, δx; config = Options(x_abstol = zero(eltype(x))))

    # @test x1 ≈ xₛ  atol=∛(10eps())
    # @test x2 ≈ xₛ  atol=∛(10eps())
    # @test x3 ≈ xₛ  atol=∛(10eps())
    

    # n = 1
    # x₀ = -0.5*ones(n)
    # x₁ = +1.0*ones(n)
    # x  = copy(x₀)
    # f  = zeros(n)
    # g  = zeros(n,n)
    # ls = Solver(F!, x, f)
    
    # F!(f,x)
    # J!(g,x)

    # solve!(x, f, g, x₀, x₁, ls)

    # F!(f,x)

    # @test x ≈ zero(x) atol=4E-1
    # @test f ≈ zero(f) atol=8E-2

    # @test solve!(x, f, g, x₀, x₁, ls) == ls(x, f, g, x₀, x₁) == SFunc(F!, x, f, g, x₀, x₁)


    # n = 3
    # x₀ = -ones(n)
    # x₁ = +ones(n)
    # x  = copy(x₀)
    # f  = zeros(n)
    # g  = zeros(n,n)
    # ls = Solver(F!, x, f)
    
    # F!(f,x)
    # J!(g,x)

    # solve!(x, f, g, x₀, x₁, ls)

    # F!(f,x)

    # @test x == zero(x)
    # @test f == zero(f)

    # @test solve!(x, f, g, x₀, x₁, ls) == ls(x, f, g, x₀, x₁) == SFunc(F!, x, f, g, x₀, x₁)

end
