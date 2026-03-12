using Random
using SimpleSolvers
using Test

using LinearAlgebra: rmul!, ldiv!
using SimpleSolvers: AbstractOptimizerProblem, BierlaireQuadratic, Quadratic, NullParameters
using SimpleSolvers: factorize!, linearsolver, jacobian, jacobian!, cache, linesearch_problem, direction, compute_new_iterate, direction!, nonlinearproblem, iteration_number

f(x) = x^2 - 1
g(x) = 2x
δx(x) = -g(x) / 2

function compute_next_iterate(ls::Linesearch, x₀::T) where {T}
    α = solve(ls, 1.0, (x = x₀,))
    compute_new_iterate(x₀, α, δx(x₀))
end

function compute_next_iterate(ls::Linesearch, x₀::T, n::Integer) where {T}
    x = x₀
    for _ in 1:n
        x = compute_next_iterate(ls, x)
    end
    x
end

function make_linesearch_problem(x₀::Number)
    _f(α, _) = f(compute_new_iterate(x₀, α, δx(x₀)))
    _d(α, _) = g(compute_new_iterate(x₀, α, δx(x₀)))
    LinesearchProblem{typeof(x₀)}(_f, _d)
end

function test_linesearch(method::LinesearchMethod, n::Integer=1)
    x₀ = -3.0
    x₁ = +3.0
    xₛ = 0.0

    ls = Linesearch(make_linesearch_problem(x₀), method; x_abstol=zero(x₀))

    @test compute_next_iterate(ls, x₀, n) ≈ xₛ atol = ∛(2eps())
    @test compute_next_iterate(ls, x₀, n) ≈ xₛ atol = ∛(2eps())
end


@testset "$(rpad("Bracketing",80))" begin
    @test bracket_minimum(x -> x^2, 0.0) == (-SimpleSolvers.DEFAULT_BRACKETING_s, +SimpleSolvers.DEFAULT_BRACKETING_s)
    @test bracket_minimum(x -> (x - 1)^2, 0.0) == (0.64, 2.56)
end

@testset "$(rpad("Static",80))" begin
    x₀ = -3.0
    x₁ = +3.0
    δx = x₁ - x₀
    x = copy(x₀)

    ls_problem = make_linesearch_problem(x₀)

    @test Linesearch(ls_problem, Static()) == Linesearch(ls_problem, Static(1.0))

    ls1 = Linesearch(ls_problem, Static())
    ls2 = Linesearch(ls_problem, Static(1.0))
    ls3 = Linesearch(ls_problem, Static(0.8))

    @test solve(ls1, 0.0) == 1.0
    @test solve(ls2, 0.0) == 1.0
    @test solve(ls3, 0.0) == 0.8

end

@testset "$(rpad("Bisection", 80))" begin

    test_linesearch(Bisection(), 1)

end

@testset "$(rpad("Backtracking", 80))" begin

    test_linesearch(Backtracking(), 20)

end

@testset "$(rpad("Quadratic Linesearch (Bierlaire)", 80))" begin

    test_linesearch(BierlaireQuadratic(), 1)

end

@testset "$(rpad("Quadratic Linesearch (Derivative-Based)", 80))" begin

    test_linesearch(BierlaireQuadratic(), 1)

end


@testset "$(rpad("Linesearch Integration Tests", 80))" begin

    Random.seed!(1234)

    x = -10 * rand(1)

    function linesearch_factory(x::AbstractVector{T}, params) where {T}
        f(x::T) where {T<:Number} = exp(x) * (T(0.5) * x^3 - 5x^2 + 2x) + 2one(T)
        f(x::AbstractArray{T}) where {T<:Number} = @. exp(x) * (T(0.5) * x^3 - 5 * x^2 + 2x) + 2one(T)
        f!(y::AbstractVector{T}, x::AbstractVector{T}, params) where {T} = y .= f.(x)

        function j!(j::AbstractMatrix{T}, x::AbstractVector{T}, params) where {T}
            f_closure!(y, x) = f!(y, x, params)
            SimpleSolvers.ForwardDiff.jacobian!(j, f_closure!, similar(x), x)
        end

        jacobian = JacobianFunction{T}(f!, j!)
        solver = NewtonSolver(x, f.(x); F=f!, (DF!)=j!, jacobian=jacobian)
        state = NonlinearSolverState(x, value(cache(solver)))

        direction!(solver, x, params, iteration_number(state))
        # update!(state, x, value(cache(solver)))

        linesearch_problem(solver)
    end

    function check_linesearch(T, ls_method)
        params = (x=T.(x), parameters=NullParameters())
        ls = Linesearch(linesearch_factory(params.x, params.parameters), ls_method)
        α = solve(ls, one(T), params)
        @test ≈(problem(ls).D(α, params), zero(T); atol=(∛(2eps(T))))
    end

    for T ∈ (Float32, Float64)
        for ls_method ∈ (Bisection(T), Quadratic(T), BierlaireQuadratic(T))
            check_linesearch(T, ls_method)
        end
    end
end


@testset "$(rpad("Linesearch Conversion Tests", 80))" begin

    function allocate_linesearch_methods(T::DataType)
        st = Static(T; α=one(T))
        bt = Backtracking(T)
        qu = Quadratic(T; ε=T(1e-5)) # here this constant is specified manually as it otherwise depends on the DataType used
        bq = BierlaireQuadratic(T)
        bi = Bisection(T)
        st, bt, qu, bq, bi
    end

    function convert_linesearches_test(T₁::DataType, T₂::DataType; rtol=T₂(1e-3))
        st₁, bt₁, qu₁, bq₁, bi₁ = allocate_linesearch_methods(T₁)
        st₂, bt₂, qu₂, bq₂, bi₂ = allocate_linesearch_methods(T₂)

        @test ≈(st₂, convert(T₂, st₁); rtol=rtol)
        @test ≈(bt₂, convert(T₂, bt₁); rtol=rtol)
        @test ≈(qu₂, convert(T₂, qu₁); rtol=rtol)
        @test ≈(bq₂, convert(T₂, bq₁); rtol=rtol)
        @test ≈(bi₂, convert(T₂, bi₁); rtol=rtol)

        nothing
    end

    convert_linesearches_test(Float32, Float64)
    convert_linesearches_test(Float64, Float32)

end
