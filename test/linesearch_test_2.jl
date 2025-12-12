using SimpleSolvers
using SimpleSolvers: factorize!, linearsolver, jacobian, jacobian!, cache, linesearch_problem, direction, LinesearchState, Quadratic
using LinearAlgebra: rmul!, ldiv!
using Test
using Random
Random.seed!(1234)

f(x::T) where {T<:Number} = exp(x) * (T(.5) * x ^ 3 - 5x ^ 2 + 2x) + 2one(T)
f(x::AbstractArray{T}) where {T<:Number} = exp.(x) .* (T(.5) * (x .^ 3) - 5 * (x .^ 2) + 2x) .+ 2one(T)
f!(y::AbstractVector{T}, x::AbstractVector{T}, params) where {T} = y .= f.(x)
function j!(j::AbstractMatrix{T}, x::AbstractVector{T}, params) where {T}
    f_closure!(y, x) = f!(y, x, params)
    SimpleSolvers.ForwardDiff.jacobian!(j, f_closure!, similar(x), x)
end
x = -10 * rand(1)

function make_linesearch_problem(x::AbstractVector{T}, params=nothing) where {T}
    jacobian_instance = JacobianFunction{T}(f!, j!)
    solver = NewtonSolver(x, f.(x); F = f!, DF! = j!, jacobian = jacobian_instance)
    update!(solver, x, params)
    jacobian!(solver, x, params)

    # compute rhs
    f!(cache(solver).rhs, x, params)
    rmul!(cache(solver).rhs, -1)

    # multiply rhs with jacobian
    factorize!(linearsolver(solver), jacobian(solver))
    ldiv!(direction(cache(solver)), linearsolver(solver), cache(solver).rhs)

    nlp = NonlinearProblem(f!, j!, x, f.(x))
    linesearch_problem(nlp, jacobian_instance, cache(solver), params)
end

function check_linesearch(ls::LinesearchState, ls_obj::LinesearchProblem)
    α = ls(ls_obj)
    T = eltype(α)
    @test ≈(ls_obj.D(α), zero(T); atol = atol=∛(2eps(T)))
end

for T ∈ (Float32, Float64)
    for ls_method ∈ (Bisection(), Quadratic(), BierlaireQuadratic())
        ls = LinesearchState(ls_method; T = T)
        ls_obj = make_linesearch_problem(T.(x))
        check_linesearch(ls, ls_obj)
    end
end
