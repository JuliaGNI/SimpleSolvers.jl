using SimpleSolvers
using SimpleSolvers: compute_jacobian!, factorize!, linearsolver, jacobian, cache, linesearch_objective, direction, LinesearchState, Quadratic2
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

function make_linesearch_objective(x::AbstractVector, params=nothing)
    solver = NewtonSolver(x, f.(x); F = f!)
    update!(solver, x, params)
    compute_jacobian!(solver, x, j!, params; mode = :function)

    # compute rhs
    f!(cache(solver).rhs, x, params)
    rmul!(cache(solver).rhs, -1)

    # multiply rhs with jacobian
    factorize!(linearsolver(solver), jacobian(solver))
    ldiv!(direction(cache(solver)), linearsolver(solver), cache(solver).rhs)

    nls = NonlinearSystem(f!, x, f.(x))
    linesearch_objective(nls, cache(solver), params)
end

function check_linesearch(ls::LinesearchState, ls_obj::TemporaryUnivariateObjective)
    α = ls(ls_obj)
    T = eltype(α)
    @test ≈(ls_obj.D(α), zero(T); atol = atol=∛(2eps(T)))
end

for T ∈ (Float32, Float64)
    for ls_method ∈ (Bisection(), Quadratic2(), BierlaireQuadratic())
        ls = LinesearchState(ls_method; T = T)
        ls_obj = make_linesearch_objective(T.(x))
        check_linesearch(ls, ls_obj)
    end
end