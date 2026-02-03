using SimpleSolvers
using SimpleSolvers: l2norm

f(x::T) where {T} = 5(abs(x - T(2)))^3 - (one(T) / abs(x))
F(x::AbstractVector) = sum(f.(x))

function test_nan_handling_for_linesearches(n::Integer, ::Type{T}, η=T(10); kwargs...) where {T}
    η = T(η)
    x = 3ones(T, n)
    opt = Optimizer(x, F; algorithm=Newton(), linesearch=Static(η), verbosity=2, kwargs...)
    state = OptimizerState(Newton(), x)
    solve!(x, state, opt)
    println(x)
end

test_nan_handling_for_linesearches(1, Float64, 2.0; max_iterations=5)
@test_warn "NaN or Inf detected in nonlinear solver. Reducing length of direction vector." test_nan_handling_for_linesearches(1, Float64, 5.9411764705882355; max_iterations=5)
