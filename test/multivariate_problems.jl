using SimpleSolvers
using Test

function F(x)
    1 + sum(x.^2)
end

function G!(g, x)
    g .= 0
    for i in eachindex(x,g)
        g[i] = 2x[i]
    end
end

const n = 2
const x = rand(n)
const f = F(x)
const g = SimpleSolvers.alloc_g(x)

const f_gradient = GradientFunction(x)
const ad_gradient = GradientAutodiff(F, x)

G!(g, x)

obj1 = OptimizerProblem(F, zero(x))
obj2 = OptimizerProblem(F, G!, zero(x))

# test if the correct value is returned and if the counter goes up
function return_correct_value(obj1::OptimizerProblem, obj2::OptimizerProblem, x::AbstractVector, y::Number)
    @test value(obj1, x) == value(obj2, x) == y
end

# similar to above, but inplace.
function return_correct_value_inplace(obj1::OptimizerProblem, obj2::OptimizerProblem, x::AbstractVector, y::Number)
    @test value!(obj1, x) == value!(obj2, x) == y
    @test value(obj1) == value(obj2) == y
end

# test value-related functionality (clear Objective object after every run)
for (x_temp, y_temp) ∈ zip((x, 2x), (f, F(2x)))
    return_correct_value(obj1, obj2, x_temp, y_temp)
    return_correct_value_inplace(obj1, obj2, x_temp, y_temp)
    SimpleSolvers.clear!(obj1)
    SimpleSolvers.clear!(obj2)
end

function return_correct_gradients(obj1::OptimizerProblem, obj2::OptimizerProblem, x::AbstractVector, z::AbstractVector)
    @test_throws "There is no analytic gradient stored in the problem!" gradient!(obj1, f_gradient, x)
    @test gradient!(obj1, ad_gradient, x) == gradient!(obj2, f_gradient, x) == z
end

# test gradient-related functionality (clear Objective object after every run)
for (x_temp, z_temp) ∈ zip((x, 2x), (g, 4x))
    return_correct_gradients(obj1, obj2, x_temp, z_temp)
    SimpleSolvers.clear!(obj1)
    SimpleSolvers.clear!(obj2)
end