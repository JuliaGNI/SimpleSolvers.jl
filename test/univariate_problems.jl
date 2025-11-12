using SimpleSolvers
using Test

f = x -> 1 + x^2
d = x -> 2x

const x = Float64(π)
const y = f(x)
const z = d(x)

obj1 = UnivariateProblem(f, x)
obj2 = UnivariateProblem(f, d, x)
obj3 = LinesearchProblem(f, d, x)

function f_d_calls_initialization(obj1::UnivariateProblem, obj2::UnivariateProblem)
    @test f_calls(obj1) == f_calls(obj2) == 0
    @test d_calls(obj1) == d_calls(obj2) == 0
end

# test if the correct value is returned and if the counter goes up
function return_correct_value(obj1::UnivariateProblem, obj2::UnivariateProblem, obj3::LinesearchProblem, x::Number, y::Number)
    @test value(obj1, x) == value(obj2, x) == value(obj3, x) == y
    @test f_calls(obj1) == f_calls(obj2) == 1
    @test d_calls(obj1) == d_calls(obj2) == 0
end

# similar to above, but inplace.
function return_correct_value_inplace(obj1::UnivariateProblem, obj2::UnivariateProblem, obj3::LinesearchProblem, x::Number, y::Number)
    @test value!(obj1, x) == value!(obj2, x) == value!(obj3, x) == y
    @test value(obj1) == value(obj2) == value(obj3, x) == y
    @test f_calls(obj1) == f_calls(obj2) == 2
    @test d_calls(obj1) == d_calls(obj2) == 0
end

# test value-related functionality (clear Objective object after every run)
for (x_temp, y_temp) ∈ zip((x, 2x), (y, f(2x)))
    f_d_calls_initialization(obj1, obj2)
    return_correct_value(obj1, obj2, obj3, x_temp, y_temp)
    return_correct_value_inplace(obj1, obj2, obj3, x_temp, y_temp)
    SimpleSolvers.clear!(obj1)
    SimpleSolvers.clear!(obj2)
end

function return_correct_derivatives(obj1::UnivariateProblem, obj2::UnivariateProblem, obj3::LinesearchProblem, x::Number, z::Number)
    @test derivative(obj1, x) == derivative(obj2, x) == derivative(obj3, x) == z
    @test f_calls(obj1) == f_calls(obj2) == 0
    @test d_calls(obj1) == d_calls(obj2) == 1
end

function return_correct_derivatives_inplace(obj1::UnivariateProblem, obj2::UnivariateProblem, obj3::LinesearchProblem, x::Number, z::Number)
@test derivative!(obj1, x) == derivative!(obj2, x) == derivative!(obj3, x) == z
@test f_calls(obj1) == f_calls(obj2) == 0
@test d_calls(obj1) == d_calls(obj2) == 2
end

# test derivative-related functionality (clear Objective object after every run)
for (x_temp, z_temp) ∈ zip((x, 2x), (z, 4x))
    f_d_calls_initialization(obj1, obj2)
    return_correct_derivatives(obj1, obj2, obj3, x_temp, z_temp)
    return_correct_derivatives_inplace(obj1, obj2, obj3, x_temp, z_temp)
    SimpleSolvers.clear!(obj1)
    SimpleSolvers.clear!(obj2)
end