using SimpleSolvers, Test
using Random

Random.seed!(123)

function check_value_for_univariate_objective(T::DataType)
    f(x::Number) = x ^ 2
    x = rand(T)
    obj = UnivariateObjective(f, x)
    y = value(obj, x)
    statement_we_should_have = 
    "UnivariateObjective:

    f(x)              = NaN 
    d(x)              = NaN 
    x_f               = NaN 
    x_d               = NaN 
    number of f calls = 1 
    number of d calls = 0 \n"
    io = IOBuffer()
    show(io, obj)
    statement_we_have = String(take!(io))
    @test statement_we_have == statement_we_should_have
end

function check_value_for_multivariate_objective(T::DataType)
    f(x::AbstractArray) = sum(x .^ 2)
    x = rand(T, 3)
    
    obj = MultivariateObjective(f, x)
    statement_we_should_have = 
    "MultivariateObjective (for vector-valued quantities only the first component is printed):

    f(x)              = NaN 
    g(x)₁             = NaN 
    x_f₁              = NaN 
    x_g₁              = NaN 
    number of f calls = 0 
    number of g calls = 0 \n"
    io = IOBuffer()
    show(io, obj)
    statement_we_have = String(take!(io))
    @test statement_we_have == statement_we_should_have
end

for T in (Float32, Float64)
    check_value_for_univariate_objective(T)
    check_value_for_multivariate_objective(T)
end