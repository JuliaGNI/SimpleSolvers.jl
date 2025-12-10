using SimpleSolvers, Test
using Random

Random.seed!(123)

function check_value_for_nonlinearsolverstatus(T::DataType)
    f(x::Number) = x^2
    f(x::AbstractArray) = f.(x)
    F(y::AbstractArray, x::AbstractArray, params) = y .= f(x)
    x = rand(T, 3)
    
    # s₁ = NewtonSolver(x₁, f)
    s = NewtonSolver(x, F, f(x))
    expected_statement = 
    "i=   0,
x= NaN,
f= NaN,
rxₐ= NaN,
rfₐ= NaN"
    compare_statements(s, expected_statement)
end

function compare_statements(s::NewtonSolver, expected_statement::String)
    io = IOBuffer()
    show(io, s)
    statement_we_have = String(take!(io))
    @test statement_we_have == expected_statement
end

for T in (Float32, Float64)
    check_value_for_nonlinearsolverstatus(T)
end