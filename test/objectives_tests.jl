
using SimpleSolvers
using Test


f = x -> 1 + x^2
d = x -> 2x

x = Float64(π)
y = f(x)
z = d(x)


obj1 = UnivariateObjective(f, x)
obj2 = UnivariateObjective(f, d, x)

@test f_calls(obj1) == f_calls(obj2) == 0
@test d_calls(obj1) == d_calls(obj2) == 0

@test value(obj1, x) == value(obj2, x) == y
@test f_calls(obj1) == f_calls(obj2) == 1
@test d_calls(obj1) == d_calls(obj2) == 0

@test value!(obj1, x) == value!(obj2, x) == y
@test f_calls(obj1) == f_calls(obj2) == 2
@test d_calls(obj1) == d_calls(obj2) == 0

@test value!(obj1, x) == value!(obj2, x) == y
@test f_calls(obj1) == f_calls(obj2) == 2
@test d_calls(obj1) == d_calls(obj2) == 0

@test value!!(obj1, x) == value!!(obj2, x) == y
@test f_calls(obj1) == f_calls(obj2) == 3
@test d_calls(obj1) == d_calls(obj2) == 0

@test value!(obj1, 2x) == value!(obj2, 2x)
@test f_calls(obj1) == f_calls(obj2) == 4
@test d_calls(obj1) == d_calls(obj2) == 0

SimpleSolvers.clear!(obj1)
SimpleSolvers.clear!(obj2)

@test f_calls(obj1) == f_calls(obj2) == 0
@test d_calls(obj1) == d_calls(obj2) == 0

@test derivative(obj1, x) == derivative(obj2, x) == z
@test f_calls(obj1) == f_calls(obj2) == 0
@test d_calls(obj1) == d_calls(obj2) == 1

@test derivative!(obj1, x) == derivative!(obj2, x) == z
@test f_calls(obj1) == f_calls(obj2) == 0
@test d_calls(obj1) == d_calls(obj2) == 2

@test derivative!(obj1, x) == derivative!(obj2, x) == z
@test f_calls(obj1) == f_calls(obj2) == 0
@test d_calls(obj1) == d_calls(obj2) == 2

@test derivative!!(obj1, x) == derivative!!(obj2, x) == z
@test f_calls(obj1) == f_calls(obj2) == 0
@test d_calls(obj1) == d_calls(obj2) == 3

@test derivative!(obj1, 2x) == derivative!(obj2, 2x)
@test f_calls(obj1) == f_calls(obj2) == 0
@test d_calls(obj1) == d_calls(obj2) == 4



function F(x)
    1 + sum(x.^2)
end

function G!(g, x)
    g .= 0
    for i in eachindex(x,g)
        g[i] = 2x[i]
    end
end

n = 2
x = rand(n)
f = F(x)
g = SimpleSolvers.alloc_g(x)
G!(g, x)


obj1 = MultivariateObjective(F, x)
obj2 = MultivariateObjective(F, G!, x)

@test f_calls(obj1) == f_calls(obj2) == 0
@test g_calls(obj1) == g_calls(obj2) == 0

@test value(obj1, x) == value(obj2, x) == f
@test f_calls(obj1) == f_calls(obj2) == 1
@test g_calls(obj1) == g_calls(obj2) == 0


@test value!(obj1, x) == value!(obj2, x) == f
@test f_calls(obj1) == f_calls(obj2) == 2
@test g_calls(obj1) == g_calls(obj2) == 0

@test value!(obj1, x) == value!(obj2, x) == f
@test f_calls(obj1) == f_calls(obj2) == 2
@test g_calls(obj1) == g_calls(obj2) == 0

@test value!!(obj1, x) == value!!(obj2, x) == f
@test f_calls(obj1) == f_calls(obj2) == 3
@test g_calls(obj1) == g_calls(obj2) == 0

@test value!(obj1, 2x) == value!(obj2, 2x)
@test f_calls(obj1) == f_calls(obj2) == 4
@test g_calls(obj1) == g_calls(obj2) == 0

SimpleSolvers.clear!(obj1)
SimpleSolvers.clear!(obj2)

@test f_calls(obj1) == f_calls(obj2) == 0
@test g_calls(obj1) == g_calls(obj2) == 0

@test gradient(obj1, x) == gradient(obj2, x) == g
@test f_calls(obj1) == f_calls(obj2) == 0
@test g_calls(obj1) == g_calls(obj2) == 1

@test gradient!(obj1, x) == gradient!(obj2, x) == g
@test f_calls(obj1) == f_calls(obj2) == 0
@test g_calls(obj1) == g_calls(obj2) == 2

@test gradient!(obj1, x) == gradient!(obj2, x) == g
@test f_calls(obj1) == f_calls(obj2) == 0
@test g_calls(obj1) == g_calls(obj2) == 2

@test gradient!!(obj1, x) == gradient!!(obj2, x) == g
@test f_calls(obj1) == f_calls(obj2) == 0
@test g_calls(obj1) == g_calls(obj2) == 3

@test gradient!(obj1, 2x) == gradient!(obj2, 2x)
@test f_calls(obj1) == f_calls(obj2) == 0
@test g_calls(obj1) == g_calls(obj2) == 4
