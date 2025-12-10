using Profile
using SimpleSolvers


function F!(f, x, params)
    f .= tan.(x)
    nothing
end


function profile(Solver, kwarguments, n=100)
    x = ones(n)
    y = zero(x)
    nl = Solver(x, y; F=F!, kwarguments...)

    solve!(nl, x)

    x = ones(n)

    Profile.clear()
    Profile.clear_malloc_data()

    Profile.Allocs.@profile solve!(nl, x)
end

profile(NewtonSolver, (linesearch=Static(),))
# profile(NewtonSolver, (linesearch=Backtracking(),))
# profile(NewtonSolver, (linesearch=Quadratic2(),))
# profile(NewtonSolver, (linesearch=BierlaireQuadratic(),))
# profile(NewtonSolver, (linesearch=Bisection(),))
# profile(QuasiNewtonSolver, (linesearch = Static(),))
# profile(QuasiNewtonSolver, (linesearch = Backtracking(),))
# profile(QuasiNewtonSolver, (linesearch = Quadratic2(),))
# profile(QuasiNewtonSolver, (linesearch = BierlaireQuadratic(),))
# profile(QuasiNewtonSolver, (linesearch = Bisection(),))




# using Profile
# using ProfileCanvas
# using SimpleSolvers


# function F!(f, x)
#     f .= x.^2
#     nothing
# end

# Solver = NewtonSolver
# kwarguments = (linesearch = Backtracking(),)

# n = 100
# x = ones(n)
# y = zero(x)
# nl = Solver(x, y; kwarguments...)

# solve!(x, F!, nl)

# x = ones(n)

# Profile.clear()
# Profile.clear_malloc_data()

# @profview solve!(x, F!, nl)
