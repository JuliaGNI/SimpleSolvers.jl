using Profile
using SimpleSolvers


function F!(f, x, params)
    f .= tan.(x)
    nothing
end

# function F!(f, x)
#     f .= x.^2
#     nothing
# end


function profile(Solver, kwarguments, n=100)
    x = ones(n)
    y = zero(x)
    nl = Solver(x, y; F=F!, kwarguments...)

    solve!(x, nl)

    x = ones(n)

    Profile.clear()
    Profile.clear_malloc_data()

    Profile.Allocs.@profile solve!(x, nl)
end

profile(NewtonSolver, (linesearch=Static(),))
# profile(NewtonSolver, (linesearch=Backtracking(),))
# profile(NewtonSolver, (linesearch=Quadratic(),))
# profile(NewtonSolver, (linesearch=BierlaireQuadratic(),))
# profile(NewtonSolver, (linesearch=Bisection(),))
# profile(FixedPointIterator, (linesearch = Static(),))
# profile(FixedPointIterator, (linesearch = Backtracking(),))
# profile(FixedPointIterator, (linesearch = Quadratic(),))
# profile(FixedPointIterator, (linesearch = BierlaireQuadratic(),))
# profile(FixedPointIterator, (linesearch = Bisection(),))
