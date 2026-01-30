using Profile
using SimpleSolvers

const T = Float64
# const T = Float32

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
    ss = NonlinearSolverState(x, y)

    solve!(x, nl, ss)

    x = ones(n)

    Profile.clear()
    Profile.clear_malloc_data()

    Profile.Allocs.@profile solve!(x, nl, ss)
end

# profile(NewtonSolver, (linesearch=Static(T),))
# profile(NewtonSolver, (linesearch=Backtracking(T),))
# profile(NewtonSolver, (linesearch=Quadratic(T, NewtonMethod()),))
# profile(NewtonSolver, (linesearch=BierlaireQuadratic(T),))
# profile(NewtonSolver, (linesearch=Bisection(T),))
# profile(FixedPointIterator, (linesearch=Static(T),))
# profile(FixedPointIterator, (linesearch=Backtracking(T),))
# profile(FixedPointIterator, (linesearch=Quadratic(T, NewtonMethod()),))
# profile(FixedPointIterator, (linesearch=BierlaireQuadratic(T),))
# profile(FixedPointIterator, (linesearch=Bisection(T),))
