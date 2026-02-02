using SimpleSolvers


function F!(f, x, params)
    f .= x .^ 2 # .+ .05 * x .^ 3
    nothing
end


function test(n)
    for T in (Float64, Float32)
        for (Solver, kwarguments) in (
            (NewtonSolver, (linesearch=Static(T),)),
            (NewtonSolver, (linesearch=Backtracking(T),)),
            (NewtonSolver, (linesearch=Quadratic(T, NewtonMethod()),)),
            (NewtonSolver, (linesearch=BierlaireQuadratic(T),)),
            (NewtonSolver, (linesearch=Bisection(T),)),
            (FixedPointIterator, (linesearch=Static(T),)),
            (FixedPointIterator, (linesearch=Backtracking(T),)),
            (FixedPointIterator, (linesearch=Quadratic(T, NewtonMethod()),)),
            (FixedPointIterator, (linesearch=BierlaireQuadratic(T),)),
            (FixedPointIterator, (linesearch=Bisection(T),)),
        )

            x = ones(T, n)
            y = zero(x)
            nl = Solver(x, y; F=F!, verbosity=2, kwarguments...)
            ss = SolverState(nl)

            println(Solver, ", ", kwarguments, ", ", T, "\n")

            x = ones(T, n)
            solve!(x, nl, ss)

            x = ones(T, n)
            @time solve!(x, nl, ss)

            println()
        end
    end
end

test(2)
# test(200)
