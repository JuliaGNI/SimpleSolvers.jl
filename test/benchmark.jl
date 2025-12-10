using SimpleSolvers


function F!(f, x, params)
    f .= x .^ 2
    nothing
end


function test(n)
    for (Solver, kwarguments) in (
        (NewtonSolver, (linesearch=Static(),)),
        (NewtonSolver, (linesearch=Backtracking(),)),
        (NewtonSolver, (linesearch=Quadratic2(),)),
        (NewtonSolver, (linesearch=BierlaireQuadratic(),)),
        (NewtonSolver, (linesearch=Bisection(),)),
    )

        for T in (Float64, Float32)
            x = ones(T, n)
            y = zero(x)
            nl = Solver(x, y; F=F!, kwarguments...)

            println(Solver, ", ", kwarguments, ", ", T)

            x = ones(T, n)
            solve!(x, nl)

            x = ones(T, n)
            @time solve!(x, nl)

            println()
        end
    end
end

test(200)
