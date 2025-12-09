using SimpleSolvers


function F!(f, x)
    f .= x.^2
    nothing
end


function test(n)
    for (Solver, kwarguments) in (
                    (NewtonSolver, (linesearch = Static(),)),
                    (NewtonSolver, (linesearch = Backtracking(),)),
                    (NewtonSolver, (linesearch = Quadratic(),)),
                    (NewtonSolver, (linesearch = Bisection(),)),
                    (QuasiNewtonSolver, (linesearch = Static(),)),
                    (QuasiNewtonSolver, (linesearch = Backtracking(),)),
                    (QuasiNewtonSolver, (linesearch = Quadratic(),)),
                    (QuasiNewtonSolver, (linesearch = Bisection(),)),
                )

        for T in (Float64, Float32)
            x = ones(T, n)
            y = zero(x)
            nl = Solver(x, y; kwarguments...)

            println(Solver, ", ", kwarguments, ", ", T)

            x = ones(T, n)
            solve!(x, F!, nl)

            x = ones(T, n)
            @time solve!(x, F!, nl)

            println()
        end
    end
end

test(200)
